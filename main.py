"""GTA VI trophy earn-rate forecaster for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its Exophase earn rate (percent of tracked players who hold it)
at the case's target time, from the house's recorded reads of the trophy page
before the case's checkpoint.

Model in one paragraph
----------------------
An earn rate is a ratio, rate = earners / tracked players, and the two sides
move in opposite directions after a launch: the tracked-player denominator
grows as more players sync their profiles (which dilutes every rate), while
the earner numerator grows as the existing players push further into the
game (which raises every rate).  So the model splits the ratio and forecasts
each side separately, in log time since release:

* Tracked players N(t): one power law fitted to the whole history's distinct
  player counts, ln N = c + bN * ln t.  Denominator growth is shared by every
  trophy, so it is fitted once, globally, where the data is plentiful.
* Earners E(t) per trophy: a power law per trophy, ln(E + 0.5) = a + bE * ln t,
  fitted to that trophy's distinct reads, then shrunk toward a class prior
  slope (story chapters rise fast, grind trophies rise slowly).  The fit is
  anchored on the trophy's most recent distinct read, not on the fitted
  intercept, so a noisy early read cannot drag the forecast.
* The two extrapolations are divided back into a rate, which is then blended
  in logit space toward a class prior rate for week one (built from GTA V's
  lifetime trophy rates on public trackers and GTA VI's story length, see
  README.md).  The blend weight is higher when the trophy's own history is
  thin or the extrapolation runs far from the prior, so the data leads when
  it is informative and the prior catches it when it is not.
* The spread (logit standard deviation) grows with the horizon and with the
  fit's residual and with how far the trend runs from the prior, so the
  90% band is honest about a 4.5-day extrapolation.

One special case: a history in which the page never changed at all (launch
night, before Exophase's first refresh).  There is no trend to fit, and the
page is frozen between refreshes, so the forecast is a two-component mixture:
either no refresh lands before the target (the rate stays exactly where it
is) or one does (the denominator jumps and the rate is diluted).  The
refresh probability grows with how long the page has already been frozen,
and the dilution factor comes from the one refresh the kit's rules describe
(638 -> 3,868 tracked players, with new players earning at roughly 7% of the
launch-night cohort's rate).  The mixture's quantiles are computed by
inverting its cumulative distribution on a logit grid.

Only the Python standard library is used.  The program is deterministic:
it reads only the case files, uses no clock, no randomness and no network,
and writes the cases in name order.

Usage (local):  MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader:         python3 -P -s /bot/main.py   (reads /input, writes /output)
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release of GTA VI, the origin of the log-time axis (2026-11-19T00:00:00Z).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# A read stamped before release still counts as a few minutes in, so that
# log(hours since release) stays defined.
MIN_HOURS = 0.25
# Sanity ceiling on extrapolated tracked players: week-one PS5 sales are
# forecast at roughly 10-25 million copies and Exophase tracks only the
# players who sync a profile, a small fraction of buyers.
MAX_TRACKED = 2_000_000

# --- trophy classes -------------------------------------------------------
# Each class carries three numbers, all chosen by hand (see README.md for
# where they come from and how uncertain they are):
#   prior : the earn rate (percent) I expect at one week after release
#           (168 hours), among tracked players.
#   slope : the prior growth exponent of the earner count, E ~ t**slope,
#           used when a trophy's own history is too thin to fit one.
#   cap   : a loose ceiling the median forecast is never allowed above.
# Classes are matched by keywords in the trophy description, in order.
CLASSES = {
    # The opening mission everyone completes; it sat at 100% through the
    # first 6x jump in tracked players, so it is expected to stay there.
    "intro":      (99.9, 1.0, 100.0),
    "story_ch1":  (40.0, 2.0, 75.0),
    "story_ch2":  (30.0, 2.1, 65.0),
    "story_ch3":  (22.0, 2.2, 55.0),
    "story_ch4":  (16.0, 2.2, 45.0),
    "story_final": (10.0, 2.3, 35.0),
    # Named story missions (Breakfast Run, The Cartel Heist, ...).
    "mission":    (20.0, 1.8, 55.0),
    # Final mission of a stranger strand (Andy, Kenton, Billy, ...).
    "stranger":   (12.0, 1.7, 40.0),
    "completion": (0.10, 1.5, 0.5),
    "platinum":   (0.05, 1.5, 1.0),
    # Long grinds: 250 stunts, 100 WAiNK spots, 3 card sets, 50 challenges,
    # gold medals everywhere, 8 stranger strands, 5 bank vaults, every gang.
    "grind":      (1.0, 1.1, 8.0),
    # Repeatable activities: races, fishing, arcade, vehicles, money.
    "activity":   (8.0, 1.3, 30.0),
    "explore":    (6.0, 1.4, 25.0),
    # One-off feats: wanted-level escapes, robberies, stunts, set pieces.
    "feat":       (10.0, 1.2, 35.0),
}


def classify(trophy):
    """Assign a trophy to one of the classes above by its description."""
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    if tier == "platinum":
        return "platinum"
    if "100% completion" in desc:
        return "completion"
    if "gone fishing" in desc:
        return "intro"
    if "complete chapter 1" in desc:
        return "story_ch1"
    if "complete chapter 2" in desc:
        return "story_ch2"
    if "complete chapter 3" in desc:
        return "story_ch3"
    if "complete chapter 4" in desc:
        return "story_ch4"
    if "complete the final mission" in desc:
        return "stranger" if " for " in desc else "story_final"
    grind_marks = (
        "250 stunts", "gold medal in all", "drone race at every location",
        "8 mission strands", "3 collectible card sets", "waink it at 100",
        "meditate at every spot", "complete 50 challenges",
        "rob 5 bank vaults", "belonging to each gang",
    )
    if any(mark in desc for mark in grind_marks):
        return "grind"
    if desc.startswith("complete ") and "mission" in desc:
        return "mission"
    if desc.startswith("complete ") and ("breakfast run" in desc
            or "heist" in desc or "child's play" in desc
            or "headliners" in desc or "takedown" in desc
            or "savage glory" in desc):
        return "mission"
    if "explore all of leonida" in desc:
        return "explore"
    activity_marks = (
        "try each type of activity", "win 20 different races",
        "fishing challenge", "arcade game", "unique animal",
        "discover 100 vehicles", "laundering marked bills",
        "spend $100,000", "purchase all garage properties",
        "register 10 vehicles",
    )
    if any(mark in desc for mark in activity_marks):
        return "activity"
    return "feat"


# --- small numeric helpers ------------------------------------------------

def clamp_rate(pct):
    """Clamp a percent into the range the logit is defined on."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return 100.0 * ex / (1.0 + ex)


def round4(x):
    """Round to 4 decimals, half up, returning an int for whole numbers."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into hours since release."""
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    hours = (moment - RELEASE).total_seconds() / 3600.0
    return max(MIN_HOURS, hours)


def least_squares(xs, ys):
    """Ordinary least squares y = a + b*x; returns (a, b, residual_sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(
            sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --- history preparation --------------------------------------------------

def distinct_reads(case):
    """Collapse the history to one entry per distinct page state.

    Exophase refreshes a game's counts in jumps hours apart, so long runs of
    reads print the same page.  Only the first read showing each state
    carries timing information; the rest are duplicates.  Returns a list of
    (hours_since_release, players_tracked, rates) oldest first.
    """
    out = []
    seen = set()
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key in seen:
            continue
        seen.add(key)
        out.append((parse_time(read["at"]), read["players_tracked"],
                    list(read["rates"])))
    return out


def fit_tracked_growth(reads, t_last, t_target):
    """Fit the denominator's power law and extrapolate it to the target.

    Returns (n_target, exponent).  The exponent is fitted to the distinct
    player counts, using the most recent 48 hours when that leaves at
    least two points: early launch-night growth was far faster than the
    pace that holds two days in, and the old pace must not drag the
    extrapolation.  With fewer than two distinct counts it falls back to
    0.9, the rough pace of launch-week tracker growth.  The extrapolated
    count is capped, and so is the total growth factor.
    """
    points = [(math.log(t), math.log(n)) for t, n, _ in reads if n and n > 0]
    window = [(math.log(t), math.log(n)) for t, n, _ in reads
              if n and n > 0 and t >= t_last - 48.0]
    if len(window) >= 2:
        points = window
    if len(points) >= 2:
        _, beta, _ = least_squares([p[0] for p in points],
                                   [p[1] for p in points])
        beta = min(2.0, max(0.2, beta))
    else:
        beta = 0.9
    n_last = reads[-1][1]
    growth = (t_target / t_last) ** beta
    growth = min(growth, 8.0)
    return min(MAX_TRACKED, n_last * growth), beta


# --- the two forecast paths -----------------------------------------------

def forecast_trend(reads, index, trophy, t_last, t_target, n_target):
    """Forecast one trophy from its own history plus its class prior."""
    cls = classify(trophy)
    prior_week, slope_prior, cap = CLASSES[cls]
    horizon_days = (t_target - t_last) / 24.0

    # The class prior at the target: week-one prior, scaled down for an
    # earlier target along a saturating curve, never above the week-one value.
    if t_target >= 168.0:
        prior_rate = prior_week
    else:
        prior_rate = prior_week * min(1.0, (t_target / 168.0) ** 0.6)

    # Distinct (time, earners) points for this trophy.
    points = []
    for t, n, rates in reads:
        rate = rates[index]
        if rate is None or n is None:
            continue
        earners = n * rate / 100.0
        if not points or abs(points[-1][1] - earners) > 1e-9:
            points.append((t, earners))
    n_pts = len(points)

    last_rate = reads[-1][2][index]
    if last_rate is None:
        last_rate = prior_rate
    e_last = reads[-1][1] * last_rate / 100.0

    # Earner growth exponent: fitted where the trophy moved enough, shrunk
    # toward the class prior slope where it did not.  The fit uses the most
    # recent 48 hours when that leaves at least three points: the earliest
    # reads are a frozen launch-night page, and their flatness would drag
    # the slope down just when the players are speeding up.
    fit_points = [p for p in points if p[0] >= t_last - 48.0]
    if len(fit_points) < 3:
        fit_points = points
    if len(fit_points) >= 3:
        _, beta_fit, resid = least_squares(
            [math.log(t) for t, _ in fit_points],
            [math.log(e + 0.5) for _, e in fit_points])
        beta_fit = min(3.0, max(0.0, beta_fit))
        w_fit = len(fit_points) / (len(fit_points) + 3.0)
        beta = w_fit * beta_fit + (1.0 - w_fit) * slope_prior
        resid = min(resid, 0.8)
    elif n_pts == 2:
        beta_fit = (math.log(points[1][1] + 0.5)
                    - math.log(points[0][1] + 0.5)) / (
                        math.log(points[1][0]) - math.log(points[0][0]))
        beta_fit = min(3.0, max(0.0, beta_fit))
        beta = 0.5 * beta_fit + 0.5 * slope_prior
        resid = 0.0
    else:
        beta = slope_prior
        resid = 0.0

    # Extrapolate earners from the most recent read, divide by the
    # extrapolated denominator, and never let earners exceed players.
    e_target = e_last * (t_target / t_last) ** beta
    e_target = min(e_target, n_target)
    trend_rate = 100.0 * e_target / n_target if n_target > 0 else last_rate

    # Blend the trend with the class prior in logit space.  The prior gets
    # more weight when the trophy barely moved, and when the trend runs far
    # from the prior (an extreme extrapolation is not trusted on its own).
    # And the prior only speaks for week one: the earlier the target, the
    # less a week-one rate says about it, so its weight is scaled by how
    # far into the week the target sits.  (Without that scaling, a target
    # a few hours after launch would be dragged toward week-one rates
    # twenty times above the diluted reality.)
    if n_pts <= 2:
        w_prior = 0.55
    elif n_pts < 8:
        w_prior = 0.30
    else:
        w_prior = 0.18
    if abs(logit(trend_rate) - logit(prior_rate)) > 3.0:
        w_prior = max(w_prior, 0.5)
    w_prior *= min(1.0, max(0.0, t_target / 168.0))
    median_logit = ((1.0 - w_prior) * logit(trend_rate)
                    + w_prior * logit(prior_rate))
    median = inv_logit(median_logit)
    median = min(median, cap)
    median = clamp_rate(median)

    # Spread: a base that grows with the horizon, the fit's residual, a
    # term for how far the trend sits from the prior, and, over short
    # horizons, the lumpiness of discrete refreshes: the page only moves
    # when Exophase refreshes it, so a refresh landing inside a short
    # horizon can jump the rate in one step.  That lumpiness washes out
    # over a multi-day horizon, where many refreshes average out.
    spread = math.sqrt(0.30 ** 2 + (0.30 * math.sqrt(max(horizon_days, 0.0))) ** 2
                       + resid ** 2)
    if len(reads) >= 2:
        horizon_hours = max(0.0, t_target - t_last)
        spread = math.sqrt(spread ** 2
                           + (0.35 * math.exp(-horizon_hours / 24.0)) ** 2)
    spread += 0.12 * abs(logit(trend_rate) - logit(prior_rate))
    spread = min(1.8, max(0.35, spread))

    return [quantile(median, spread, z) for z in Z]


def quantile(median, spread, z):
    q = inv_logit(logit(median) + z * spread)
    return round4(clamp_rate(q))


def forecast_frozen(reads, index, trophy, t_first, t_last, t_target):
    """Forecast one trophy when the page never changed in the history.

    Two-component mixture: with probability (1 - p) no refresh lands before
    the target and the rate stays exactly where it is; with probability p
    one does, the tracked-player count jumps by a factor G and the rate is
    diluted to r * (1 + k*(G-1)) / G, where k is the rate at which the newly
    tracked players earn the trophy relative to the launch-night cohort.
    G and k come from the one refresh the kit's rules describe (638 -> 3,868
    tracked players; the refreshed totals imply k ~= 0.07).  The quantiles
    are found by inverting the mixture's distribution on a logit grid.
    """
    last_rate = reads[-1][2][index]
    if last_rate is None:
        last_rate = 0.01
    if last_rate >= 99.99:
        # The opening mission: it survived the first 6x jump in tracked
        # players at 100%, so it is expected to stay there; only the 5%
        # quantile hedges a small chance that a newcomer has not done it.
        return [99.5, 100, 100, 100, 100]

    freeze_hours = max(0.0, t_last - t_first)
    horizon_hours = max(0.0, t_target - t_last)
    # Refreshes land hours apart; a page frozen for most of that interval is
    # overdue one.  The probability is capped below 1: a refresh is likely,
    # not certain.
    p_refresh = min(0.95, (freeze_hours + horizon_hours) / 5.0)

    growth = 6.0
    new_earner_factor = 0.07
    diluted = last_rate * (1.0 + new_earner_factor * (growth - 1.0)) / growth

    mu_stay = logit(last_rate)
    mu_diluted = logit(diluted)
    s_stay = 0.08
    s_diluted = 0.55

    # Invert the mixture CDF on a logit grid.
    lo, hi = logit(0.01), logit(99.99)
    steps = 2000
    grid = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
    cdf = []
    for x in grid:
        cdf.append((1.0 - p_refresh) * norm_cdf((x - mu_stay) / s_stay)
                   + p_refresh * norm_cdf((x - mu_diluted) / s_diluted))
    out = []
    for level in LEVELS:
        j = 0
        while j < steps and cdf[j] < level:
            j += 1
        if j == 0:
            x = grid[0]
        elif j > steps:
            x = grid[steps]
        else:
            c0, c1 = cdf[j - 1], cdf[j]
            frac = 0.0 if c1 <= c0 else (level - c0) / (c1 - c0)
            x = grid[j - 1] + frac * (grid[j] - grid[j - 1])
        out.append(round4(clamp_rate(inv_logit(x))))
    # Rounding can only break monotonicity by a hair; enforce it.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


# --- case driver ----------------------------------------------------------

def forecast_case(case):
    """Forecast every trophy of one case; returns the output document."""
    if not case["history"]:
        raise ValueError("case has no history")
    reads = distinct_reads(case)
    t_last = parse_time(case["at"])
    t_target = t_last + case["horizon_minutes"] / 60.0
    t_first = reads[0][0]

    # Frozen page and a short horizon: the mixture path.  A frozen page
    # over a long horizon is not a refresh story any more (players keep
    # playing), so it falls through to the trend path with prior slopes.
    frozen = len(reads) == 1
    short_horizon = (t_target - t_last) <= 12.0
    n_target, _ = fit_tracked_growth(reads, t_last, t_target)

    rows = []
    for index, trophy in enumerate(case["trophies"]):
        if frozen and short_horizon:
            qs = forecast_frozen(reads, index, trophy,
                                 t_first, t_last, t_target)
        else:
            qs = forecast_trend(reads, index, trophy,
                                t_last, t_target, n_target)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
