"""Forecast GTA VI trophy earn rates for make.fun gta6-trophy-rarity-forecast-01.

For every trophy of Grand Theft Auto VI on PlayStation 5, the program emits
five quantiles (5% / 25% / 50% / 75% / 95%) of its Exophase earn rate -- the
share of tracked players who hold it -- at the target time of one case,
from the house's reads of the trophy page up to the case's checkpoint.

Why the forecast is built the way it is
----------------------------------------
An earn rate is a fraction, earners over tracked players, and after a
launch the two parts of the fraction move against each other: the player
base swells as more people sync their profiles, which waters every rate
down, while the earners for a trophy climb as the players already tracked
work through the game, which pushes every rate up.  Forecasting the
fraction directly would tangle those two motions, so each side gets its
own power law in log time since release, and the two extrapolations are
divided back into a rate at the end:

* The player base is one curve for the whole game, fitted once from every
  distinct player count in the history, because the denominator is the
  same for all 55 trophies and the data for it is plentiful.
* Each trophy's earners get their own curve, fitted to that trophy's
  distinct reads and pulled toward a slope typical of its kind (story
  trophies climb steeply, long grinds barely move).  The curve is pinned
  to the trophy's latest read rather than to its fitted intercept, so a
  noisy early read cannot drag the forecast around.
* The resulting rate is then mixed, in logit space, with a prior rate for
  the trophy's kind at one week after release.  The prior carries more
  weight when the trophy's own history is thin, or when the trend runs
  far away from the prior; and it only speaks for week one, so its weight
  fades for earlier targets.
* The width of the five quantiles grows with the horizon, with the
  scatter left over by the earner fit, and with how far the trend sits
  from the prior.

One history shape needs a different treatment: a page that never moved at
all during the whole history.  That happens on launch night, before
Exophase's first refresh, and then there is no trend to fit -- the page is
simply frozen between refreshes.  The forecast becomes a mixture of two
possibilities: either no refresh lands before the target, and the rate
stays exactly where it is, or one does, the player count jumps, and the
rate is watered down.  The refresh chance grows with how long the page
has already sat frozen, and the dilution is calibrated on the one refresh
the kit's rules describe (638 -> 3,868 tracked players).  The mixture's
quantiles come from inverting its distribution on a logit grid.

The program uses only the Python standard library.  It is deterministic:
no clock, no randomness, no network, and cases are written in name order.

Local use:  MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader:     python3 -P -s /bot/main.py  (reads /input, writes /output)
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILE_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL_QUANTILES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release of GTA VI; the log-time axis starts here (2026-11-19T00:00:00Z).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# A read stamped before release is still treated as a few minutes in, so
# that log(hours since release) stays defined.
MIN_HOURS = 0.25
# Ceiling on extrapolated tracked players: week-one PS5 sales are forecast
# at roughly 10-25 million copies, and Exophase tracks only the players
# who sync a profile, a small fraction of buyers.
MAX_TRACKED = 2_000_000
# One week after release, in hours; the class priors are week-one rates.
WEEK_HOURS = 168.0

# --- trophy kinds ---------------------------------------------------------
# Every kind carries three hand-chosen numbers (see README.md for where
# they come from and how uncertain they are):
#   week_rate : the earn rate (percent) expected at one week after release
#               among tracked players.
#   slope     : the prior growth exponent of the earner count, E ~ t**slope,
#               used when a trophy's own history is too thin to fit one.
#   cap       : a loose ceiling the median forecast is never allowed above.
# Kinds are matched by keywords in the trophy description, in order.
PRIOR_BY_KIND = {
    # The opening mission everyone completes; it sat at 100% through the
    # first 6x jump in tracked players, so it is expected to stay there.
    "intro":       (99.9, 1.0, 100.0),
    "story_ch1":   (40.0, 2.0, 75.0),
    "story_ch2":   (30.0, 2.1, 65.0),
    "story_ch3":   (22.0, 2.2, 55.0),
    "story_ch4":   (16.0, 2.2, 45.0),
    "story_final": (10.0, 2.3, 35.0),
    # Named story missions (Breakfast Run, The Cartel Heist, ...).
    "mission":     (20.0, 1.8, 55.0),
    # Final mission of a stranger strand (Andy, Kenton, Billy, ...).
    "stranger":    (12.0, 1.7, 40.0),
    "completion":  (0.10, 1.5, 0.5),
    "platinum":    (0.05, 1.5, 1.0),
    # Long grinds: 250 stunts, 100 WAiNK spots, 3 card sets, 50 challenges,
    # gold medals everywhere, 8 stranger strands, 5 bank vaults, every gang.
    "grind":       (1.0, 1.1, 8.0),
    # Repeatable activities: races, fishing, arcade, vehicles, money.
    "activity":    (8.0, 1.3, 30.0),
    "explore":     (6.0, 1.4, 25.0),
    # One-off feats: wanted-level escapes, robberies, stunts, set pieces.
    "feat":        (10.0, 1.2, 35.0),
}

# Phrases that mark a long grind, in the order they are checked.
GRIND_PHRASES = (
    "250 stunts",
    "gold medal in all",
    "drone race at every location",
    "8 mission strands",
    "3 collectible card sets",
    "waink it at 100",
    "meditate at every spot",
    "complete 50 challenges",
    "rob 5 bank vaults",
    "belonging to each gang",
)

# Phrases that mark a repeatable activity.
ACTIVITY_PHRASES = (
    "try each type of activity",
    "win 20 different races",
    "fishing challenge",
    "arcade game",
    "unique animal",
    "discover 100 vehicles",
    "laundering marked bills",
    "spend $100,000",
    "purchase all garage properties",
    "register 10 vehicles",
)

# Named story missions, matched when the description starts with
# "complete ".
NAMED_MISSION_PHRASES = (
    "breakfast run",
    "heist",
    "child's play",
    "headliners",
    "takedown",
    "savage glory",
)

# "Complete chapter N" maps to the story kind for that chapter.
CHAPTER_KINDS = {
    "complete chapter 1": "story_ch1",
    "complete chapter 2": "story_ch2",
    "complete chapter 3": "story_ch3",
    "complete chapter 4": "story_ch4",
}


def kind_of(trophy):
    """Assign a trophy to one of the kinds above by its description.

    The checks run in a fixed order because a trophy can match more than
    one: the platinum tier and the 100%-completion trophy are caught
    before anything else, and a named story mission is only recognised
    when its description starts with "complete ".
    """
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    if tier == "platinum":
        return "platinum"
    if "100% completion" in desc:
        return "completion"
    if "gone fishing" in desc:
        return "intro"
    for phrase, kind in CHAPTER_KINDS.items():
        if phrase in desc:
            return kind
    if "complete the final mission" in desc:
        return "stranger" if " for " in desc else "story_final"
    if any(phrase in desc for phrase in GRIND_PHRASES):
        return "grind"
    if desc.startswith("complete ") and "mission" in desc:
        return "mission"
    if desc.startswith("complete ") and any(
            phrase in desc for phrase in NAMED_MISSION_PHRASES):
        return "mission"
    if "explore all of leonida" in desc:
        return "explore"
    if any(phrase in desc for phrase in ACTIVITY_PHRASES):
        return "activity"
    return "feat"


# --- small numeric helpers ------------------------------------------------

def clamp_percent(pct):
    """Clamp a percent into the range the logit is defined on."""
    return min(99.99, max(0.01, pct))


def to_logit(pct):
    """Logit of a percent, with the percent clamped first."""
    p = clamp_percent(pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(x):
    """Inverse logit, as a percent, computed without overflow."""
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return 100.0 * ex / (1.0 + ex)


def round_4dp(x):
    """Round to 4 decimals, half up, returning an int for whole numbers."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def hours_since_release(text):
    """Parse an ISO 8601 UTC timestamp into hours since release."""
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    hours = (moment - RELEASE).total_seconds() / 3600.0
    return max(MIN_HOURS, hours)


def ols_line(xs, ys):
    """Ordinary least squares y = a + b*x; returns (a, b, residual_sd)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    spread_xx = sum((x - mean_x) ** 2 for x in xs)
    if spread_xx <= 0:
        return mean_y, 0.0, 0.0
    slope = sum((x - mean_x) * (y - mean_y)
                for x, y in zip(xs, ys)) / spread_xx
    intercept = mean_y - slope * mean_x
    if n > 2:
        resid = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2
                for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return intercept, slope, resid


def normal_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --- history preparation --------------------------------------------------

def collapse_reads(case):
    """Collapse the history to one entry per distinct page state.

    Exophase refreshes a game's counts in jumps hours apart, so long runs
    of reads print the same page.  Only the first read showing each state
    carries timing information; the rest are duplicates.  Returns a list
    of (hours_since_release, players_tracked, rates) oldest first.
    """
    out = []
    seen = set()
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key in seen:
            continue
        seen.add(key)
        out.append((hours_since_release(read["at"]),
                    read["players_tracked"], list(read["rates"])))
    return out


def extrapolate_players(reads, t_last, t_target):
    """Fit the denominator's power law and extrapolate it to the target.

    Returns (n_target, exponent).  The exponent is fitted to the distinct
    player counts, using the most recent 48 hours when that leaves at
    least two points: early launch-night growth was far faster than the
    pace that holds two days in, and the old pace must not drag the
    extrapolation.  With fewer than two distinct counts it falls back to
    0.9, the rough pace of launch-week tracker growth.  The extrapolated
    count is capped, and so is the total growth factor.
    """
    all_points = [(math.log(t), math.log(n))
                  for t, n, _ in reads if n and n > 0]
    recent = [(math.log(t), math.log(n)) for t, n, _ in reads
              if n and n > 0 and t >= t_last - 48.0]
    if len(recent) >= 2:
        all_points = recent
    if len(all_points) >= 2:
        _, exponent, _ = ols_line([p[0] for p in all_points],
                                  [p[1] for p in all_points])
        exponent = min(2.0, max(0.2, exponent))
    else:
        exponent = 0.9
    n_last = reads[-1][1]
    growth = (t_target / t_last) ** exponent
    growth = min(growth, 8.0)
    return min(MAX_TRACKED, n_last * growth), exponent


# --- the two forecast paths -----------------------------------------------

def earner_points(reads, index):
    """Distinct (time, earners) points for one trophy, oldest first.

    A point is only kept when its earner count differs from the previous
    kept point: repeated identical page states add no timing information.
    """
    points = []
    for t, n, rates in reads:
        rate = rates[index]
        if rate is None or n is None:
            continue
        earners = n * rate / 100.0
        if not points or abs(points[-1][1] - earners) > 1e-9:
            points.append((t, earners))
    return points


def estimate_earner_slope(points, slope_prior, t_last):
    """Estimate one trophy's earner growth exponent.

    Returns (slope, residual_sd).  Where the trophy moved enough, the
    exponent is fitted to the most recent 48 hours of distinct points and
    then shrunk toward the class prior slope; the earliest reads are a
    frozen launch-night page, and their flatness would drag the slope down
    just when the players are speeding up.  With exactly two distinct
    states the slope comes from the log-time gap between them -- and a
    non-positive gap (two states stamped at the same time, e.g. a recorder
    overlap) falls back to the class prior slope instead of dividing by
    zero and crashing the run.  With fewer points there is nothing to fit,
    so the prior slope is used as-is.
    """
    recent = [p for p in points if p[0] >= t_last - 48.0]
    if len(recent) < 3:
        recent = points
    if len(recent) >= 3:
        _, fitted, resid = ols_line(
            [math.log(t) for t, _ in recent],
            [math.log(e + 0.5) for _, e in recent])
        fitted = min(3.0, max(0.0, fitted))
        weight_fit = len(recent) / (len(recent) + 3.0)
        slope = weight_fit * fitted + (1.0 - weight_fit) * slope_prior
        return slope, min(resid, 0.8)
    if len(points) == 2:
        gap_log = math.log(points[1][0]) - math.log(points[0][0])
        if gap_log <= 0:
            return slope_prior, 0.0
        fitted = (math.log(points[1][1] + 0.5)
                  - math.log(points[0][1] + 0.5)) / gap_log
        fitted = min(3.0, max(0.0, fitted))
        return 0.5 * fitted + 0.5 * slope_prior, 0.0
    return slope_prior, 0.0


def prior_blend_weight(n_points, trend_rate, prior_rate, t_target):
    """How much of the class prior to mix into the trend, in logit space.

    The prior gets more weight when the trophy barely moved, and when the
    trend runs far from the prior (an extreme extrapolation is not trusted
    on its own).  And the prior only speaks for week one: the earlier the
    target, the less a week-one rate says about it, so its weight is
    scaled by how far into the week the target sits.  (Without that
    scaling, a target a few hours after launch would be dragged toward
    week-one rates twenty times above the diluted reality.)
    """
    if n_points <= 2:
        weight = 0.55
    elif n_points < 8:
        weight = 0.30
    else:
        weight = 0.18
    if abs(to_logit(trend_rate) - to_logit(prior_rate)) > 3.0:
        weight = max(weight, 0.5)
    weight *= min(1.0, max(0.0, t_target / WEEK_HOURS))
    return weight


def trend_spread(horizon_days, horizon_hours, resid, trend_rate,
                 prior_rate, n_reads):
    """Logit standard deviation of the trend forecast.

    A base that grows with the horizon and with the fit's residual, plus
    a term for how far the trend sits from the prior, plus, over short
    horizons, the lumpiness of discrete refreshes: the page only moves
    when Exophase refreshes it, so a refresh landing inside a short
    horizon can jump the rate in one step.  That lumpiness washes out
    over a multi-day horizon, where many refreshes average out.
    """
    spread = math.sqrt(0.30 ** 2
                       + (0.30 * math.sqrt(max(horizon_days, 0.0))) ** 2
                       + resid ** 2)
    if n_reads >= 2:
        spread = math.sqrt(spread ** 2
                           + (0.35 * math.exp(-horizon_hours / 24.0)) ** 2)
    spread += 0.12 * abs(to_logit(trend_rate) - to_logit(prior_rate))
    return min(1.8, max(0.35, spread))


def quantile_from(median, spread, z):
    """One quantile of a logit-normal forecast, rounded to 4 decimals."""
    return round_4dp(clamp_percent(from_logit(to_logit(median) + z * spread)))


def forecast_from_trend(reads, index, trophy, t_last, t_target, n_target):
    """Forecast one trophy from its own history plus its class prior."""
    kind = kind_of(trophy)
    prior_week, slope_prior, cap = PRIOR_BY_KIND[kind]
    horizon_days = (t_target - t_last) / 24.0

    # The class prior at the target: week-one prior, scaled down for an
    # earlier target along a saturating curve, never above the week-one value.
    if t_target >= WEEK_HOURS:
        prior_rate = prior_week
    else:
        prior_rate = prior_week * min(1.0, (t_target / WEEK_HOURS) ** 0.6)

    points = earner_points(reads, index)
    n_points = len(points)

    last_rate = reads[-1][2][index]
    if last_rate is None:
        last_rate = prior_rate
    earners_last = reads[-1][1] * last_rate / 100.0

    slope, resid = estimate_earner_slope(points, slope_prior, t_last)

    # Extrapolate earners from the most recent read, divide by the
    # extrapolated denominator, and never let earners exceed players.
    earners_target = earners_last * (t_target / t_last) ** slope
    earners_target = min(earners_target, n_target)
    trend_rate = (100.0 * earners_target / n_target
                  if n_target > 0 else last_rate)

    # Blend the trend with the class prior in logit space.
    weight = prior_blend_weight(n_points, trend_rate, prior_rate, t_target)
    median_logit = ((1.0 - weight) * to_logit(trend_rate)
                    + weight * to_logit(prior_rate))
    median = from_logit(median_logit)
    median = min(median, cap)
    median = clamp_percent(median)

    spread = trend_spread(horizon_days, max(0.0, t_target - t_last), resid,
                          trend_rate, prior_rate, len(reads))

    return [quantile_from(median, spread, z) for z in NORMAL_QUANTILES]


def invert_cdf_on_logit_grid(grid, cdf, level):
    """Smallest grid point whose CDF reaches `level`, linearly interpolated.

    Walks the grid until the cumulative probability first reaches the
    level, then interpolates between the two bracketing grid points.  A
    level below the first grid value returns the grid's low end.
    """
    steps = len(grid) - 1
    j = 0
    while j < steps and cdf[j] < level:
        j += 1
    if j == 0:
        return grid[0]
    c0, c1 = cdf[j - 1], cdf[j]
    frac = 0.0 if c1 <= c0 else (level - c0) / (c1 - c0)
    return grid[j - 1] + frac * (grid[j] - grid[j - 1])


def forecast_from_frozen(reads, index, trophy, t_first, t_last, t_target):
    """Forecast one trophy when the page never changed in the history.

    Two-component mixture: with probability (1 - p) no refresh lands
    before the target and the rate stays exactly where it is; with
    probability p one does, the tracked-player count jumps by a factor G
    and the rate is diluted to r * (1 + k*(G-1)) / G, where k is the rate
    at which the newly tracked players earn the trophy relative to the
    launch-night cohort.  G and k come from the one refresh the kit's
    rules describe (638 -> 3,868 tracked players; the refreshed totals
    imply k ~= 0.07).  The quantiles are found by inverting the mixture's
    distribution on a logit grid.
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
    # Refreshes land hours apart; a page frozen for most of that interval
    # is overdue one.  The probability is capped below 1: a refresh is
    # likely, not certain.
    p_refresh = min(0.95, (freeze_hours + horizon_hours) / 5.0)

    growth = 6.0
    new_earner_factor = 0.07
    diluted = (last_rate * (1.0 + new_earner_factor * (growth - 1.0))
               / growth)

    mu_stay = to_logit(last_rate)
    mu_diluted = to_logit(diluted)
    sd_stay = 0.08
    sd_diluted = 0.55

    # Invert the mixture CDF on a logit grid.
    lo = to_logit(0.01)
    hi = to_logit(99.99)
    steps = 2000
    grid = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
    cdf = []
    for x in grid:
        cdf.append((1.0 - p_refresh) * normal_cdf((x - mu_stay) / sd_stay)
                   + p_refresh * normal_cdf((x - mu_diluted) / sd_diluted))
    out = []
    for level in QUANTILE_LEVELS:
        x = invert_cdf_on_logit_grid(grid, cdf, level)
        out.append(round_4dp(clamp_percent(from_logit(x))))
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
    reads = collapse_reads(case)
    t_last = hours_since_release(case["at"])
    t_target = t_last + case["horizon_minutes"] / 60.0
    t_first = reads[0][0]

    # Frozen page and a short horizon: the mixture path.  A frozen page
    # over a long horizon is not a refresh story any more (players keep
    # playing), so it falls through to the trend path with prior slopes.
    frozen = len(reads) == 1
    short_horizon = (t_target - t_last) <= 12.0
    n_target, _ = extrapolate_players(reads, t_last, t_target)

    rows = []
    for index, trophy in enumerate(case["trophies"]):
        if frozen and short_horizon:
            qs = forecast_from_frozen(reads, index, trophy,
                                      t_first, t_last, t_target)
        else:
            qs = forecast_from_trend(reads, index, trophy,
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
