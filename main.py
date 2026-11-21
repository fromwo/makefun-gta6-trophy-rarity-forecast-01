#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

Reads every ``*.in`` case file from $MAKEFUN_INPUT_DIR (default /input) and
writes one ``<id>.out`` forecast per case to $MAKEFUN_OUTPUT_DIR
(default /output).  Standard library only, no network, deterministic:
the same input always produces byte-identical output.

Model, in one paragraph
-----------------------
A trophy's earn rate is a ratio, ``rate = earners / players_tracked``, and
the two sides move for different reasons, so they are forecast separately
and divided at the target instead of extrapolating the ratio itself.
Both counts are modelled as power laws in ``1 + hours since release``,
fitted by least squares on log-log scale over the recent history:

* ``players(t)`` -- Exophase's tracked-player count.  It only ever grows,
  in jumps when Exophase refreshes the page.  When the history shows
  several distinct counts the exponent is fitted; when every read shows
  the same count the page is stale (no refresh has been observed yet) and
  a prior exponent of 0.9 is used, anchored at the first read.  That
  matters most on launch night: the page sat frozen at 638 players for
  hours while the real tracked population kept growing, so a flat
  history must still imply a dilution of every rate at the target.
* ``earners_i(t)`` -- how many tracked players hold trophy ``i``.  Fitted
  per trophy the same way, with the slope shrunk halfway toward the
  cross-trophy median slope whenever a trophy has few earners or few
  distinct reads, because a count of 4 or 7 is mostly noise.  In the
  stale-page regime the observed earners are frozen too, so a small
  "fresh cohort" term is added: the players revealed by the coming
  refresh are assumed to have earned each trophy at 1.5% of the
  established rate (a brand-new player has barely started), on top of
  the trophy everyone holds, which stays at 100%.

The median forecast is ``100 * earners_i(target) / players(target)``.
Story trophies are then forced to respect the game's own order -- a
later chapter cannot be held by more players than an earlier one, and
the platinum cannot exceed the 100%-completion gold -- by a running
minimum along the chapter chain.  Quantiles are the median shifted on
the logit scale by the standard normal quantiles times a spread that
grows with the horizon, with the staleness of the page, with the fit
residual and for very rare trophies, so a short, stale horizon still
gets an honest band and a 4.5-day extrapolation gets a wide one.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone

# Standard normal quantiles for the forecast levels 0.05, 0.25, 0.5, 0.75,
# 0.95, in that order.
Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

# Prior exponent for tracked-player growth while the page has never
# refreshed: players accumulate roughly linearly in hours since release.
PLAYER_GROWTH_PRIOR = 0.9
# Fraction of the established rate at which a freshly revealed player is
# assumed to have earned a trophy by the target (stale-page regime only).
FRESH_COHORT_FACTOR = 0.015
# Recent window used for the trend fits, in hours.
WINDOW_HOURS = 36.0


def parse_time(text):
    """Parse an ISO-8601 timestamp (trailing Z allowed) to an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def logit(pct):
    """logit of a percent in [0.01, 99.99], clamped so 0 and 100 have one."""
    p = min(max(pct, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    """Inverse logit, returned as a percent, guarded against overflow."""
    value = max(min(value, 20.0), -20.0)
    return 100.0 / (1.0 + math.exp(-value))


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def time_factor(hours):
    """1 + hours since release, floored so a read from just before
    release (the first probe, minutes early) still anchors the power
    law instead of being clamped to exactly 1."""
    return 1.0 + max(hours, -0.8)


def least_squares(xs, ys):
    """Ordinary least squares line; returns (slope, intercept, rmse)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, mean_y, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    rmse = math.sqrt(
        sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / n
    )
    return slope, intercept, rmse


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def chapter_number(trophy):
    """Chapter index for story trophies, else None (final mission = 99)."""
    text = trophy.get("description") or ""
    match = re.search(r"Chapter (\d+)", text)
    if match:
        return int(match.group(1))
    if "final mission" in text.lower():
        return 99
    return None


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_minutes = float(case.get("horizon_minutes") or 0.0)
    horizon_days = horizon_minutes / 1440.0

    game = case.get("game") or {}
    release = game.get("release_date") or ""
    try:
        release_start = datetime.fromisoformat(release).replace(
            tzinfo=timezone.utc)
    except (ValueError, TypeError):
        release_start = None

    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])

    def hours_since_release(moment):
        if release_start is None:
            return None
        return (moment - release_start).total_seconds() / 3600.0

    # ---- Collect the history as (hours since release, players, rates). ----
    reads = []
    for entry in history:
        try:
            moment = parse_time(entry["at"])
        except (KeyError, ValueError, TypeError):
            continue
        hours = hours_since_release(moment)
        if hours is None:
            hours = (moment - parse_time(history[0]["at"])).total_seconds() / 3600.0
        players = entry.get("players_tracked")
        rates = entry.get("rates") or []
        reads.append((hours, players, rates))
    reads.sort(key=lambda row: row[0])

    if not reads:
        # No history at all: fall back to tier priors with a wide spread.
        priors = {"platinum": 0.5, "gold": 1.0, "silver": 0.7, "bronze": 1.0}
        rows = []
        for trophy in trophies:
            med = priors.get(trophy.get("tier"), 1.0)
            rows.append((trophy["id"], med, 0.9))
        return {"trophies": [quantile_row(tid, med, spread)
                             for tid, med, spread in rows]}

    last_hours, last_players, last_rates = reads[-1]
    target_hours = hours_since_release(target)
    if target_hours is None:
        target_hours = last_hours + horizon_minutes / 60.0
    checkpoint_hours = hours_since_release(checkpoint)
    if checkpoint_hours is None:
        checkpoint_hours = last_hours

    # Last non-null rate per trophy, walking back through the reads.
    last_rate = []
    for index in range(len(trophies)):
        rate = None
        for _, _, rates in reversed(reads):
            if index < len(rates) and rates[index] is not None:
                rate = float(rates[index])
                break
        last_rate.append(rate if rate is not None else 0.0)

    # ---- Players-tracked model: power law in (1 + hours). ----
    player_series = [(h, float(p)) for h, p, _ in reads if p]
    window = [(h, p) for h, p in player_series
              if h >= last_hours - WINDOW_HOURS]
    if len(window) < 3:
        window = player_series
    distinct_players = {p for _, p in window}
    flat_players = len(distinct_players) < 2

    if not flat_players and len(window) >= 3:
        xs = [math.log(time_factor(h)) for h, _ in window]
        ys = [math.log(p) for _, p in window]
        slope, _, _ = least_squares(xs, ys)
        slope = clamp(slope, 0.05, 1.5)
        anchor_hours, anchor_players = window[-1]
        growth = (time_factor(target_hours)
                  / time_factor(anchor_hours)) ** slope
        players_target = anchor_players * growth
        players_target = clamp(players_target, anchor_players,
                               6.0 * anchor_players)
        players_at_checkpoint = anchor_players
    else:
        # Stale page: no refresh observed, so the last count is old.  Anchor
        # the power law at the first read and let it run underneath the
        # frozen reads; the coming refresh reveals the accumulated players.
        anchor_hours, anchor_players = player_series[0]
        players_target = anchor_players * (
            time_factor(target_hours)
            / time_factor(anchor_hours)) ** PLAYER_GROWTH_PRIOR
        players_target = clamp(players_target, last_players or anchor_players,
                               8.0 * (last_players or anchor_players))
        players_at_checkpoint = anchor_players * (
            time_factor(checkpoint_hours)
            / time_factor(anchor_hours)) ** PLAYER_GROWTH_PRIOR

    # ---- Earners model, one power law per trophy. ----
    # First pass: fit every trophy that has enough distinct counts, so the
    # rare ones can be shrunk toward the typical slope.
    slopes = {}
    residuals = {}
    for index in range(len(trophies)):
        points = []
        for hours, players, rates in reads:
            if (hours >= last_hours - WINDOW_HOURS
                    and index < len(rates) and rates[index] is not None
                    and players):
                earners = float(rates[index]) * float(players) / 100.0
                if earners > 0:
                    points.append((hours, earners))
        if len(points) < 3:
            points = []
            for hours, players, rates in reads:
                if (index < len(rates) and rates[index] is not None
                        and players):
                    earners = float(rates[index]) * float(players) / 100.0
                    if earners > 0:
                        points.append((hours, earners))
        distinct = {round(e, 6) for _, e in points}
        if len(points) >= 3 and len(distinct) >= 2:
            xs = [math.log(time_factor(h)) for h, _ in points]
            ys = [math.log(e) for _, e in points]
            slope, _, rmse = least_squares(xs, ys)
            slopes[index] = clamp(slope, -0.5, 3.0)
            residuals[index] = rmse
    typical_slope = median_of(list(slopes.values())) if slopes else 0.0

    medians = []
    spreads = []
    for index, trophy in enumerate(trophies):
        rate = last_rate[index]
        if rate >= 99.99:
            # Everybody holds it; it stays (nearly) universal.
            medians.append(100.0)
            spreads.append(0.0)
            continue
        players_last = last_players or players_target
        earners_last = rate * players_last / 100.0
        if earners_last <= 0:
            # Nobody holds it yet: a small hedge, wide spread.
            medians.append(0.05)
            spreads.append(0.9)
            continue
        if flat_players:
            slope = 0.0
        elif index in slopes:
            slope = slopes[index]
            if earners_last < 10.0:
                slope = 0.5 * slope + 0.5 * typical_slope
        else:
            slope = typical_slope if slopes else 0.0
        earners_target = earners_last * (
            time_factor(target_hours)
            / time_factor(last_hours)) ** slope
        if flat_players:
            # The coming refresh reveals players accumulated since the
            # anchor read; a few of them will already hold this trophy.
            revealed = max(players_target - player_series[0][1], 0.0)
            earners_target += revealed * (rate / 100.0) * FRESH_COHORT_FACTOR
        earners_target = clamp(earners_target, 0.0, 0.995 * players_target)
        median = 100.0 * earners_target / players_target
        median = clamp(median, 0.01, 99.99)
        medians.append(median)

        resid = residuals.get(index, 0.15)
        variance = (0.30 ** 2 + 0.10 * horizon_days
                    + (0.0625 if flat_players else 0.0)
                    + min(resid, 0.30) ** 2
                    + (0.01 if earners_last < 8.0 else 0.0))
        spreads.append(clamp(math.sqrt(variance), 0.35, 0.95))

    # ---- Story order: later chapters cannot exceed earlier ones. ----
    chain = sorted(
        (chapter_number(t), i) for i, t in enumerate(trophies)
        if chapter_number(t) is not None)
    running = None
    for _, index in chain:
        if medians[index] >= 99.99:
            continue
        if running is None or medians[index] < running:
            running = medians[index]
        else:
            medians[index] = running
    gold_medians = [medians[i] for i, t in enumerate(trophies)
                    if t.get("tier") == "gold" and medians[i] < 99.99]
    for index, trophy in enumerate(trophies):
        if trophy.get("tier") == "platinum" and gold_medians:
            medians[index] = min(medians[index], min(gold_medians))

    rows = []
    for index, trophy in enumerate(trophies):
        rows.append(quantile_row(trophy["id"], medians[index],
                                 spreads[index]))
    return {"trophies": rows}


def quantile_row(trophy_id, median, spread):
    """Five quantiles around a median on the logit scale."""
    if median >= 99.99:
        return {"id": trophy_id, "q": [99.2, 99.8, 99.97, 100.0, 100.0]}
    centre = logit(median)
    quantiles = []
    previous = 0.01
    for z in Z_SCORES:
        value = round(inv_logit(centre + z * spread), 4)
        value = clamp(value, 0.01, 99.99)
        if value < previous:
            value = previous
        previous = value
        quantiles.append(value)
    return {"id": trophy_id, "q": quantiles}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(input_dir)
                   if name.endswith(".in"))
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as handle:
            case = json.load(handle)
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        out_name = str(case.get("id") or name[:-3]) + ".out"
        with open(os.path.join(output_dir, out_name), "w",
                  encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
