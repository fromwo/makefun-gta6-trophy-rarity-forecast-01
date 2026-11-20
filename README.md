# GTA VI Trophy Rarity Forecast — make.fun entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fromwo/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)

A forecaster for the make.fun competition
**gta6-trophy-rarity-forecast-01**: given the house's recorded reads of
Grand Theft Auto VI's PlayStation 5 trophy page on Exophase up to a
checkpoint, predict every trophy's earn rate (the percent of tracked players
who hold it) at the target time, as five quantiles per trophy
(5% / 25% / 50% / 75% / 95%). Scored by mean pinball loss on the logit
scale. The final case is cut at 2026-11-21T11:30:00Z and targets
2026-11-26T00:00:00Z, a week after release.

The entry is a single Python file, `main.py`, standard library only. It
reads every `*.in` file in `$MAKEFUN_INPUT_DIR` (default `/input`) and
writes one `*.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`).
It is deterministic: same input, same bytes out, no clock, no randomness,
no network.

## The model

An earn rate is a ratio, **rate = earners / tracked players**, and after a
launch its two sides pull in opposite directions:

* the **denominator** grows as more players sync their profiles with
  Exophase, which *dilutes* every rate;
* the **numerator** grows as the players already tracked push further into
  the game, which *raises* every rate.

So the model splits the ratio and forecasts each side separately, in log
time since release (2026-11-19T00:00:00Z):

1. **Tracked players N(t).** One power law fitted to the whole history's
   distinct player counts, `ln N = c + bN * ln t`, using the most recent
   48 hours when possible. Denominator growth is shared by every trophy,
   so it is fitted once, globally, where the data is plentiful. The
   extrapolated count is capped at 2,000,000 and the total growth factor
   at 8x.
2. **Earners E(t), per trophy.** A power law per trophy,
   `ln(E + 0.5) = a + bE * ln t`, fitted to that trophy's distinct reads
   over the most recent 48 hours, then shrunk toward a class prior slope
   (story chapters rise fast, long grinds rise slowly). The extrapolation
   is anchored on the trophy's most recent read, not on the fitted
   intercept, so a noisy early read cannot drag the forecast. Earners are
   never allowed to exceed tracked players.
3. **Blend with a class prior.** The trend rate is blended in logit space
   toward a week-one prior rate for the trophy's class (see below). The
   prior's weight is higher when the trophy's own history is thin and
   when the trend runs far from the prior; and the prior only speaks for
   week one, so its weight scales with how far into the week the target
   sits — a target a few hours after launch is not dragged toward
   week-one rates.
4. **Spread.** The logit standard deviation grows with the horizon and
   with the fit's residual, plus a term for how far the trend sits from
   the prior, plus (over short horizons) the lumpiness of discrete
   Exophase refreshes. The 90% band is meant to cover about 90% of the
   time.

**Frozen page.** When the history never changed at all (launch night,
before Exophase's first refresh), there is no trend to fit and the page
is frozen between refreshes, so the forecast is a two-component mixture:
either no refresh lands before the target (the rate stays exactly where
it is) or one does (the tracked count jumps and the rate is diluted).
The refresh probability grows with how long the page has already been
frozen, and the dilution uses the one refresh the kit's rules describe
(638 → 3,868 tracked players, with the new players earning at roughly 7%
of the launch-night cohort's rate, inferred from the refreshed totals the
kit publishes: 4,276 trophies earned). Quantiles come from inverting the
mixture's distribution on a logit grid. The opening mission, which sat at
100% through that 6x jump in tracked players, is forecast to stay at
100%.

Trophies are classified by their description into: intro, story chapters
1–4, final mission, named story missions, stranger-strand finals, 100%
completion, platinum, long grinds, repeatable activities, exploration,
and one-off feats.

## Assumptions, priors and uncertainty

The class priors (week-one earn rates, in percent) are judgment calls,
not measurements — nobody has week-one GTA VI rates yet, because the week
is not over. They are built from three public anchors:

* **GTA V's lifetime trophy rates** on public trackers (e.g. PocketPSN:
  first story trophy ~38.6%, mid-story ~10–16%, final mission ~7.4%,
  100% completion ~0.2%, platinum ~0.1%). GTA VI's week-one rates should
  be *higher* than these for early trophies (launch-week buyers are
  enthusiasts) and *lower* for late ones (a week is not enough to finish
  a 45–80 hour story).
* **GTA VI's story length**: review playthroughs put a story-focused run
  at roughly 45–55 hours and a typical run at 60–80 hours, with 90+
  missions across five chapters and 200–300 hours for the platinum. In
  seven days most players will be mid-story, not finished.
* **GTA VI's launch scale**: analysts put week-one sales at roughly
  14 million copies (Compound base case) to 28–35 million (consensus) and
  higher; PS5 is about three quarters of that. Exophase tracks only the
  players who sync a profile, a small fraction of buyers, so tracked
  players in the low hundreds of thousands by 26 November is the working
  assumption behind the 2,000,000 cap.

Resulting week-one priors: chapter 1 ~40%, chapter 2 ~30%, chapter 3
~22%, chapter 4 ~16%, final mission ~10%, named story missions ~20%,
stranger finals ~12%, repeatable activities ~8%, one-off feats ~10%,
exploration ~6%, long grinds ~1%, 100% completion ~0.1%, platinum
~0.05%, opening mission ~100%.

What could be wrong, and how the model limits the damage:

* The priors could be off by 2x or more for any class. They only ever
  blend with the trend (weight 0.18–0.55, and only in full at week one),
  never replace it, and the spread widens when trend and prior disagree.
* The power-law extrapolations assume the recent pace holds. If player
  progression saturates faster than a power law (it will, eventually),
  the trend overstates late-story trophies; the class caps (e.g. chapter
  1 at 75%, final mission at 35%) bound the worst case.
* Exophase's refresh cadence is irregular. The frozen-page mixture
  assumes refreshes land hours apart; if the page freezes for a day, the
  mixture's dilution is the wrong shape — but that path only fires for
  short horizons.
* The practice cases only test the frozen-page path (their histories are
  flat). The trend path is validated on synthetic histories, not real
  ones; its first real test is the final case itself.

## Local validation

Practice cases, scored against the 04:30Z refreshed rates reconstructed
from the kit's published totals (3,868 players tracked, 4,276 trophies
earned; the reconstruction reproduces the house baseline's published
practice figures — loss ~1.51, typical miss ~0.68 points — closely):

| forecast | loss | typical miss | 90% coverage |
|---|---:|---:|---:|
| this model, p00 | 0.170 | 0.029 pp | 1.00 |
| this model, p01 | 0.135 | 0.011 pp | 1.00 |
| house baseline (same answers) | 1.401 | 0.677 pp | 0.00 |

(The reconstructed answers are an estimate — the proportional-dilution
assumption smooths the real distribution — so the absolute numbers are
indicative, not exact. The point is the direction: expecting the refresh
beats assuming the rates stay put.)

Two synthetic checks were also run: a case with one simulated refresh
appended to p00's history (medians track the diluted rates, e.g. chapter
1 median 0.33% vs diluted 0.28%, 100% completion 0.14% vs 0.14%), and a
final-like case with 2.5 days of synthetic history (chapter 1 median
36% vs the synthetic truth's 38%, final mission 7% vs 9%, platinum
0.01%). Both run cleanly.

Format checks on the practice run: 55 rows, ids 1–55 once each,
quantiles non-decreasing, all in [0, 100], outputs ~2.8 KB each (limit
65,536), two runs byte-identical, both cases in ~0.9 s total on this
machine (limit 8 s per case).

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

## Sources

* Kit rules and practice cases: `gta6-trophy-rarity-forecast-01-kit`
  (RULES.txt, competition.yaml, input/p00.in, input/p01.in).
* GTA V trophy rates: PocketPSN's GTA V trophy page (lifetime rates
  among its tracked players).
* GTA VI story length: review playthroughs reported around the
  18 November 2026 embargo (story-focused ~45–55 h, typical 60–80 h,
  platinum 200–300 h).
* GTA VI week-one sales: Compound's base case ~14M copies; analyst
  consensus ~28–35M; Newzoo's range higher (37–51M). Sensor Tower
  counted 17M+ pre-orders before launch.
* Exophase refresh behaviour: the kit's rules (first refresh between the
  04:00Z and 04:30Z reads on launch night: 638 → 3,868 tracked players,
  943 → 4,276 trophies earned).

## License

MIT — see `LICENSE`.
