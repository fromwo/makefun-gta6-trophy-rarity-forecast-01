# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fromwo/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fromwo/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasting every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase one week after launch (2026-11-26 00:00 UTC), as five quantiles per trophy, from the house's recording of the trophy page up to the deadline. One program, `main.py`, reads a case and writes the forecast. It is scored by mean pinball loss on the logit scale; lower is better.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `*.in` file in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`). Each output is a single JSON object `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, 55 rows, every value between 0 and 100, quantiles non-decreasing, well under the 65,536-byte cap. It uses only the Python standard library, needs no network, and is deterministic: running it twice on the same input gives byte-identical output.

## How the forecast works

An earn rate is a ratio, `earners / players tracked`, and the two sides move for different reasons, so the program forecasts them separately and divides at the target instead of extrapolating the ratio itself.

- **Players tracked** follows a power law in `1 + hours since release`, fitted on log-log scale over the recent history. When every read shows the same count, the page is stale — no refresh has been observed yet — so a prior exponent of 0.9 anchored at the first read is used instead. That is the key launch-night effect: the page sat frozen at 638 tracked players for hours while the real tracked population kept growing, so a flat history still implies every rate will be diluted at the target.
- **Earners per trophy** follow the same kind of power law, fitted per trophy, with the slope shrunk halfway toward the cross-trophy median slope for trophies with few earners or few distinct reads, because a count of 4 or 7 is mostly noise. In the stale-page regime a small fresh-cohort term is added: the players revealed by the coming refresh are assumed to have earned each trophy at 1.5% of the established rate, since a brand-new player has barely started.
- **Story order** is enforced afterwards: a later chapter cannot be held by more players than an earlier one, and the platinum cannot exceed the 100%-completion gold. The trophy everyone holds stays (nearly) universal.
- **Quantiles** are the median shifted on the logit scale by the standard normal quantiles times a spread that grows with the horizon, with the staleness of the page, with the fit residual, and for very rare trophies — so a short, stale horizon still gets an honest band and a 4.5-day extrapolation gets a wide one.

## Local check on the practice cases

On the two practice cases (checkpoints 03:00 and 03:30 UTC on launch night, forecasting 90 minutes ahead across Exophase's first refresh), the program's loss is about 0.10–0.13 when the answers are reconstructed from the refresh figures the rules publish (3,868 players tracked, 4,276 trophies earned at 04:30 UTC), against about 1.51 for the house baseline on the same reconstruction. The 90% bands cover the reconstructed answers on both cases, and two runs produce identical output files.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
