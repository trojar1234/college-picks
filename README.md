# College football model

A market-blind college football model. It projects every FBS game, compares its numbers with the
sportsbooks, records bets before kickoff, grades itself, and publishes a website that updates on
its own through GitHub Actions and GitHub Pages.

Sportsbook lines are never model inputs. They're only used to find disagreements, grade results
and display prices.

## The website (five pages)

| Page | What it answers |
|---|---|
| **Projections** | This week's games: projected score, win chance, the line vs the model, the model's lean in every game (a lean is not a bet), and the week's **bets** and **flags**. Tap a game for probabilities (calibrated and the model's raw chance), the worst price a bet still breaks even at, the score distribution, every book's line and price (best first), line history, and when a bet was flagged. |
| **Matchup** | Any two teams at any venue (home, away, neutral), with spread, total, win chance, fair odds, an 80% range, the chance to cover at nearby lines (with pushes on 3, 7, 10...), a what-if slider and a breakdown of why. |
| **Team ratings** | Every FBS team's power rating (points better than an average FBS team on a neutral field), movement, expected wins, offense/defense, rating history, how much of the rating is still the preseason prior, remaining schedule strength, and a conference comparison. |
| **How it works** | The method in plain language: ratings, priors, projections, probabilities, bet rules, testing, limits, and a glossary. |
| **Performance** | Live results this season (spread and total scorecards: record, units, ROI, drawdown, closing-line value), model vs market on the same games broken down by kind of game, a shadow totals challenger, the historical backtest (research, not a betting record), results by edge size, how the cautions perform, an out-of-sample test of the 7-point rule, probability calibration, model tests, every graded game (with CSV download) and data health. |

## Bet rules

- A **bet** is a 7+ point gap between the model and the middle of the sportsbooks' lines, on the
  spread or the total, with no blocking caution.
- **Cautions** (shown as flags, not bets): QB change, manual adjustment, FCS team, wind/rain (totals),
  or a 10+ point gap (usually missing news). An 8–10 point bet gets a "check the news" note.
- **Early season** (weeks 1–3) blocks bets again (Nov 2026). In those weeks the model's average miss is
  0.9 to 1.6 points worse than the closing line's, against 0.2 to 0.4 from week 7 on, and with the
  2021–2022 opening lines read, early spread gaps went 134–127. The switches are in `cfb_config.py`
  (`EARLY_SEASON_BLOCKS_SPREADS`, `EARLY_SEASON_BLOCKS_TOTALS`).
- **Near the open only** (Nov 2026). A gap becomes a bet only within `BET_WINDOW_H` (30) hours of the
  game's first quote of its own week, and never when the gap exists only because the line has since moved
  away from the model. The backtest only ever measured gaps at opening lines; against closing lines the
  same gaps lose.
- **Worth the price** (Nov 2026). A bet needs an expected value of at least `MIN_EV` (2%) at the best
  available book's real line and price, from the calibrated chance. If a whole market shows no bets, its
  calibrated chance on a 7-point gap is under break-even. Right now that is the case for spreads: with
  five seasons of opening lines, 7+ spread gaps went 224–213. Set `MIN_EV = None` to switch the floor off.
- **QB change** blocks a bet only while the change is fresh: after `QB_FRESH_STARTS` (3) starts this
  season the new quarterback no longer counts.
- Set `MY_BOOKS` in `cfb_config.py` to the books you can use and every line, best price, bet and grade
  uses only those.
- A bet keeps the line it was first flagged at and is graded there (1 unit at −110). If the line
  later moves so the gap is under 7, it shows as **LINE MOVED**: still graded, but not a bet at
  today's number.
- A bet needs a **live sportsbook quote less than 36 hours old for that market**, offered by at least one
  book in the newest snapshot. Spreads and totals are checked separately, so a fresh total never makes
  an old or pulled spread count. A game that only has the backup line from CollegeFootballData can be
  flagged but never becomes a bet.
- Each bet also shows the **best single book** for that side, chosen by expected value (line and price
  together). Grading stays at the middle of the books, which is the conservative choice.
- Every bet and blocked flag is written once, never edited, to `data/decisions.csv` (also downloadable
  from the Performance page) with its own id, the time, line, best book, price, calibrated and raw
  probability, reasons, the quote time, the model fit and the betting policy that made it (model
  version and gap thresholds). If a later policy makes a fresh decision on the same game, the new entry
  names the one it supersedes. QB-change alerts come from the previous game's box score and are not
  confirmed starters.

## How it works

1. **Data**: game results and team stats from CollegeFootballData, lines and prices from The Odds
   API, weather from Open-Meteo.
2. **Ratings**: opponent-adjusted team ratings (points, efficiency per play, pace) re-solved after
   every game, starting from preseason priors built from returning production, QB quality and
   transfers.
3. **Points**: ratings are converted to expected points for each team. Those expected points are the
   projected spread and total (the numbers the backtest scores). Win, cover and over chances come from
   a normal curve sized by past out-of-sample misses; spread pushes and the value of a half point use how
   often games really end on each margin (3 and 7 about 2.5 times a smooth curve, never 0). One function
   produces every probability, expected value and best-book choice. A score simulation gives likely scores.
4. **Service-academy correction**: Army, Navy and Air Force beat their projections in every backtest
   season, so their margins get a correction learned only from past out-of-sample misses. The backtest
   applies the same correction (from earlier seasons only), so its record matches the live system.
5. **Self-checks**: every optional input is tested on past seasons (walk-forward: each week predicted
   only from data before it, with the points conversion refit every week as the live model does). It's kept only if (1) the accuracy gain is real, meaning at least 0.01
   points with a 95% range above zero (resampling whole weeks), (2) it doesn't cost bets: units on
   7+ point gaps against opening lines can't drop by more than chance explains, and (3) it doesn't make
   the newest complete season, which played no part in (1) and (2), clearly worse. Rule (2) exists
   because some inputs make the model more accurate on every game while agreeing more with the
   market, which wins fewer bets.
6. **Records**: every prediction is frozen at kickoff and graded when the game is final.
7. **Drift checks**: the Performance page shows whether the model is leaning one way this season
   (home teams, totals, favorites) and compares the home-field edge it applies with what this season's
   results show.
8. **Challenger**: a separate totals model (more accurate in the backtest, worse bets) is recorded with
   every projection but never bet, so whether to switch can be decided on games it never saw.

## When it runs (Central time, all year)

| When | What happens |
|---|---|
| Daily 8am | Full update: results, ratings, grading, projections |
| Weekdays 6pm | New line snapshot, re-projects |
| Saturdays every 2 hours, 10am–10pm | Line snapshots close to kickoff |
| Sundays noon, 4pm, 8pm | Catch next week's opening lines (most open on Sunday; the edge is against early lines) |

GitHub schedules run in UTC, so each run is listed twice (daylight and standard time) and the first
step skips the one that doesn't match the current clock.

This stays inside both free API tiers.

## Checks before anything is published

- `run.py` stops if any projection isn't a real number in a sensible range, if team ratings are missing,
  or if this season's completed games or rated teams fell since the last update (an incomplete download).
- `cfb_selftest.py` then checks the grading rules, prices and expected value, which lines can become
  bets (including adverse cases: a fresh total next to an expired or pulled spread), that the
  probabilities fit together at every line a book could hang (win + push + loss = 1, more points never
  hurt, half a point off 3, 7, 10 or 14 turns exactly those pushes into wins, one side's win is the
  other's loss), both spellings of CollegeFootballData's line fields, best-line logic, odds matching,
  record keeping, saving and reloading the bet store and line history, the feature keep rule on known
  cases, and that the backtest can't see the future (rewriting one week's results must leave every
  earlier projection unchanged); that the decision log's ids and supersede links are intact; that every data file loads; that every page's JavaScript is valid; that the bet list adds
  up to the season totals; and that the Matchup page's formula still reproduces the weekly projections.

If any check fails, the run stops, the last good site stays up, and GitHub emails you.

## Files

Production code is everything starting with `cfb_`, plus `run.py`, `overrides.csv` and
`requirements.txt` (exact package versions, so a library update can't quietly change results). Each
run saves a fingerprint of the fitted model with every prediction, and a copy of the fit to
`data/states/` once a day, along with a summary of that day's backtest (`data/backtests/`), so a change
in method never silently rewrites the historical record. The schedule is `.github/workflows/update.yml`. Generated data lives in `data/`
and the website in `docs/`. Predictions are saved one file per season and week (`data/predictions/`)
and line snapshots one file per snapshot (`data/lines/`), so finished files never change and the
repository doesn't balloon. `data/predictions.json` and `data/line_history.csv` hold everything
recorded before October 27, 2026; they're still read, never written. Downloads retry automatically
when a data provider has a brief outage.

## Manual adjustments

If a starting QB is ruled out and you want the model to know, edit `overrides.csv` and add a line like:

```
Colorado,-5,starting QB out
```

The number is added to that team's expected points, and the game is marked low confidence (no bet).
Delete the line when it no longer applies.

## Running it by hand

Uploading any code file, `overrides.csv` or the workflow starts a full update on its own. To run one
yourself: Actions tab → **Update model** → **Run workflow**, with mode `update` (full), `odds` (lines
only) or `retune` (re-tune the prior strength).

## Privacy

The site tells search engines not to index it, but it isn't private: anyone with the link can open
it, and anyone can read a public repository. A private repository (GitHub Pro) hides the code and
data, but the published Pages site is still public to anyone with the address.

## If something breaks

Open the **Actions** tab, click the run with a red X, and copy the error text. The most common causes
are an expired or mistyped API key, or a change at one of the data providers. The Performance page's
**Data health** section shows what the latest run had to work with.

## Changelog

- **Nov 16, 2026** (from an audit of the code and data; every number below is from the walk-forward):
  - `cfb_data.lines_table` now reads the opening line from any book when the consensus row has none.
    2021 and 2022 (about 1,700 games of Bovada openers) were being skipped. With them, every 7+ gap
    against opening lines went 224–213 on spreads (was 143–102 on 2023–2025 alone).
  - Inputs are kept or dropped on accuracy alone (`cfb_model.BET_CHECK = False`, keep rule 3). The
    "doesn't cost bets" test rested on about 200 bets and penalised agreeing with the market.
  - That brings in the separate totals model (`TOTALS_MODEL = True`): total miss 13.26 → 13.19 on
    2017–2024 and 12.92 → 12.63 on 2025; actual totals now move 0.95 points per projected point
    (was 0.80). Its 7+ gaps against opening lines went 136–109 over 2021–2026, between 52% and 62% every
    season. The backtest refits the margin conversion weekly with the totals model too.
  - New bet entry rules: near the open only, not after the line moved away, 2%+ expected value at a real
    price (see Bet rules). The policy string saved with each bet now includes them, and model version is 6.
  - Big gaps are calibrated on big gaps (`SHRINK_BIG_GAP`).
  - The QB-change caution covers fresh changes only (it was blocking about 40% of November games).
  - Weeks 1–3 block bets again.
  - The Performance report gains the weight the model deserves next to the line (`market_weight`) and
    the closing-line value of past 7+ gaps (`clv_open`), both saved in `data/backtests/`.
  - The page's "past seasons" sentence now leads with the fairer numbers and the closing-line record.
  - Tested and not adopted: re-centering totals on this season's scoring each week (less accurate at
    every setting, 2018–2025). Home field was left alone: on results it is accurate (model 4.28, actual
    4.32 points on average); it only runs above the market's.
  - The first run after this upload re-tunes and re-tests (a few minutes longer than usual).

- **Oct 28, 2026** (from a third-party audit, each item checked against the code and data first):
  quote freshness is checked per market; CollegeFootballData's snake_case line fields are read; one
  probability function for every cover, push, best-book and EV number, with key-number pushes (better
  out of sample than the simulation or a smooth curve); fixed a slip that let a likely push inflate the
  published cover chance by up to about 2 points (EV by about 0.01 on average; no bet changed); the
  backtest refits the points conversion every week like the live model; decision-log ids, policy and
  supersede links; adverse-case and leakage self-tests; model-vs-market breakdown and a shadow totals
  challenger on the Performance page; "Model lean", raw vs calibrated chances, break-even prices,
  quote ages, prior share and a key-number cover table on the site. Tested and not adopted: play-count
  weights in the ratings and a team-specific prior strength (both cost bets and made 2025 worse).
  Projections and bet rules unchanged.
  Follow-up the same day: every probability now comes from one whole-number distribution per market, with
  automated checks that they fit together at every line; recorded bets compared with the line at the moment
  they were flagged; daily backtest archive. Tested and not adopted: dynamic (week-to-week drifting)
  ratings, which made totals worse at every setting and moved margin error by less than 0.01.
- **Oct 27, 2026**: redesigned site (new type, light and dark themes, team colors, an edge meter on
  every game) and a How-it-works page; early-season gaps become bets from 2027; the backtest applies
  the service-academy correction like the live projections; drift checks on the Performance page;
  smaller repository (weekly prediction files, one file per line snapshot); download retries; profit
  ranges need 20+ bets over 8+ weeks. Projections unchanged.
- **Oct 23, 2026 (later)**: feature keep rule replaced (a fixed 0.05-point bar became a test of whether
  the gain is real, plus a bet check and a confirmation season; the current inputs pass all three, so
  projections didn't change); QB changes replayed in the backtest's caution analysis (QB-change flags won
  53% of spreads vs 58% for clean bets, so the block stays); Sunday line snapshots; site fixes (edge
  column for moved bets, book order, moved-line direction, wording). Bet rules unchanged.
- **Oct 23, 2026**: service-academy correction; projected spreads and totals now equal the expected
  points the backtest validated (the simulation had been adding about half a point to totals);
  probabilities use out-of-sample error sizes; bets need a live sportsbook quote under 36 hours old;
  bets stamped at the time the odds were fetched; best book chosen by expected value; decision log;
  model fingerprints; rating ranges; accuracy, calibration and profit ranges on the Performance page;
  shadow records for other gap sizes; schedule fixed to Central time all year; pinned packages.
  Bet rules unchanged (7 points, same cautions).
- **Oct 2026**: Matchup page; Performance page (spread/total scorecards, ROI, drawdown, edge-size and
  caution analysis, out-of-sample threshold test, win-probability calibration, data health);
  best available book on every bet; sportsbook prices recorded; complete season history; rating
  history; publish checks and self-test. No change to the model or the bet rules.

Data: CollegeFootballData.com, The Odds API, Open-Meteo. For information only, not betting advice.

## Model 2.0 (research mode, Nov 2026)

Bets are suspended (`BETS_SUSPENDED = True` in `cfb_config.py`): anything that would have been a bet is logged
in `data/decisions.csv` as a flag with the reason "bets suspended (paper only)". The site shows a notice.

Model 2.0 runs after the current model on every update and publishes `docs/v2.html` ("Model 2.0" in the menu).
It never stops the current model from publishing: any error is printed and skipped.

| File | What it does |
|---|---|
| `v2_data.py` | Team-game table (scores, rushing/passing efficiency, drives, field position), personnel tables, QB changes |
| `v2_ratings.py` | 14 Kalman-filter ratings (offense and defense, each with its own uncertainty), preseason priors |
| `v2_models.py` | Models 2-5 (possession, matchup, personnel, machine learning) and Massey/Elo baselines |
| `v2_ensemble.py` | Weights for the five models plus the current model, fitted on earlier seasons' unseen forecasts; game-specific error sizes |
| `v2_sim.py` | Drive-by-drive score simulator: whole, non-negative scores; overtime; 3s and 7s |
| `v2_eval.py`, `v2_backtest.py` | Walk-forward backtest and its report |
| `v2_live.py`, `v2_site.py` | Live projections, frozen records (`data/v2/predictions_<season>.json`), grading, the page |

Seasons: 2017-2022 tune, 2023-2024 validate, 2025 test (opened once after the design was fixed), 2026 research.
Results on FBS-vs-FBS games (average miss in points; gain over the current model with a 95% week-resampled range):

| | Model 2.0 margin | v1 margin | gain | Model 2.0 total | v1 total | Closing line margin / total |
|---|---|---|---|---|---|---|
| Tune | 12.64 | 12.73 | +0.11 (0.02 to 0.20) | 12.79 | 12.84 | 12.27 / 12.33 |
| Validate | 12.42 | 12.46 | +0.04 (-0.09 to 0.15) | 12.83 | 12.83 | 12.09 / 12.75 |
| Test 2025 | 12.03 | 12.19 | +0.17 (0.00 to 0.32) | 12.75 | 12.73 | 11.81 / 12.54 |
| 2026 | 12.19 | 12.22 | +0.04 (-0.12 to 0.17) | 12.82 | 12.89 | 11.70 / 12.26 |

What did not work, kept on record: model agreement does not predict accuracy (so it changes no probability);
the possession model is the weakest component (the ensemble gives it almost no weight; it powers the score
simulator); context terms in the ensemble changed nothing. Like v1, Model 2.0 still projects totals about 0.7
points high in 2025-2026 and leans to the home side against the market. Model 2.0 replaces v1 only if it is
more accurate on games played after it went live.
