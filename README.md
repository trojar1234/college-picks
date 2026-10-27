# College football model

A market-blind college football model. It projects every FBS game, compares its numbers with the
sportsbooks, records bets before kickoff, grades itself, and publishes a website that updates on
its own through GitHub Actions and GitHub Pages.

Sportsbook lines are never model inputs. They're only used to find disagreements, grade results
and display prices.

## The website (five pages)

| Page | What it answers |
|---|---|
| **Projections** | This week's games: projected score, win chance, the line vs the model, the model's pick in every game, and the week's **bets** and **flags**. Tap a game for probabilities, the score distribution, every book's line and price (best first), line history, and when a bet was flagged. |
| **Matchup** | Any two teams at any venue (home, away, neutral), with spread, total, win chance, fair odds, an 80% range, a what-if slider and a breakdown of why. |
| **Team ratings** | Every FBS team's power rating (points better than an average FBS team on a neutral field), movement, expected wins, offense/defense, rating history, remaining schedule strength, and a conference comparison. |
| **How it works** | The method in plain language: ratings, priors, projections, probabilities, bet rules, testing, limits, and a glossary. |
| **Performance** | Live results this season (spread and total scorecards: record, units, ROI, drawdown, closing-line value), the historical backtest, results by edge size, how the cautions perform, an out-of-sample test of the 7-point rule, probability calibration, model tests, every graded game (with CSV download) and data health. |

## Bet rules

- A **bet** is a 7+ point gap between the model and the middle of the sportsbooks' lines, on the
  spread or the total, with no blocking caution.
- **Cautions** (shown as flags, not bets): QB change, manual adjustment, FCS team, wind/rain (totals),
  or a 10+ point gap (usually missing news). An 8–10 point bet gets a "check the news" note.
- **Early season** (weeks 1–3) no longer blocks bets, starting with the 2027 season. In the 2023–2026
  walk-forward, early-season gaps with no other caution did at least as well as later bets in both
  markets (spreads 31–20 vs open and 35–25 vs close; totals 24–8 and 16–13). The switches are in
  `cfb_config.py` (`EARLY_SEASON_BLOCKS_SPREADS`, `EARLY_SEASON_BLOCKS_TOTALS`) if you want the old rule.
- A bet keeps the line it was first flagged at and is graded there (1 unit at −110). If the line
  later moves so the gap is under 7, it shows as **LINE MOVED**: still graded, but not a bet at
  today's number.
- A bet needs a **live sportsbook quote less than 36 hours old**. A game that only has the backup line
  from CollegeFootballData can be flagged but never becomes a bet.
- Each bet also shows the **best single book** for that side, chosen by expected value (line and price
  together). Grading stays at the middle of the books, which is the conservative choice.
- Every bet and blocked flag is written once, never edited, to `data/decisions.csv` (also downloadable
  from the Performance page) with the time, line, best book, price, probability, reasons and model fit.

## How it works

1. **Data**: game results and team stats from CollegeFootballData, lines and prices from The Odds
   API, weather from Open-Meteo.
2. **Ratings**: opponent-adjusted team ratings (points, efficiency per play, pace) re-solved after
   every game, starting from preseason priors built from returning production, QB quality and
   transfers.
3. **Points**: ratings are converted to expected points for each team. Those expected points are the
   projected spread and total (the numbers the backtest scores). A score simulation (10,000 games)
   gives the shape of the outcomes: win, cover and push chances, likely scores.
4. **Service-academy correction**: Army, Navy and Air Force beat their projections in every backtest
   season, so their margins get a correction learned only from past out-of-sample misses. The backtest
   applies the same correction (from earlier seasons only), so its record matches the live system.
5. **Self-checks**: every optional input is tested on past seasons (walk-forward: each week predicted
   only from data before it). It's kept only if (1) the accuracy gain is real, meaning at least 0.01
   points with a 95% range above zero (resampling whole weeks), (2) it doesn't cost bets: units on
   7+ point gaps against opening lines can't drop by more than chance explains, and (3) it doesn't make
   the newest complete season, which played no part in (1) and (2), clearly worse. Rule (2) exists
   because some inputs make the model more accurate on every game while agreeing more with the
   market, which wins fewer bets.
6. **Records**: every prediction is frozen at kickoff and graded when the game is final.
7. **Drift checks**: the Performance page shows whether the model is leaning one way this season
   (home teams, totals, favorites) and compares the home-field edge it applies with what this season's
   results show.

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

- `run.py` stops if any projection isn't a real number in a sensible range, or if team ratings are missing.
- `cfb_selftest.py` then checks the grading rules, prices and expected value, which lines can become
  bets, best-line logic, odds matching, record keeping, saving and reloading the bet store and line
  history, and the feature keep rule on known cases; that every data file loads; that every page's JavaScript is valid; that the bet list adds
  up to the season totals; and that the Matchup page's formula still reproduces the weekly projections.

If any check fails, the run stops, the last good site stays up, and GitHub emails you.

## Files

Production code is everything starting with `cfb_`, plus `run.py`, `overrides.csv` and
`requirements.txt` (exact package versions, so a library update can't quietly change results). Each
run saves a fingerprint of the fitted model with every prediction, and a copy of the fit to
`data/states/` once a day. The schedule is `.github/workflows/update.yml`. Generated data lives in `data/`
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

Actions tab → **Update model** → **Run workflow**, with mode `update` (full), `odds` (lines only) or
`retune` (re-tune the prior strength).

## Privacy

The site tells search engines not to index it, but it isn't private: anyone with the link can open
it, and anyone can read a public repository. A private repository (GitHub Pro) hides the code and
data, but the published Pages site is still public to anyone with the address.

## If something breaks

Open the **Actions** tab, click the run with a red X, and copy the error text. The most common causes
are an expired or mistyped API key, or a change at one of the data providers. The Performance page's
**Data health** section shows what the latest run had to work with.

## Changelog

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
