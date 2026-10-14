# College football model

A market-blind college football model. It projects every FBS game, compares its numbers with the
sportsbooks, records bets before kickoff, grades itself, and publishes a website that updates on
its own through GitHub Actions and GitHub Pages.

Sportsbook lines are never model inputs. They're only used to find disagreements, grade results
and display prices.

## The website (four pages)

| Page | What it answers |
|---|---|
| **Projections** | This week's games: projected score, win chance, the line vs the model, the model's pick in every game, and the week's **bets** and **flags**. Tap a game for probabilities, the score distribution, every book's line and price (best first), line history, and when a bet was flagged. |
| **Matchup** | Any two teams at any venue (home, away, neutral), with spread, total, win chance, fair odds, an 80% range, a what-if slider and a breakdown of why. |
| **Team ratings** | Every FBS team's power rating (points better than an average FBS team on a neutral field), movement, expected wins, offense/defense, rating history, remaining schedule strength, and a conference comparison. |
| **Performance** | Live results this season (spread and total scorecards: record, units, ROI, drawdown, closing-line value), the historical backtest, results by edge size, how the cautions perform, an out-of-sample test of the 7-point rule, probability calibration, model tests, every graded game (with CSV download) and data health. |

## Bet rules

- A **bet** is a 7+ point gap between the model and the middle of the sportsbooks' lines, on the
  spread or the total, with no blocking caution.
- **Cautions** (shown as FLAG, not bets): QB change, manual adjustment, FCS team, early season,
  wind/rain, or a 10+ point gap (usually missing news). An 8–10 point bet gets a "check the news" note.
- A bet keeps the line it was first flagged at and is graded there (1 unit at −110). If the line
  later moves so the gap is under 7, it shows as **LINE MOVED**: still graded, but not a bet at
  today's number.
- Each bet also shows the **best single book** for that side. Grading stays at the middle of the
  books, which is the conservative choice.

## How it works

1. **Data**: game results and team stats from CollegeFootballData, lines and prices from The Odds
   API, weather from Open-Meteo.
2. **Ratings**: opponent-adjusted team ratings (points, efficiency per play, pace) re-solved after
   every game, starting from preseason priors built from returning production, QB quality and
   transfers.
3. **Points**: ratings are converted to expected points for each team, then a score simulation
   (10,000 games) gives win and cover probabilities.
4. **Self-checks**: every option is tested on past seasons and kept only if it lowers error on seasons
   the model hasn't seen. The backtest is walk-forward: each week predicted only from data before it.
5. **Records**: every prediction is frozen at kickoff and graded when the game is final.

## When it runs

GitHub schedules run in UTC, so Central times move an hour earlier from early November to mid-March.

| When (Central, daylight time / standard time) | What happens |
|---|---|
| Daily ~8am / ~7am | Full update: results, ratings, grading, projections |
| Weekdays ~6pm / ~5pm | New line snapshot, re-projects |
| Saturdays every 2 hours 10am–6pm / 9am–5pm, plus 8pm and 10pm / 7pm and 9pm | Line snapshots close to kickoff |

This stays inside both free API tiers.

## Checks before anything is published

- `run.py` stops if any projection isn't a real number in a sensible range, or if team ratings are missing.
- `cfb_selftest.py` then checks the grading rules, best-line logic, odds matching and record keeping on
  known cases; that every data file loads; that every page's JavaScript is valid; that the bet list adds
  up to the season totals; and that the Matchup page's formula still reproduces the weekly projections.

If any check fails, the run stops, the last good site stays up, and GitHub emails you.

## Files

Production code is everything starting with `cfb_`, plus `run.py`, `overrides.csv` and
`requirements.txt`. The schedule is `.github/workflows/update.yml`. Generated data lives in `data/`
(predictions, line history, model state) and the website in `docs/`.

Older files from the first version are no longer used and can be deleted: `config.py`, `data.py`,
`fetch.py`, `lines.py`, `model.py`, `predict.py`, `ratings.py`, `report.py`, `sim.py`, `site.py`,
`fake_data.py`, `index.html`, `update.yml` (the root copy), `workflow-update.yml`, `download` and
`download (1)`.

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

- **Oct 2026**: Matchup page; Performance page (spread/total scorecards, ROI, drawdown, edge-size and
  caution analysis, out-of-sample threshold test, win-probability calibration, data health);
  best available book on every bet; sportsbook prices recorded; complete season history; rating
  history; publish checks and self-test. No change to the model or the bet rules.

Data: CollegeFootballData.com, The Odds API, Open-Meteo. For information only, not betting advice.
