# College football model — setup guide

This project predicts every FBS game, tracks betting lines automatically, grades
itself, and publishes a website that updates on its own. Setup takes about 30
minutes, once. After that there's nothing you have to do.

## What you'll need

- A free **CollegeFootballData** API key: go to collegefootballdata.com, choose
  "Get API Key", and enter your email. The key arrives by email.
- A free **The Odds API** key: go to the-odds-api.com, pick the free "Starter"
  plan, and enter your email. The key arrives by email.
- A free **GitHub** account: github.com → Sign up.

## Step 1 — Create the repository

1. On GitHub, click **+** (top right) → **New repository**.
2. Name it something nondescript, like `weekend-numbers`.
3. Choose **Public** (required for free GitHub Pages; see "Privacy" below).
4. Click **Create repository**.

## Step 2 — Upload the files

1. On the new repository's page, click **uploading an existing file**.
2. Unzip `cfb-model.zip` on your computer, open the `cfb-model` folder, select
   everything inside it, and drag it into the browser window.
3. Click **Commit changes**.

Your computer may hide the `.github` folder, so add the schedule file by hand:

4. Click **Add file → Create new file**.
5. In the name box, type exactly: `.github/workflows/update.yml`
6. Paste the contents of `workflow-update.yml` (included in the zip) into the
   big text box, then click **Commit changes**.

## Step 3 — Add your API keys

1. In the repository, go to **Settings → Secrets and variables → Actions**.
2. Click **New repository secret**. Name: `CFBD_API_KEY`, value: your
   CollegeFootballData key. Save.
3. Click **New repository secret** again. Name: `ODDS_API_KEY`, value: your
   Odds API key. Save.

## Step 4 — Turn on the website

1. Go to **Settings → Pages**.
2. Under "Build and deployment", set **Source** to *Deploy from a branch*.
3. Set **Branch** to `main` and the folder to `/docs`. Save.
4. After a minute the page shows your site's address, something like
   `https://yourname.github.io/weekend-numbers/`. Bookmark it.

## Step 5 — Run it the first time

1. Go to the **Actions** tab. If GitHub asks, click to enable workflows.
2. Click **Update model** on the left, then **Run workflow → Run workflow**.
3. The first run downloads about ten past seasons and tunes the model, which
   takes 5–10 minutes. When it shows a green check, refresh your site.

That's it. From then on it runs itself:

| When (Central time) | What happens |
|---|---|
| Every day, ~8am | Pulls results, updates ratings, grades predictions, re-projects the week |
| Weekdays, ~6pm | New line snapshot, re-projects |
| Saturdays, every 2 hours 10am–10pm | Line snapshots, so closing lines are captured |

These stay inside both free tiers (about 150 of CollegeFootballData's 1,000
monthly calls and about 170 of The Odds API's 500 monthly credits).

## Optional — manual adjustments

If a starting QB is ruled out and you want the model to know, edit
`data/overrides.csv` on GitHub (open the file, click the pencil icon) and add a
line like:

```
Colorado,-5,starting QB out
```

The number is added to that team's expected points. The game will be marked
low confidence. Delete the line when it no longer applies.

## Reading the site

- **Projections**: every FBS game this week, the model's score and line next to
  the market's, line movement, and flagged edges. Click any game for score
  probabilities, the margin distribution, inputs, and lines by book.
- **Report card**: this season's live results (graded at kickoff, including
  closing line value), the walk-forward backtest by season, calibration, error
  diagnostics by segment, and team ratings.

Trust the model only where the report card shows it earns it. The most reliable
sign of a real edge is consistently beating the closing line (positive average
CLV), not the win-loss record over a few weeks.

## How it learns

- **Every run**: team ratings are re-solved with all completed games, so each
  result moves the ratings, weighted against the preseason prior.
- **Every run**: the conversion from ratings to points, the simulator's
  variance, and the calibration of cover probabilities are refit on every graded
  game, historical and current.
- **Monthly**: the prior strength (how long preseason expectations matter) is
  re-tuned on the full backtest. Run the workflow with mode `retune` to force it.
- **Error diagnostics** flag systematic biases (by week, matchup type, favorite
  size) so you can see whether something needs a structural fix.

Betting lines are stored and displayed but never used as model inputs.

## Privacy

The site isn't linked or searchable anywhere, and it tells search engines not to
index it. But because the repository is public, anyone who finds it on GitHub
can see the code and data. If you later want it fully private, GitHub Pro
($4/month) lets you make the repository private.

## If something breaks

Open the **Actions** tab, click the run with a red X, and copy the error text.
Paste it to Claude and it can be fixed. The most common causes are an expired or
mistyped API key, or a change in one of the data providers.

Data provided by CollegeFootballData.com, The Odds API, and Open-Meteo.
