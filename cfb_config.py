"""Settings for the model. Everything you might want to tweak lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
SITE = ROOT / "docs"
RAW.mkdir(parents=True, exist_ok=True)
SITE.mkdir(exist_ok=True)

CFBD_KEY = os.environ.get("CFBD_API_KEY", "")
ODDS_KEY = os.environ.get("ODDS_API_KEY", "")
OFFLINE = os.environ.get("CFB_OFFLINE") == "1"  # use cached files only (testing)

CFBD_BASE = "https://api.collegefootballdata.com"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Seasons used to train and backtest. The first one is a burn-in season only.
FIRST_SEASON = 2015

# Edge thresholds: a game is flagged only when the model disagrees with the
# current line by at least this many points. Set from the 2017-2025 backtest:
# smaller disagreements showed no edge, even against opening lines.
SPREAD_EDGE = 7.0
TOTAL_EDGE = 7.0

# Prior-strength values tried during tuning (bigger = trust preseason priors longer).
LAMBDA_GRID = [2, 3, 4, 6, 8]

# Games where the model and the market differ by this much are marked low confidence:
# in 2018-2025 the market was closer in about two of three such games.
DISAGREE_SPREAD = 10.0  # 8-10 pt spread gaps went 51-33 vs opening lines, so they stay bets
NEWS_NOTE = 8.0  # gap where a bet gets a "worth a quick news check" reminder (does not block)
DISAGREE_TOTAL = 10.0

# The model doesn't use weather (it didn't improve backtests), but the market does. Total
# edges in games with this much wind (mph) or rain (inches, kickoff to ~halftime) are marked
# low confidence.
WEATHER_WIND = 15.0
WEATHER_RAIN = 0.10

SIMS = 10000

POWER_CONFS = {"SEC", "Big Ten", "Big 12", "ACC"}

# Separate game-level totals model (cfb_model.fit_total_model). Off: on 2017-2025 it made totals more
# accurate (MAE 13.20 -> 13.12; 2025 holdout 12.88 -> 12.63) but flagged O/U bets did worse against
# opening lines (59-45 vs 92-52), so the betting edge comes from the current totals.
TOTALS_MODEL = False

# Service academies (triple-option offenses). The ratings systematically underrate them: in the
# 2017-2025 walk-forward they beat the projected margin in every season (3 to 9 points, about 5 on
# average), and their games went about 2 points under the projected total. A correction estimated
# only from earlier seasons' out-of-sample errors (shrunk toward zero by ACADEMY_SHRINK games) cut
# the error on academy games from 14.2 to 13.5 points. Set ACADEMIES = [] to switch it off.
ACADEMIES = ["Army", "Navy", "Air Force"]
ACADEMY_SHRINK = 50

# A bet needs a live sportsbook quote: lines that only come from the CollegeFootballData backup,
# or a snapshot older than this many hours, can be shown but never become a recorded bet.
BET_MAX_QUOTE_AGE_H = 36

# Optional caution changes (both off = today's rules). In the 2023-2025 backtest against opening lines,
# 7+ point spread gaps in weeks 1-3 went 88-52 (63%) and in FCS games 61-36 (63%), vs 42-33 (56%) for
# the spread bets the rules allow (better in 2023 and 2024, worse in 2025). Against closing lines
# (2017-2025) the early-season gaps went 52-53% in total and beat 50% in 6 of 9 seasons, while the
# allowed spread bets went about 45% and were under 50% in 8 of 9. Set either to False to let those SPREAD gaps become bets
# (totals stay blocked: FCS totals went 7-13 against opening lines). Only three seasons of opening lines
# exist and they also shaped the rules, so the live shadow record is the place to confirm it.
EARLY_SEASON_BLOCKS_SPREADS = True
FCS_BLOCKS_SPREADS = True
