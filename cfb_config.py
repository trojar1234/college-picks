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

# Wind: each team's expected points shrink by this fraction per mph above WIND_START.
# A fixed starting value, not fitted. Check the "wind" row in the error report.
WIND_START = 10.0
WIND_PER_MPH = 0.006

SIMS = 10000

POWER_CONFS = {"SEC", "Big Ten", "Big 12", "ACC"}
