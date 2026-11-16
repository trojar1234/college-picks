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

# Separate game-level totals model (cfb_model.fit_total_model). True = it is offered to the feature test
# and used if it passes on accuracy (it does: total MAE 13.26 -> 13.19 on 2017-2024, 12.92 -> 12.63 on the
# 2025 holdout, and actual totals now move 0.95 points per projected point instead of 0.80). Until Nov 2026
# it was held out because its flagged O/U bets did worse against 2023-2025 opening lines (59-45 vs 92-52).
# With the 2021-2022 opening lines read as well, its 7+ point gaps are the steadier of the two: 136-109,
# between 52% and 62% in each of six seasons, against 183-136 that ranged from 45% to 65%.
TOTALS_MODEL = True

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

# Early-season bets. Until Oct 2026 every 7+ point gap in weeks 1-3 (before the typical FBS team has played
# 3 games) was blocked. The evidence says the block costs money: early gaps that no other caution touches
# (gap under 10, no FCS team, no QB change) did at least as well as the bets the rules allowed, in both markets.
#                         early (weeks 1-3)             allowed (after week 3)
#   spreads vs opening    31-20 (61%)                   31-28 (53%)
#   spreads vs closing    35-25 (58%)                   26-31 (46%)
#   totals vs opening     24-8  (75%)                   41-24 (63%)
#   totals vs closing     16-13 (55%)                   31-30 (51%)
# (walk-forward 2023-2026 with the live model; 2026 was never used to set any rule and agreed: spreads 9-6,
# totals 5-0 against opening lines.) From 2027 week 1 these become bets. Set either back to True to restore
# the old rule for that market.
# Nov 2026: the block is back on. The table above is 50 and 30 bets from the same seasons that shaped the
# rules. The larger, steadier fact is accuracy: in weeks 1-3 the model's average miss is 0.9 to 1.6 points
# worse than the closing line's (7,700 games, 2017-2025), against 0.2 to 0.4 from week 7 on. With the
# 2021-2022 opening lines read, weeks 1-3 spread gaps went 134-127. Set to False to allow them again.
EARLY_SEASON_BLOCKS_SPREADS = True
EARLY_SEASON_BLOCKS_TOTALS = True
# FCS games: spread gaps went 9-5 against opening lines but 8-8 against closing lines (2023-2026); too few to
# change the rule, so FCS games still block spread bets. FCS totals went 7-13 and stay blocked.
FCS_BLOCKS_SPREADS = True

# ---- Bet entry rules added Nov 2026 (each shows as a flag with its reason when it blocks a bet) ----
# A bet must be taken near the open. The backtest only ever measured gaps against opening lines; against
# closing lines the same gaps lose (455-466 spreads, 339-380 totals). So a gap only becomes a bet within
# this many hours of the game's first usable quote (the first snapshot after both teams' previous games).
# None = no time limit (the rule before Nov 2026).
BET_WINDOW_H = 30  # Sunday morning's first quote through Monday morning's full update
# ...and never when the gap only exists because the line has moved AWAY from the model since that first
# quote (the market acting on news the model can't see). In the backtest, 7+ spread gaps where the line
# later moved away from the model went 50-48; where it moved toward the model, 126-78.
BLOCK_MOVED_AWAY = True
# Minimum expected value (profit per unit risked) at the best available book's actual line and price,
# using the calibrated probability. 0.02 = 2%. None = no floor. Note what this does: if the calibrated
# chance on a 7-point gap is below about 53.4%, no 7-point gap in that market becomes a bet. That is the
# model's own maths saying the gap isn't worth the price, not a fault.
MIN_EV = 0.02
# Calibration by gap size. One shrink factor for every game understates big total gaps and overstates
# big spread gaps (2021-2025 opening lines: spreads 0.10 on all games and 0.08 on 5+ point gaps, totals
# 0.31 and 0.35; with the old totals it was 0.16 and 0.30). Games
# with a gap of at least this many points use a factor measured on such games only. None = one factor.
SHRINK_BIG_GAP = 5.0
# QB-change caution: only a FRESH change blocks a bet. A stand-in who has now started this many games
# this season is the team's quarterback, and the ratings have seen him. (The old rule kept a team
# flagged until the new starter passed the old one in season attempts: 40% of games by November.)
QB_FRESH_STARTS = 3
# Books you can actually bet at, by the names in the line files (e.g. ["DraftKings", "FanDuel", "BetMGM"]).
# When set, the consensus line, the best book, bets and grading use only these. Empty = every book.
MY_BOOKS = []
