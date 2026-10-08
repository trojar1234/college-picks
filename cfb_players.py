"""Player-level preseason information, built from free CollegeFootballData data.

For each season and team:
  qbd    - quality of this season's expected starting QB minus last season's starter
           (career efficiency at any school, so transfers count)
  qbl    - quality of the expected starter
  dret   - share of defensive production returning, counting incoming transfers the way
           SP+ does (transfers are added to both the top and bottom of the ratio)
  oret   - same for offensive production, from player EPA (when that data is available)

All inputs are cached in data/raw; missing data just leaves a feature blank.
"""
import glob
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import cfb_config as C
import cfb_fetch as F

QB_PRIOR_ATT, QB_PRIOR_YPA = 150.0, 5.0  # an unknown QB is assumed to be a below-average backup
DEF_W = {"TOT": 1.0, "SOLO": 0.0, "TFL": 2.0, "SACKS": 2.0, "PD": 2.0, "QB HUR": 1.0}


def _load(name):
    p = C.RAW / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _passing_rows():
    """Every QB passing line from the cached weekly box scores."""
    import cfb_extra as Ex
    rows = []
    for f in sorted(glob.glob(str(C.RAW / "passing_*.json"))):
        m = re.search(r"passing_(\d{4})_", f)
        season = int(m.group(1))
        try:
            for r in Ex._parse_passing(json.loads(open(f).read())):
                r["season"] = season
                rows.append(r)
        except Exception:
            continue
    return pd.DataFrame(rows)


def _quality(att, adj):
    return (adj + QB_PRIOR_YPA * QB_PRIOR_ATT) / (att + QB_PRIOR_ATT)


def qb_features(seasons, passing):
    """{season: {team: (qbd, qbl)}} using rosters to find each team's QBs."""
    out = {}
    if passing is None or passing.empty:
        return out
    by_qb_season = passing.groupby(["qb", "season"]).agg(att=("att", "sum"), adj=("adj", "sum")).reset_index()
    team_qb = passing.groupby(["season", "team", "qb"]).att.sum().reset_index()
    for s in seasons:
        roster = _load(f"roster_{s}")
        if not roster:
            continue
        hist = by_qb_season[by_qb_season.season < s]
        car = hist.groupby("qb").agg(att=("att", "sum"), adj=("adj", "sum"))
        last = by_qb_season[by_qb_season.season == s - 1].set_index("qb").att
        qbs = defaultdict(list)
        for p in roster:
            if str(p.get("position", "")).upper() == "QB":
                qbs[p.get("team")].append(str(p.get("id")))
        prev = team_qb[team_qb.season == s - 1]
        prev_starter = prev.sort_values("att", ascending=False).drop_duplicates("team").set_index("team").qb
        res = {}
        for team, ids in qbs.items():
            cand = [(last.get(q, 0.0), car.att.get(q, 0.0), q) for q in ids]
            cand.sort(reverse=True)
            best = cand[0][2] if cand and (cand[0][0] > 0 or cand[0][1] > 0) else None
            q_new = _quality(car.att.get(best, 0.0), car.adj.get(best, 0.0)) if best else QB_PRIOR_YPA
            old = prev_starter.get(team)
            if old is None:
                continue
            old_hist = by_qb_season[(by_qb_season.qb == old) & (by_qb_season.season <= s - 1)]
            q_old = _quality(old_hist.att.sum(), old_hist.adj.sum())
            res[team] = (q_new - q_old, q_new)
        out[s] = res
    return out


def _def_production(s):
    stats = _load(f"defstats_{s}")
    if not stats:
        return None
    prod, team_of = defaultdict(float), {}
    for st in stats:
        w = DEF_W.get(str(st.get("statType", "")).upper())
        if not w:
            continue
        try:
            v = float(st.get("stat") or 0)
        except ValueError:
            continue
        pid = str(st.get("playerId"))
        prod[pid] += w * v
        team_of[pid] = st.get("team")
    return prod, team_of


def _off_production(s):
    """Last season's offensive production per player from CFBD player EPA (totalPPA.all)."""
    data = _load(f"playerppa_{s}")
    if not data:
        return None
    prod, team_of = {}, {}
    for p in data:
        tot = (p.get("totalPPA") or {}).get("all")
        if tot is None:
            continue
        pid = str(p.get("id"))
        prod[pid] = max(0.0, float(tot))  # negative production isn't "lost" when a player leaves
        team_of[pid] = p.get("team")
    return prod, team_of


def returning_share(s, production):
    """SP+-style returning production: (returners + incoming) / (last year's total + incoming)."""
    if production is None:
        return {}
    prod, team_of = production
    roster = _load(f"roster_{s}")
    if not roster:
        return {}
    now = {str(p.get("id")): p.get("team") for p in roster}
    total, kept, incoming = defaultdict(float), defaultdict(float), defaultdict(float)
    for pid, v in prod.items():
        old = team_of.get(pid)
        total[old] += v
        new = now.get(pid)
        if new is None:
            continue
        if new == old:
            kept[old] += v
        else:
            incoming[new] += v
    out = {}
    for team in set(total) | set(incoming):
        den = total.get(team, 0.0) + incoming.get(team, 0.0)
        if den > 0:
            out[team] = (kept.get(team, 0.0) + incoming.get(team, 0.0)) / den
    return out


PLAYER_BUDGET = 40  # max new CollegeFootballData calls per run for player data


def ensure_data(seasons, current, first_qb_season):
    """Download (within budget, cached forever for past seasons) the player data the
    features need: rosters, last season's defensive stats and player EPA, and QB box scores."""
    import cfb_extra as Ex
    budget = Ex.Budget(PLAYER_BUDGET if not C.OFFLINE else 0)
    missing = {"core": [], "off": []}  # "off" = player EPA, only needed for the offense option
    for s in seasons:
        for endpoint, params, name in (
            ("/roster", {"year": s}, f"roster_{s}"),
            ("/stats/player/season", {"year": s - 1, "category": "defensive"}, f"defstats_{s - 1}"),
            ("/ppa/players/season", {"year": s - 1}, f"playerppa_{s - 1}"),
        ):
            if _load(name) is not None:
                continue
            bucket = missing["off"] if name.startswith("playerppa") else missing["core"]
            try:
                budget.get(endpoint, params, name)
            except Ex.BudgetExhausted:
                bucket.append(name)
            except Exception as e:  # one failed endpoint shouldn't stop the others
                bucket.append(f"{name} ({type(e).__name__})")
    # QB box scores for every season (most are already cached by earlier runs)
    games = []
    for s in seasons:
        sd = F._load(f"games_{s}")
        if sd:
            import cfb_data as Dt
            games.append(Dt.games_table(sd, s))
    if games:
        allg = pd.concat(games, ignore_index=True)
        allg = allg[(allg.home_cls == "fbs") | (allg.away_cls == "fbs")]
        try:
            _, miss = Ex.qb_table(allg, budget, first_qb_season, current)
            if miss:
                missing["core"].append(f"{miss} weeks of QB box scores")
        except Exception as e:
            missing["core"].append(f"QB box scores ({type(e).__name__})")
    return missing


def build(seasons):
    """{season: {team: {qbd, qbl, dret, oret, hc}}} and a status string."""
    status = []
    try:
        passing = _passing_rows()
    except Exception as e:
        passing, _ = None, status.append(f"QB box scores failed: {e}")
    qb = qb_features(seasons, passing)
    out = defaultdict(dict)
    have_def = have_off = 0
    for s in seasons:
        dr = returning_share(s, _def_production(s - 1))
        orr = returning_share(s, _off_production(s - 1))
        have_def += bool(dr)
        have_off += bool(orr)
        teams = set(qb.get(s, {})) | set(dr) | set(orr)
        for t in teams:
            q = qb.get(s, {}).get(t)
            out[s][t] = dict(qbd=q[0] if q else np.nan, qbl=q[1] if q else np.nan,
                             dret=dr.get(t, np.nan), oret=orr.get(t, np.nan))
    counts = dict(qb=sum(bool(qb.get(s)) for s in seasons), defense=have_def, offense=have_off)
    status.append(f"seasons with QB data {counts['qb']}, defense {counts['defense']}, offense {counts['offense']}")
    return {s: dict(v) for s, v in out.items()}, "; ".join(status), counts
