"""Model 2.0 data layer.

Builds one row per team per game (that team on offense) with everything known after the game, plus
season-level personnel tables known before the season. Nothing here is a model input by itself: the
rating engine (v2_ratings) turns these after-game numbers into before-game ratings, which is what keeps
every forecast free of look-ahead. Sportsbook lines are kept apart (lines_table) and are used only to
grade and compare, never to forecast.
"""
import glob
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import cfb_config as C

D1 = ("fbs", "fcs")
LOWER = "_lower division"  # every non-Division I opponent shares one rating


def _raw(name):
    p = C.RAW / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def games_table(SEASONS):
    """All games involving at least one Division I team (FBS and FCS), every season."""
    g = pd.concat([S["games_all"] for S in SEASONS.values() if len(S.get("games_all", []))], ignore_index=True)
    g = g.drop_duplicates("game_id").copy()
    for side in ("home", "away"):
        low = ~g[f"{side}_cls"].isin(D1)
        g.loc[low, side] = LOWER
        g.loc[low, f"{side}_cls"] = "lower"
    g["neutral"] = g.neutral.fillna(False).astype(bool)
    g["start"] = pd.to_datetime(g.start, utc=True)
    g = g.sort_values(["start", "game_id"]).reset_index(drop=True)
    return g


def _adv_detail(season):
    """Rushing/passing split and line play from CFBD's advanced game stats."""
    raw = _raw(f"adv_{season}") or []
    rows = []
    for r in raw:
        o = r.get("offense") or {}
        ru, pa = o.get("rushingPlays") or {}, o.get("passingPlays") or {}
        rows.append(dict(game_id=int(r.get("gameId")), team=r.get("team"),
                         plays=o.get("plays"), drives=o.get("drives"), ppa=o.get("ppa"), sr=o.get("successRate"),
                         xpl=o.get("explosiveness"), rppa=ru.get("ppa"), pppa=pa.get("ppa"),
                         rsr=ru.get("successRate"), psr=pa.get("successRate"),
                         rtot=ru.get("totalPPA"), ptot=pa.get("totalPPA"),
                         stuff=o.get("stuffRate"), line=o.get("lineYards"),
                         sd_sr=(o.get("standardDowns") or {}).get("successRate"),
                         pd_sr=(o.get("passingDowns") or {}).get("successRate")))
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    for c in d.columns:
        if c not in ("team",):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # play counts by type: totalPPA / ppa (CFBD does not publish the counts directly)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["rplays"] = (d.rtot / d.rppa).where(d.rppa.abs() > 1e-6)
        d["pplays"] = (d.ptot / d.pppa).where(d.pppa.abs() > 1e-6)
    tot = d.rplays.fillna(0) + d.pplays.fillna(0)
    d["rshare"] = (d.rplays / tot).where(tot > 10)
    return d.drop(columns=["rtot", "ptot"])


def _drives(season):
    parts = []
    for f in sorted(glob.glob(str(C.RAW / f"pbp_{season}_*.csv"))):
        try:
            parts.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            continue
    if not parts:
        return pd.DataFrame(columns=["game_id", "team", "ng_drives", "ng_pts", "opps", "opp_pts", "fpos"])
    d = pd.concat(parts, ignore_index=True).rename(columns={"off": "team", "start": "fpos"})
    return d.drop_duplicates(["game_id", "team"])


def team_games(G):
    """One row per (game, team on offense) for completed games."""
    done = G[G.completed & G.home_pts.notna() & G.away_pts.notna()]
    rows = []
    for side, other, sgn in (("home", "away", 1), ("away", "home", -1)):
        x = pd.DataFrame(dict(game_id=done.game_id, season=done.season, wk=done.wk, start=done.start,
                              team=done[side], opp=done[other], cls=done[f"{side}_cls"], opp_cls=done[f"{other}_cls"],
                              h=np.where(done.neutral, 0, sgn), pts=done[f"{side}_pts"], opp_pts=done[f"{other}_pts"]))
        rows.append(x)
    tg = pd.concat(rows, ignore_index=True)
    det = []
    for s in sorted(tg.season.unique()):
        a = _adv_detail(s)
        dr = _drives(s)
        if len(a):
            det.append(("adv", a))
        if len(dr):
            det.append(("drv", dr))
    adv = pd.concat([a for k, a in det if k == "adv"], ignore_index=True) if any(k == "adv" for k, _ in det) else None
    drv = pd.concat([a for k, a in det if k == "drv"], ignore_index=True) if any(k == "drv" for k, _ in det) else None
    if adv is not None:
        tg = tg.merge(adv.drop_duplicates(["game_id", "team"]), on=["game_id", "team"], how="left")
    if drv is not None:
        drv = drv.rename(columns={"opp_pts": "zone_pts"})  # points from scoring opportunities
        tg = tg.merge(drv[["game_id", "team", "ng_drives", "ng_pts", "opps", "zone_pts", "fpos"]],
                      on=["game_id", "team"], how="left")
    # derived per-drive rates (only with enough drives to mean something)
    ok = tg.ng_drives >= 4
    tg["ppd"] = (tg.ng_pts / tg.ng_drives).where(ok)
    tg["opp_rate"] = (tg.opps / tg.ng_drives).where(ok)
    tg["fin"] = (tg.zone_pts / tg.opps).where(ok & (tg.opps >= 2))
    tg["fpos"] = tg.fpos.where(ok)
    return tg.sort_values(["start", "game_id"]).reset_index(drop=True)


def lines_table(SEASONS):
    """CFBD consensus lines (home spread, negative = home favored). Evaluation only."""
    L = pd.concat([S["lines"] for S in SEASONS.values() if len(S.get("lines", []))], ignore_index=True)
    return L.drop_duplicates("game_id")


# ---------------- Personnel (known before the season) ----------------

def _recruit_composite(season, window=4):
    """Average recruit rating of each team's last `window` signing classes, weighted by count."""
    pts = defaultdict(list)
    for y in range(season - window, season):
        for r in _raw(f"recruits_{y}") or []:
            if r.get("committedTo") and r.get("rating"):
                pts[r["committedTo"]].append(float(r["rating"]))
    return {t: (float(np.mean(v)), len(v)) for t, v in pts.items()}


def _portal(season):
    """Transfers into and out of each team before `season` (counts and average stars of arrivals)."""
    inn, out = defaultdict(list), defaultdict(int)
    for r in _raw(f"portal_{season}") or []:
        if r.get("destination"):
            inn[r["destination"]].append(float(r.get("stars") or 2.0))
        if r.get("origin"):
            out[r["origin"]] += 1
    return {t: (len(v), float(np.mean(v))) for t, v in inn.items()}, dict(out)


def _experience(season):
    """Share of each roster in its 3rd year or later (CFBD roster 'year')."""
    tot, old = defaultdict(int), defaultdict(int)
    for p in _raw(f"roster_{season}") or []:
        y = p.get("year")
        if y is None:
            continue
        tot[p["team"]] += 1
        old[p["team"]] += int(y >= 3)
    return {t: old[t] / tot[t] for t in tot if tot[t] >= 40}


def _returning(season):
    out = {}
    for r in _raw(f"returning_{season}") or []:
        out[r["team"]] = dict(ret_all=r.get("percentPPA"), ret_pass=r.get("percentPassingPPA"),
                              ret_rush=r.get("percentRushingPPA"), ret_rec=r.get("percentReceivingPPA"))
    return out


def personnel(seasons, pfeat=None):
    """{season: DataFrame(team, talent, rec, rec_n, tin, tin_stars, tout, exp, ret_*, qbd, qbl, dret, oret)}.
    pfeat: the v1 player tables (cfb_players.build) - QB quality and returning production counting
    transfers - reused rather than rebuilt."""
    out = {}
    for s in seasons:
        tal = {r["team"]: float(r["talent"]) for r in (_raw(f"talent_{s}") or []) if r.get("talent")}
        rec = _recruit_composite(s)
        tin, tout = _portal(s)
        exp = _experience(s)
        ret = _returning(s)
        pf = (pfeat or {}).get(s, {})
        teams = set(tal) | set(rec) | set(ret) | set(pf)
        rows = []
        for t in teams:
            r = dict(team=t, talent=tal.get(t), rec=rec.get(t, (None, 0))[0], rec_n=rec.get(t, (None, 0))[1],
                     tin=tin.get(t, (0, None))[0], tin_stars=tin.get(t, (0, None))[1], tout=tout.get(t, 0),
                     exp=exp.get(t))
            r.update(ret.get(t, {}))
            r.update({k: v for k, v in (pf.get(t) or {}).items() if k in ("qbd", "qbl", "dret", "oret")})
            rows.append(r)
        df = pd.DataFrame(rows)
        for c in ("talent", "rec", "rec_n", "tin", "tin_stars", "tout", "exp", "ret_all", "ret_pass", "ret_rush",
                  "ret_rec", "qbd", "qbl", "dret", "oret"):
            if c not in df:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
        out[s] = df
    return out


# ---------------- In-season QB (from the previous game's box score) ----------------

def qb_context(G, passing):
    """For every game and side: did the team's most recent starter differ from its regular starter this
    season, and his passing quality (adjusted yards per attempt, shrunk). Uses only earlier games.
    No injury reports or depth charts exist in this data, so this is the only in-season personnel signal."""
    import cfb_extra as Ex
    if passing is None or len(passing) == 0:
        return pd.DataFrame(dict(game_id=G.game_id, qbchg_h=0.0, qbchg_a=0.0, qbq_h=np.nan, qbq_a=np.nan))
    d = Ex.qb_deltas(G[["game_id", "season", "start", "home", "away", "completed"]].copy(), passing)
    d = d.rename(columns={"qbd_h": "qbchg_h", "qbd_a": "qbchg_a"})
    return d[["game_id", "qbchg_h", "qbchg_a"]]
