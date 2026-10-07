"""Turns raw API responses into tidy tables."""
import numpy as np
import pandas as pd

import cfb_config as C


def _g(d, *keys, default=None):
    """Get the first present key (CFBD v2 uses camelCase, v1 snake_case)."""
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def week_key(week, season_type):
    return int(week) + (20 if str(season_type).startswith("post") else 0)


def games_table(raw, season):
    rows = []
    for d in raw:
        hc = str(_g(d, "homeClassification", "home_division", default="") or "").lower()
        ac = str(_g(d, "awayClassification", "away_division", default="") or "").lower()
        if "fbs" not in (hc, ac):
            continue
        st = _g(d, "seasonType", "season_type", default="regular")
        rows.append(
            dict(
                game_id=int(_g(d, "id")),
                season=season,
                week=int(_g(d, "week", default=0)),
                wk=week_key(_g(d, "week", default=0), st),
                start=pd.to_datetime(_g(d, "startDate", "start_date"), utc=True, errors="coerce"),
                neutral=bool(_g(d, "neutralSite", "neutral_site", default=False)),
                venue_id=_g(d, "venueId", "venue_id"),
                home=_g(d, "homeTeam", "home_team"),
                away=_g(d, "awayTeam", "away_team"),
                home_conf=_g(d, "homeConference", "home_conference", default="") or "",
                away_conf=_g(d, "awayConference", "away_conference", default="") or "",
                home_cls=hc,
                away_cls=ac,
                home_pts=_num(_g(d, "homePoints", "home_points")),
                away_pts=_num(_g(d, "awayPoints", "away_points")),
                completed=bool(_g(d, "completed", default=False)),
            )
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.loc[df.home_pts.isna() | df.away_pts.isna(), "completed"] = False
    return df.sort_values(["start", "game_id"]).reset_index(drop=True)


def adv_table(raw, games):
    """One row per team per game: offensive PPA (EPA/play) and play count."""
    hmap = games.set_index("game_id")[["home", "neutral"]].to_dict("index")
    rows = []
    for d in raw:
        gid = _g(d, "gameId", "game_id")
        if gid is None or int(gid) not in hmap:
            continue
        gid = int(gid)
        off = _g(d, "offense", default={}) or {}
        ppa, plays = _num(_g(off, "ppa")), _num(_g(off, "plays"))
        if np.isnan(ppa) or np.isnan(plays) or plays < 20:
            continue
        team = _g(d, "team")
        info = hmap[gid]
        h = 0 if info["neutral"] else (1 if team == info["home"] else -1)
        rows.append(
            dict(
                game_id=gid,
                off=team,
                deff=_g(d, "opponent"),
                ppa=ppa,
                plays=plays,
                drives=_num(_g(off, "drives")),
                h=h,
            )
        )
    return pd.DataFrame(rows)


def lines_table(raw):
    """Market spread/total per game from CFBD (home perspective; negative = home favored)."""
    rows = []
    for d in raw:
        ls = _g(d, "lines", default=[]) or []
        if not ls:
            continue
        cons = [l for l in ls if str(_g(l, "provider", default="")).lower() == "consensus"]
        use = cons or ls
        def med(key):
            v = [_num(_g(l, key)) for l in use]
            v = [x for x in v if not np.isnan(x)]
            return float(np.median(v)) if v else np.nan
        rows.append(
            dict(
                game_id=int(_g(d, "id")),
                cfbd_spread=med("spread"),
                cfbd_spread_open=med("spreadOpen") if any(_g(l, "spreadOpen", "spread_open") is not None for l in use) else np.nan,
                cfbd_total=med("overUnder"),
                cfbd_total_open=med("overUnderOpen"),
            )
        )
    return pd.DataFrame(rows, columns=["game_id", "cfbd_spread", "cfbd_spread_open", "cfbd_total", "cfbd_total_open"])


def team_info(raw_talent, raw_ret):
    t = {(_g(d, "team", "school")): _num(_g(d, "talent")) for d in raw_talent or []}
    r = {(_g(d, "team")): _num(_g(d, "percentPPA", "percent_ppa")) for d in raw_ret or []}
    return t, r


def load_season(sd, season):
    g = games_table(sd["games"], season)
    a = adv_table(sd["adv"], g) if not g.empty else pd.DataFrame()
    ln = lines_table(sd["lines"])
    t, r = team_info(sd["talent"], sd["returning"])
    return dict(games=g, adv=a, lines=ln, talent=t, ret=r)


def tier(conf, cls, season):
    if cls != "fbs":
        return "FCS"
    if conf in C.POWER_CONFS:
        return "P4"
    if conf in ("Pac-12", "Pac-10") and season <= 2023:
        return "P4"
    return "G5"


def team_tier(team, conf, cls, season):
    if team == "Notre Dame":
        return "P4"
    return tier(conf, cls, season)
