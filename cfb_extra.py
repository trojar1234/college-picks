"""Extra data for the optional feature groups: travel and rest, QB changes,
historical weather, and richer preseason priors.

Everything here is optional. If a download fails or a data source changes
shape, that feature is marked unavailable and the model runs without it.
Downloads are capped per run so the free API tiers are never exceeded;
anything not fetched yet is picked up on the next run.
"""
import json
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

import cfb_config as C
import cfb_fetch as F

CFBD_BUDGET = 260      # max new CollegeFootballData calls per run for these extras
WEATHER_BUDGET = 2500  # max game-locations of archive weather per run (free tier limits)
WX_FILE = C.RAW / "weather_hist.json"


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k), None
    except Exception as e:  # never let an optional feature break the run
        return None, f"{type(e).__name__}: {str(e)[:160]}"


class Budget:
    def __init__(self, n):
        self.left = n

    def get(self, endpoint, params, name):
        """Cached CFBD call that counts against this run's budget when it hits the network."""
        if F._load(name) is not None or C.OFFLINE:
            return F.cfbd(endpoint, params, name)
        if self.left <= 0:
            raise BudgetExhausted(name)
        self.left -= 1
        return F.cfbd(endpoint, params, name)


class BudgetExhausted(Exception):
    pass


# ---------------- Travel and rest (no downloads needed) ----------------

def _venues():
    out = {}
    for v in F.venues() or []:
        lat, lon = v.get("latitude"), v.get("longitude")
        loc = v.get("location") or {}
        if lat is None and isinstance(loc, dict):
            lat, lon = loc.get("x"), loc.get("y")
        if v.get("id") is not None and lat is not None and lon is not None:
            out[int(v["id"])] = dict(lat=float(lat), lon=float(lon), dome=bool(v.get("dome")))
    return out


def _km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def travel_rest(all_games):
    """Per game: rest days, travel (1000 km), and time zones crossed for each side."""
    vm = _venues()
    g = all_games.copy()
    g["vloc"] = g.venue_id.map(lambda v: (vm[int(v)]["lat"], vm[int(v)]["lon"]) if v == v and v is not None and int(v) in vm else None)
    homes = {}
    for (season, team), grp in g[~g.neutral & g.vloc.notna()].groupby(["season", "home"]):
        homes[(season, team)] = grp.vloc.value_counts().index[0]
    last = {}
    rows = []
    for r in g.sort_values("start").itertuples():
        out = dict(game_id=r.game_id)
        for side, team in (("h", r.home), ("a", r.away)):
            prev = last.get((r.season, team))
            rest = 14.0 if prev is None else min(14.0, (r.start - prev).total_seconds() / 86400)
            out[f"rest_{side}"] = rest
            home_loc = homes.get((r.season, team))
            if r.vloc is None or home_loc is None or (side == "h" and not r.neutral):
                out[f"trav_{side}"], out[f"tz_{side}"] = 0.0, 0.0
            else:
                out[f"trav_{side}"] = _km(home_loc, r.vloc) / 1000
                out[f"tz_{side}"] = abs(home_loc[1] - r.vloc[1]) / 15
            last[(r.season, team)] = r.start
        rows.append(out)
    df = pd.DataFrame(rows)
    df["rest_adv_h"] = (df.rest_h - df.rest_a).clip(-7, 7)
    df["rest_adv_a"] = -df.rest_adv_h
    return df[["game_id", "rest_adv_h", "rest_adv_a", "trav_h", "trav_a", "tz_h", "tz_a"]]


# ---------------- QB changes ----------------

def _parse_passing(raw):
    """CFBD /games/players (category=passing) -> rows of (game_id, team, qb, att, adj_yds)."""
    rows = []
    for gm in raw or []:
        gid = gm.get("id")
        for tm in gm.get("teams", []) or []:
            team = tm.get("team") or tm.get("school")
            for cat in tm.get("categories", []) or []:
                if str(cat.get("name", "")).lower() != "passing":
                    continue
                stats = defaultdict(dict)
                for ty in cat.get("types", []) or []:
                    for a in ty.get("athletes", []) or []:
                        stats[(a.get("id"), a.get("name"))][str(ty.get("name", "")).upper()] = a.get("stat")
                for (pid, name), st in stats.items():
                    ca = str(st.get("C/ATT", "0/0")).split("/")
                    try:
                        att = int(ca[1])
                        yds = float(st.get("YDS", 0) or 0)
                        td = float(st.get("TD", 0) or 0)
                        it = float(st.get("INT", 0) or 0)
                    except (ValueError, IndexError):
                        continue
                    if att <= 0 or name in (None, "TEAM", "Team"):
                        continue
                    rows.append(dict(game_id=int(gid), team=team, qb=str(pid or name), qb_name=name,
                                     att=att, adj=yds + 20 * td - 45 * it))
    return rows


QB_PRIOR_ATT, QB_PRIOR_YPA = 150.0, 5.0  # unknown QBs are assumed to be below-average backups


def qb_table(all_games, budget, first_season, current):
    """Download passing box scores week by week (cached), then compute for every
    game whether each team's expected starter differs from its regular starter."""
    rows, missing = [], 0
    done = all_games[all_games.completed]
    for (season, wk), grp in done.groupby(["season", "wk"]):
        if season < first_season:
            continue
        st = "postseason" if wk > 20 else "regular"
        w = wk - 20 if wk > 20 else wk
        name = f"passing_{season}_{st}_{w}"
        week_games = all_games[(all_games.season == season) & (all_games.wk == wk)]
        if not week_games.completed.all() and F._load(name) is None:
            continue  # wait until the whole week is final
        try:
            raw = budget.get("/games/players", {"year": int(season), "week": int(w), "seasonType": st,
                                                "category": "passing"}, name)
        except BudgetExhausted:
            missing += 1
            continue
        rows += _parse_passing(raw)
    qb = pd.DataFrame(rows)
    if qb.empty:
        return None, missing
    return qb, missing


def qb_deltas(all_games, qb):
    """For each game and side: quality of the expected starter (last game's starter)
    minus quality of the team's regular starter, in adjusted yards per attempt."""
    st = all_games.set_index("game_id").start
    qb = qb[qb.game_id.isin(st.index)].copy()
    qb["start"] = qb.game_id.map(st)
    qb["season"] = qb.game_id.map(all_games.set_index("game_id").season)
    starters = qb.sort_values("att", ascending=False).drop_duplicates(["game_id", "team"])
    starters = starters[starters.att >= 5]
    career = defaultdict(lambda: [0.0, 0.0])  # qb -> [att, adj]
    season_att = defaultdict(lambda: defaultdict(float))  # (season, team) -> qb -> att
    last_starter = {}
    qb_by_game = {k: g for k, g in qb.groupby("game_id")}
    start_by_game = starters.set_index(["game_id", "team"]).qb.to_dict()
    names = dict(zip(qb.qb, qb.qb_name))

    def quality(q):
        a, adj = career[q]
        return (adj + QB_PRIOR_YPA * QB_PRIOR_ATT) / (a + QB_PRIOR_ATT)

    out = []
    for r in all_games.sort_values("start").itertuples():
        row = dict(game_id=r.game_id)
        for side, team in (("h", r.home), ("a", r.away)):
            exp = last_starter.get((r.season, team))
            sa = season_att[(r.season, team)]
            reg = max(sa, key=sa.get) if sa else None
            row[f"qbd_{side}"] = 0.0 if (exp is None or reg is None or exp == reg) else quality(exp) - quality(reg)
            row[f"qb_exp_{side}"] = names.get(exp, exp)
        out.append(row)
        # After the game: update careers and who started (only for completed games).
        if r.completed and r.game_id in qb_by_game:
            for x in qb_by_game[r.game_id].itertuples():
                career[x.qb][0] += x.att
                career[x.qb][1] += x.adj
                season_att[(r.season, x.team)][x.qb] += x.att
            for team in (r.home, r.away):
                s = start_by_game.get((r.game_id, team))
                if s is not None:
                    last_starter[(r.season, team)] = s
    return pd.DataFrame(out)


# ---------------- Historical weather (Open-Meteo archive) ----------------

def weather_history(all_games, first_season):
    cache = json.loads(WX_FILE.read_text()) if WX_FILE.exists() else {}
    vm = _venues()
    todo = all_games[all_games.completed & (all_games.season >= first_season)]
    todo = todo[~todo.game_id.astype(str).isin(cache.keys())]
    left = WEATHER_BUDGET
    if not C.OFFLINE:
        for date, grp in todo.groupby(todo.start.dt.strftime("%Y-%m-%d")):
            pts = []
            for r in grp.itertuples():
                v = vm.get(int(r.venue_id)) if r.venue_id == r.venue_id and r.venue_id is not None else None
                if v is None:
                    cache[str(r.game_id)] = None
                elif v["dome"]:
                    cache[str(r.game_id)] = "dome"
                else:
                    pts.append((r, v))
            for i in range(0, len(pts), 50):
                if left <= 0:
                    break
                chunk = pts[i:i + 50]
                left -= len(chunk)
                end = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                try:
                    resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=dict(
                        latitude=",".join(f"{v['lat']:.4f}" for _, v in chunk),
                        longitude=",".join(f"{v['lon']:.4f}" for _, v in chunk),
                        start_date=date, end_date=end,
                        hourly="wind_speed_10m,precipitation,temperature_2m",
                        wind_speed_unit="mph", temperature_unit="fahrenheit", timezone="UTC"), timeout=90)
                    resp.raise_for_status()
                    js = resp.json()
                    time.sleep(6)  # stay under the archive's per-minute limit
                    js = js if isinstance(js, list) else [js]
                except Exception as e:
                    print("Weather archive fetch failed:", e)
                    left = 0
                    break
                for (r, _), w in zip(chunk, js):
                    try:
                        t = pd.to_datetime(w["hourly"]["time"], utc=True)
                        k = int(np.argmin(np.abs((t - r.start).total_seconds())))
                        sl = slice(k, k + 3)
                        cache[str(r.game_id)] = [float(np.nanmean(w["hourly"]["wind_speed_10m"][sl])),
                                                 float(np.nansum(w["hourly"]["precipitation"][sl])),
                                                 float(w["hourly"]["temperature_2m"][k])]
                    except Exception:
                        cache[str(r.game_id)] = None
            if left <= 0:
                break
        WX_FILE.write_text(json.dumps(cache))
    remaining = int((~todo.game_id.astype(str).isin(cache.keys())).sum())
    return cache, remaining


def weather_cols(wind, precip, temp, dome=False):
    if dome:
        return dict(wind_over=0.0, precip=0.0, cold=0.0)
    return dict(wind_over=max(0.0, (wind or 0) - 10), precip=float(precip or 0), cold=max(0.0, 40 - (temp if temp is not None else 60)))


# ---------------- Richer preseason priors ----------------

QB_POS = {"QB", "PRO", "DUAL"}
OL_POS = {"OT", "OG", "OC", "IOL", "OL", "C", "G", "T"}
STAR_VAL = {2: 0.5, 3: 1.0, 4: 2.0, 5: 3.0}


def prior_extras(seasons, budget, current):
    """{season: {team: {qbrec, olrec, portal, hc, defret}}} plus a status message per piece."""
    status = {}
    out = defaultdict(lambda: defaultdict(dict))
    first, last = min(seasons), max(seasons)

    # Recruiting: best QB recruits and top OL recruits signed in the last four classes.
    try:
        rec = []
        for y in range(first - 3, last + 1):
            rec += [dict(year=y, team=r.get("committedTo"), pos=str(r.get("position") or "").upper(),
                         rating=r.get("rating"))
                    for r in budget.get("/recruiting/players", {"year": y, "classification": "HighSchool"}, f"recruits_{y}") or []]
        rec = pd.DataFrame(rec, columns=["year", "team", "pos", "rating"]).dropna(subset=["team", "rating"])
        for s in seasons:
            win = rec[(rec.year >= s - 3) & (rec.year <= s)]
            for team, grp in win.groupby("team"):
                qbs = grp[grp.pos.isin(QB_POS)].rating.sort_values(ascending=False)
                ols = grp[grp.pos.isin(OL_POS)].rating.sort_values(ascending=False)
                out[s][team]["qbrec"] = float(qbs.head(2).mean()) if len(qbs) else 0.80
                out[s][team]["olrec"] = float(ols.head(5).mean()) if len(ols) else 0.80
        status["recruiting"] = "ok" if any("qbrec" in t for s in out.values() for t in s.values()) else "no data returned"
    except BudgetExhausted:
        status["recruiting"] = "partial: will finish downloading next run"
    except Exception as e:
        status["recruiting"] = f"failed: {type(e).__name__}: {str(e)[:120]}"

    # Transfer portal: value of arrivals minus departures.
    try:
        for y in range(max(first, 2018), last + 1):
            net = defaultdict(float)
            for t in budget.get("/player/portal", {"year": y}, f"portal_{y}") or []:
                v = STAR_VAL.get(int(t.get("stars") or 0), 0.3)
                if t.get("destination"):
                    net[t["destination"]] += v
                if t.get("origin"):
                    net[t["origin"]] -= v
            for team, v in net.items():
                out[y][team]["portal"] = v
        status["portal"] = "ok" if any("portal" in t for s in out.values() for t in s.values()) else "no data returned"
    except BudgetExhausted:
        status["portal"] = "partial: will finish downloading next run"
    except Exception as e:
        status["portal"] = f"failed: {type(e).__name__}: {str(e)[:120]}"

    # Coaching: did the head coach change from last season?
    try:
        raw = budget.get("/coaches", {"minYear": first - 1, "maxYear": last}, f"coaches_{first - 1}_{last}")
        hc = {}
        for c in raw or []:
            name = f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
            for s in c.get("seasons", []) or []:
                key = (s.get("school"), int(s.get("year")))
                games = (s.get("games") or 0)
                if key not in hc or games > hc[key][1]:
                    hc[key] = (name, games)
        for (school, y), (name, _) in hc.items():
            if (school, y - 1) in hc:
                out[y][school]["hc"] = 1.0 if hc[(school, y - 1)][0] != name else 0.0
        status["coaches"] = "ok" if any("hc" in t for s in out.values() for t in s.values()) else "no data returned"
    except BudgetExhausted:
        status["coaches"] = "partial: will finish downloading next run"
    except Exception as e:
        status["coaches"] = f"failed: {type(e).__name__}: {str(e)[:120]}"

    # Defensive returning production: share of last season's defensive production
    # (tackles, TFL, sacks, passes defended) by players on this season's roster.
    try:
        for y in range(first, last + 1):
            stats = budget.get("/stats/player/season", {"year": y - 1, "category": "defensive"}, f"defstats_{y - 1}")
            roster = budget.get("/roster", {"year": y}, f"roster_{y}")
            on = {(str(p.get("id")), p.get("team")) for p in roster or []}
            prod = defaultdict(float)
            team_of = {}
            wts = {"TOT": 1.0, "TFL": 2.0, "SACKS": 2.0, "PD": 2.0}
            for st in stats or []:
                w = wts.get(str(st.get("statType", "")).upper())
                if w is None:
                    continue
                try:
                    v = float(st.get("stat") or 0)
                except ValueError:
                    continue
                pid = str(st.get("playerId"))
                prod[(pid, st.get("team"))] += w * v
                team_of[(pid, st.get("team"))] = st.get("team")
            tot, kept = defaultdict(float), defaultdict(float)
            for key, v in prod.items():
                tot[key[1]] += v
                if key in on:
                    kept[key[1]] += v
            if on:
                for team, v in tot.items():
                    if v > 0:
                        out[y][team]["defret"] = kept[team] / v
        status["defense_returning"] = "ok" if any("defret" in t for s in out.values() for t in s.values()) else "no data returned"
    except BudgetExhausted:
        status["defense_returning"] = "partial: will finish downloading next run"
    except Exception as e:
        status["defense_returning"] = f"failed: {type(e).__name__}: {str(e)[:120]}"

    ok = sum(v == "ok" for v in status.values())
    return {s: dict(t) for s, t in out.items()}, status, ok


# ---------------- Assemble everything ----------------

def build(SEASONS, current, first_eval):
    """Returns (game context table, prior extras, available groups, status)."""
    all_games = pd.concat([SEASONS[s]["games"] for s in sorted(SEASONS) if len(SEASONS[s]["games"])],
                          ignore_index=True)
    budget = Budget(CFBD_BUDGET)
    status, available = {}, {}

    tr, err = _safe(travel_rest, all_games)
    gctx = tr if tr is not None else pd.DataFrame({"game_id": all_games.game_id})
    if err:
        status["travel"] = f"failed: {err}"
        available["_why_travel"] = status["travel"]
    else:
        status["travel"] = "ok"
        available["travel"] = True

    res, err = _safe(qb_table, all_games, budget, first_eval - 1, current)
    if err or res is None or res[0] is None:
        status["qb"] = f"failed: {err}" if err else "no passing data downloaded yet"
        available["_why_qb"] = status["qb"]
    else:
        qb, missing = res
        d, err = _safe(qb_deltas, all_games, qb)
        if err:
            status["qb"] = f"failed: {err}"
            available["_why_qb"] = status["qb"]
        else:
            gctx = gctx.merge(d, on="game_id", how="left")
            if missing:
                status["qb"] = f"partial: {missing} weeks still to download"
                available["_why_qb"] = status["qb"]
            else:
                status["qb"] = "ok"
                available["qb"] = True

    res, err = _safe(weather_history, all_games, first_eval - 1)
    if err:
        status["weather"] = f"failed: {err}"
        available["_why_weather"] = status["weather"]
    else:
        cache, remaining = res
        wx = []
        for gid in all_games.game_id:
            v = cache.get(str(gid))
            if v == "dome":
                wx.append(dict(game_id=gid, **weather_cols(0, 0, 60, dome=True)))
            elif isinstance(v, list):
                wx.append(dict(game_id=gid, **weather_cols(*v)))
        if wx:
            gctx = gctx.merge(pd.DataFrame(wx), on="game_id", how="left")
        if remaining:
            status["weather"] = f"partial: {remaining} past games still to download"
            available["_why_weather"] = status["weather"]
        elif not wx:
            status["weather"] = "no weather history yet"
            available["_why_weather"] = status["weather"]
        else:
            status["weather"] = "ok"
            available["weather"] = True

    res, err = _safe(prior_extras, sorted(SEASONS), budget, current)
    if err:
        status["priors"] = f"failed: {err}"
        pe = {}
        available["_why_priors"] = status["priors"]
    else:
        pe, pst, ok = res
        status.update({f"priors/{k}": v for k, v in pst.items()})
        if any(v.startswith("partial") for v in pst.values()):
            available["_why_priors"] = "still downloading"
        elif ok:
            available["priors"] = True
        else:
            available["_why_priors"] = "no prior data available"
    status["cfbd_calls_used_for_extras"] = CFBD_BUDGET - budget.left
    return gctx, pe, available, status
