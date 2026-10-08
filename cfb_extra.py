"""Extra data: play-by-play (an optional model input, kept only if it helps) and
QB changes (shown as alerts on the projections page).

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

CFBD_BUDGET = 40  # max new CollegeFootballData calls per run for QB box scores


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


# ---------------- Raw play-by-play: drives, finishing, field position ----------------

PBP_BUDGET = 70  # max week downloads of plays per run (each is one CFBD call)
KICK = ("kickoff",)
NON_DRIVE = ("end period", "end of half", "end of game", "end of regulation", "timeout", "coin toss")


def garbage(period, diff):
    """CFBD-style garbage time: lead over 38 in the 2nd quarter, 28 in the 3rd, 22 in the 4th."""
    d = abs(diff)
    return (period == 2 and d > 38) or (period == 3 and d > 28) or (period >= 4 and d > 22)


def aggregate_plays(raw, finals):
    """One week of CFBD /plays -> one row per (game, offense) with non-garbage points per drive,
    points per scoring opportunity (drives reaching the opponent's 40), and average start."""
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    need = {"gameId", "driveId", "offense", "defense", "offenseScore", "defenseScore", "period", "yardsToGoal", "playType"}
    if not need.issubset(df.columns):
        raise ValueError(f"unexpected play format, missing {sorted(need - set(df.columns))}")
    df = df[df.gameId.isin(list(finals))].copy()
    for c in ("offenseScore", "defenseScore", "period", "yardsToGoal"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["offenseScore", "defenseScore", "period", "yardsToGoal"])
    if df.empty:
        return pd.DataFrame()
    df["pt"] = df.playType.astype(str).str.lower()
    df = df[~df.pt.isin(NON_DRIVE)]
    df["ord"] = pd.to_numeric(df["id"], errors="coerce") if "id" in df else np.arange(len(df))
    df = df.sort_values(["gameId", "ord"])
    out = []
    for gid, g in df.groupby("gameId", sort=False):
        fin = finals[gid]  # {team: final points}
        drives = []
        for did, d in g.groupby("driveId", sort=False):
            scrim = d[~d.pt.str.contains("kickoff")]
            if scrim.empty:
                continue
            first = scrim.iloc[0]
            off = first.offense
            drives.append(dict(off=off, deff=first.defense, period=int(first.period),
                               os=float(first.offenseScore), ds=float(first.defenseScore),
                               ytg=float(first.yardsToGoal), minytg=float(scrim.yardsToGoal.min())))
        if not drives:
            continue
        # Points on each drive = change in the offense's score until the next drive starts.
        for i, dv in enumerate(drives):
            if i + 1 < len(drives):
                nx = drives[i + 1]
                after = nx["os"] if nx["off"] == dv["off"] else nx["ds"]
            else:
                after = fin.get(dv["off"], dv["os"])
            dv["pts"] = max(0.0, min(8.0, after - dv["os"]))
            dv["garbage"] = garbage(dv["period"], dv["os"] - dv["ds"])
        dd = pd.DataFrame(drives)
        dd = dd[~dd.garbage]
        for off, t in dd.groupby("off"):
            opp = t[t.minytg <= 40]
            out.append(dict(game_id=int(gid), off=off, ng_drives=len(t), ng_pts=float(t.pts.sum()),
                            opps=len(opp), opp_pts=float(opp.pts.sum()), start=float(t.ytg.mean())))
    return pd.DataFrame(out)


def pbp_table(all_games, first_season):
    """Download (within budget) and aggregate play-by-play week by week. Aggregates are cached
    as small CSVs; the raw plays (large) are not kept."""
    budget = PBP_BUDGET if not C.OFFLINE else 0
    parts, missing, denied = [], 0, None
    finals = {}
    for r in all_games[all_games.completed].itertuples():
        finals[int(r.game_id)] = {r.home: r.home_pts, r.away: r.away_pts}
    for (season, wk), grp in all_games.groupby(["season", "wk"]):
        if season < first_season:
            continue
        st = "postseason" if wk > 20 else "regular"
        w = wk - 20 if wk > 20 else wk
        path = C.RAW / f"pbp_{season}_{st}_{w}.csv"
        if path.exists():
            try:
                parts.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                pass
            continue
        if not grp.completed.all():
            if grp.completed.any():
                missing += 0  # current week: picked up once the whole week is final
            continue
        if budget <= 0 or denied:
            missing += 1
            continue
        budget -= 1
        resp = requests.get(f"{C.CFBD_BASE}/plays", params={"year": int(season), "week": int(w), "seasonType": st},
                            headers={"Authorization": f"Bearer {C.CFBD_KEY}", "Accept": "application/json"}, timeout=180)
        if resp.status_code in (401, 403):
            denied = f"CFBD refused play-by-play ({resp.status_code}); it may need a Patreon tier"
            missing += 1
            continue
        resp.raise_for_status()
        agg = aggregate_plays(resp.json(), finals)
        agg.to_csv(path, index=False)
        parts.append(agg)
        time.sleep(0.5)
    if denied:
        raise PermissionError(denied)
    tab = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return tab, missing


def pbp_metrics(tab):
    """Per (game, offense): ppd, fin, fpos with light shrinkage for small samples."""
    if tab is None or tab.empty:
        return pd.DataFrame(columns=["game_id", "off", "ppd", "fin", "fpos"])
    t = tab.copy()
    t["ppd"] = (t.ng_pts + 2 * 2.2) / (t.ng_drives + 2)
    t["fin"] = (t.opp_pts + 2 * 4.0) / (t.opps + 2)
    t["fpos"] = t["start"]
    t.loc[t.ng_drives < 4, ["ppd", "fin", "fpos"]] = np.nan
    return t[["game_id", "off", "ppd", "fin", "fpos"]]


# ---------------- Assemble everything ----------------

def build(SEASONS, current, first_eval):
    """Returns (QB-change table for alerts, available optional groups, status, play-by-play metrics)."""
    all_games = pd.concat([SEASONS[s]["games"] for s in sorted(SEASONS) if len(SEASONS[s]["games"])],
                          ignore_index=True)
    status, available = {}, {}

    # QB changes: shown as alerts on the projections page (not a model input; it measured no gain).
    qb_ctx = pd.DataFrame({"game_id": all_games.game_id})
    recent = all_games[all_games.season >= current - 1]
    res, err = _safe(qb_table, recent, Budget(CFBD_BUDGET), current - 1, current)
    if err or res is None or res[0] is None:
        status["qb_alerts"] = f"failed: {err}" if err else "no passing data downloaded yet"
    else:
        qb, missing = res
        d, err = _safe(qb_deltas, recent, qb)
        if err:
            status["qb_alerts"] = f"failed: {err}"
        else:
            qb_ctx = qb_ctx.merge(d, on="game_id", how="left")
            status["qb_alerts"] = f"partial: {missing} weeks still to download" if missing else "ok"

    res, err = _safe(pbp_table, all_games, first_eval - 1)
    if err or res is None:
        status["pbp"] = f"failed: {err}"
        available["_why_pbp"] = status["pbp"]
        pbp = pbp_metrics(None)
    else:
        tab, missing = res
        pbp = pbp_metrics(tab)
        if missing:
            status["pbp"] = f"partial: {missing} weeks of plays still to download"
            available["_why_pbp"] = status["pbp"]
        elif pbp.empty:
            status["pbp"] = "no play-by-play yet"
            available["_why_pbp"] = status["pbp"]
        else:
            status["pbp"] = "ok"
            available["pbp"] = True
    return qb_ctx, available, status, pbp
