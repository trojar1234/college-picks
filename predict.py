"""Live predictions for the current week, edge flags, freezing at kickoff, and grading."""
import json

import numpy as np
import pandas as pd

from . import config as C
from . import fetch, model, sim
from .lines import _f

STORE = C.DATA / "predictions.json"
OVERRIDES = C.DATA / "overrides.csv"


def load_store():
    return json.loads(STORE.read_text()) if STORE.exists() else {}


def save_store(s):
    STORE.write_text(json.dumps(s, indent=1, default=str))


def load_overrides():
    """data/overrides.csv: team,points,note — e.g. 'Colorado,-4,QB out'. Optional."""
    if not OVERRIDES.exists():
        return {}
    try:
        df = pd.read_csv(OVERRIDES, comment="#")
    except Exception:
        return {}
    out = {}
    for r in df.itertuples():
        try:
            out[str(r.team).strip()] = (float(r.points), str(getattr(r, "note", "") or ""))
        except (TypeError, ValueError):
            pass
    return out


def current_slate(games, now):
    up = games[~games.completed & (games.start > now - pd.Timedelta(hours=6))]
    if up.empty:
        return games.iloc[0:0], None
    wk = int(up.sort_values("start").iloc[0].wk)
    return games[games.wk == wk].copy(), wk


def venue_map():
    m = {}
    for v in fetch.venues() or []:
        vid = v.get("id")
        lat = v.get("latitude")
        lon = v.get("longitude")
        loc = v.get("location") or {}
        if lat is None and isinstance(loc, dict):
            lat, lon = loc.get("x"), loc.get("y")
        if vid is not None and lat is not None and lon is not None:
            m[int(vid)] = (float(lat), float(lon), bool(v.get("dome")))
    return m


def game_weather(slate, now):
    vm = venue_map()
    todo = [r for r in slate.itertuples()
            if not r.completed and r.venue_id == r.venue_id and r.venue_id is not None
            and int(r.venue_id) in vm and not vm[int(r.venue_id)][2]
            and r.start - now < pd.Timedelta(days=7)]
    res = fetch.weather([vm[int(r.venue_id)][:2] for r in todo])
    out = {}
    for r, w in zip(todo, res):
        if not w or "hourly" not in w:
            continue
        times = pd.to_datetime(w["hourly"]["time"], utc=True)
        i = int(np.argmin(np.abs((times - r.start).total_seconds())))
        sl = slice(i, i + 3)  # kickoff through roughly halftime
        out[r.game_id] = dict(
            wind=float(np.mean(w["hourly"]["wind_speed_10m"][sl])),
            precip=float(np.sum(w["hourly"]["precipitation"][sl])),
            temp=float(w["hourly"]["temperature_2m"][i]),
        )
    for r in slate.itertuples():
        if r.venue_id == r.venue_id and r.venue_id is not None and int(r.venue_id) in vm and vm[int(r.venue_id)][2]:
            out[r.game_id] = dict(dome=True)
    return out


def predict_game(r, state, Rp, Rc, mkt, wx, ovr, tiers, gp, shrink=(1.0, 1.0)):
    f = model.game_feats(Rp, Rc, r.home, r.away, r.neutral)
    conv = np.array(state["conv"])
    eh = float(model.points(conv, [f["p_h"]], [f["q_h"]], [f["h"]])[0])
    ea = float(model.points(conv, [f["p_a"]], [f["q_a"]], [-f["h"]])[0])
    notes, low = [], []
    for team, side in ((r.home, "h"), (r.away, "a")):
        if team in ovr:
            pts, note = ovr[team]
            if side == "h":
                eh += pts
            else:
                ea += pts
            notes.append(f"{team} {pts:+g} ({note})" if note else f"{team} {pts:+g}")
            low.append("manual adjustment")
    if wx and wx.get("wind", 0) > C.WIND_START:
        k = max(0.75, 1 - C.WIND_PER_MPH * (wx["wind"] - C.WIND_START))
        eh, ea = eh * k, ea * k
        notes.append(f"wind {wx['wind']:.0f} mph")
    drives = (f["p_h"] + f["p_a"]) / 2 / state["plays_per_drive"]
    spread = mkt.get("spread")
    total = mkt.get("total")
    h, a = sim.simulate(eh, ea, drives, state["sim_sd"], state["rho"], n=C.SIMS)
    s = sim.summarize(h, a, spread, total)
    if "FCS" in (tiers.get(r.home), tiers.get(r.away)):
        low.append("FCS opponent")
    if gp < 3:
        low.append("early season")
    model_spread = round(-s["margin"], 1)
    model_total = round(s["total_mean"], 1)
    out = dict(
        game_id=int(r.game_id), start=r.start.isoformat(), wk=int(r.wk),
        home=r.home, away=r.away, neutral=bool(r.neutral),
        home_tier=tiers.get(r.home, "FCS"), away_tier=tiers.get(r.away, "FCS"),
        exp_home=round(eh, 1), exp_away=round(ea, 1),
        model_spread=model_spread, model_total=model_total,
        market=mkt, weather=wx or {}, notes=notes, low_conf=sorted(set(low)),
        ratings=dict(
            home_off=Rp["off"].get(r.home), home_def=Rp["deff"].get(r.home),
            away_off=Rp["off"].get(r.away), away_def=Rp["deff"].get(r.away),
            home_pace=f["p_h"], away_pace=f["p_a"],
        ),
        **s,
    )
    if spread is not None:
        e = spread - model_spread  # >0: home side has value
        out["spread_edge"] = round(abs(e), 1)
        out["spread_side"] = "home" if e > 0 else "away"
        out["spread_flag"] = abs(e) >= C.SPREAD_EDGE
        pc = s.get("p_home_cover", 0.5)
        raw = pc if e > 0 else 1 - pc - s.get("p_push_spread", 0)
        out["p_side_cover"] = round(0.5 + shrink[0] * (raw - 0.5), 3)  # calibrated against backtest
    if total is not None:
        e = model_total - total
        out["total_edge"] = round(abs(e), 1)
        out["total_side"] = "over" if e > 0 else "under"
        out["total_flag"] = abs(e) >= C.TOTAL_EDGE
        po = s.get("p_over", 0.5)
        raw = po if e > 0 else 1 - po - s.get("p_push_total", 0)
        out["p_side_total"] = round(0.5 + shrink[1] * (raw - 0.5), 3)
    return out


def update_store(store, preds, now):
    """Keep the latest pre-kickoff prediction per game (frozen at kickoff) and
    remember when and at what line each edge was first flagged."""
    for p in preds:
        key = str(p["game_id"])
        old = store.get(key, {})
        if pd.Timestamp(p["start"]) <= now and old:
            continue  # frozen
        new = {**p, "updated": now.isoformat()}
        for kind, line_key in (("spread", "spread"), ("total", "total")):
            fk = f"first_{kind}_flag"
            if old.get(fk):
                new[fk] = old[fk]
            elif p.get(f"{kind}_flag") and not p["low_conf"]:
                new[fk] = dict(ts=now.isoformat(), side=p[f"{kind}_side"],
                               line=p["market"].get(line_key), model=p[f"model_{kind}"])
        if old.get("result"):
            new["result"] = old["result"]
        store[key] = new
    return store


def _ats(margin, side, line):
    if line is None:
        return None
    v = margin + line if side == "home" else -(margin + line)
    return "push" if v == 0 else ("win" if v > 0 else "loss")


def _ou(total, side, line):
    if line is None:
        return None
    v = (total - line) if side == "over" else (line - total)
    return "push" if v == 0 else ("win" if v > 0 else "loss")


def grade(store, games, mview, cfbd_lines):
    g = games.set_index("game_id")
    cl = cfbd_lines.set_index("game_id") if len(cfbd_lines) else None
    for key, p in store.items():
        gid = int(key)
        if p.get("result") or gid not in g.index or not g.loc[gid, "completed"]:
            continue
        hp, ap = float(g.loc[gid, "home_pts"]), float(g.loc[gid, "away_pts"])
        mv = mview.get(gid, {})
        close_s, close_t = mv.get("spread"), mv.get("total")
        if cl is not None and gid in cl.index:
            close_s = close_s if close_s is not None else _f(cl.loc[gid, "cfbd_spread"])
            close_t = close_t if close_t is not None else _f(cl.loc[gid, "cfbd_total"])
        res = dict(
            home_pts=hp, away_pts=ap,
            err_home=p["home_med"] - hp, err_away=p["away_med"] - ap,
            err_margin=-p["model_spread"] - (hp - ap),
            err_total=p["model_total"] - (hp + ap),
            close_spread=close_s, close_total=close_t,
        )
        if close_s is not None:
            res["mkt_err_margin"] = -close_s - (hp - ap)
        if close_t is not None:
            res["mkt_err_total"] = close_t - (hp + ap)
        fs = p.get("first_spread_flag")
        if fs and fs.get("line") is not None:
            res["ats"] = _ats(hp - ap, fs["side"], fs["line"])
            if close_s is not None:
                res["clv_spread"] = (fs["line"] - close_s) if fs["side"] == "home" else (close_s - fs["line"])
        ft = p.get("first_total_flag")
        if ft and ft.get("line") is not None:
            res["ou"] = _ou(hp + ap, ft["side"], ft["line"])
            if close_t is not None:
                res["clv_total"] = (close_t - ft["line"]) if ft["side"] == "over" else (ft["line"] - close_t)
        p["result"] = res
    return store
