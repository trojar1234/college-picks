"""Live predictions for the current week, edge flags, freezing at kickoff, and grading."""
import json

import numpy as np
import pandas as pd

import cfb_config as C
import cfb_fetch as fetch
import cfb_model as model
import cfb_ratings as Rt
import cfb_sim as sim
from cfb_lines import _f

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


def predict_game(r, state, R, mkt, wx, ovr, tiers, gp, shrink=(1.0, 1.0), gctx_row=None):
    f = model.game_feats(R, r.home, r.away, r.neutral)
    conv = np.array(state["conv"])
    groups = state.get("groups", [])
    gx = dict(gctx_row or {})
    bt1 = pd.DataFrame([dict(season=r.season, game_id=r.game_id, **f)])
    pr = model.predict_bt(bt1, conv, groups)
    eh, ea = float(pr.pred_h.iloc[0]), float(pr.pred_a.iloc[0])
    notes, low = [], []
    # QB change alert (information only: the model can't see who starts).
    for side, team in (("h", r.home), ("a", r.away)):
        d = gx.get(f"qbd_{side}")
        if d is not None and d == d and abs(d) > 0.05:
            reg = gx.get(f"qb_reg_{side}")
            notes.append(f"{team}: {gx.get(f'qb_exp_{side}') or 'a backup'} started last game"
                         f"{f' instead of {reg}' if reg else ' instead of the usual starter'}"
                         f" ({abs(d):.1f} yds/att {'better' if d > 0 else 'worse'} career)")
            low.append("QB change")
    for team, side in ((r.home, "h"), (r.away, "a")):
        if team in ovr:
            pts, note = ovr[team]
            if side == "h":
                eh += pts
            else:
                ea += pts
            notes.append(f"{team} {pts:+g} ({note})" if note else f"{team} {pts:+g}")
            low.append("manual adjustment")
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
    # When the model and market differ a lot, the market usually knows something the model
    # doesn't (injury, QB, opt-outs): in the backtest the market was closer 2 times in 3.
    if (spread is not None and abs(model_spread - spread) >= C.DISAGREE_SPREAD) or \
            (total is not None and abs(model_total - total) >= C.DISAGREE_TOTAL):
        low.append("big disagreement: check injury/QB news")
    total_caution = []
    w = wx or {}
    if not w.get("dome") and ((w.get("wind") or 0) >= C.WEATHER_WIND or (w.get("precip") or 0) >= C.WEATHER_RAIN):
        total_caution.append("wind/rain forecast: the model doesn't use weather, the market does")
    out = dict(
        game_id=int(r.game_id), start=r.start.isoformat(), wk=int(r.wk),
        home=r.home, away=r.away, neutral=bool(r.neutral),
        home_tier=tiers.get(r.home, "FCS"), away_tier=tiers.get(r.away, "FCS"),
        exp_home=round(eh, 1), exp_away=round(ea, 1),
        model_spread=model_spread, model_total=model_total,
        market=mkt, weather=wx or {}, notes=notes, low_conf=sorted(set(low)), total_caution=total_caution,
        ratings=dict(
            home_off=Rt._side(R["ppa"], r.home, 0), home_def=Rt._side(R["ppa"], r.home, 1),
            away_off=Rt._side(R["ppa"], r.away, 0), away_def=Rt._side(R["ppa"], r.away, 1),
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


def update_store(store, preds, now, version):
    """Keep the latest pre-kickoff prediction per game (frozen at kickoff) and
    remember when and at what line each edge was first flagged.

    `version` identifies the model and edge thresholds. A flag only counts as a bet
    if it was made by the same version that made the final pre-kickoff prediction,
    so changing the model or the thresholds never leaves stale bets behind."""
    for p in preds:
        key = str(p["game_id"])
        old = store.get(key, {})
        if pd.Timestamp(p["start"]) <= now and old:
            continue  # frozen
        new = {**p, "updated": now.isoformat(), "version": version}
        for kind, line_key in (("spread", "spread"), ("total", "total")):
            fk = f"first_{kind}_flag"
            if old.get(fk) and old[fk].get("version") == version:
                new[fk] = old[fk]  # already bet at an earlier line: keep that line
            elif p.get(f"{kind}_flag") and not p["low_conf"] and not (kind == "total" and p.get("total_caution")):
                new[fk] = dict(ts=now.isoformat(), side=p[f"{kind}_side"], version=version,
                               line=p["market"].get(line_key), model=p[f"model_{kind}"])
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
    """(Re)grade every finished game from its frozen pre-kickoff prediction. A game counts
    toward the ATS / O-U record only if that prediction's own model version flagged it."""
    g = games.set_index("game_id")
    cl = cfbd_lines.set_index("game_id") if len(cfbd_lines) else None
    for key, p in store.items():
        gid = int(key)
        if gid not in g.index or not g.loc[gid, "completed"]:
            continue
        hp, ap = float(g.loc[gid, "home_pts"]), float(g.loc[gid, "away_pts"])
        mv = mview.get(gid, {})
        close_s, close_t = mv.get("spread"), mv.get("total")
        if cl is not None and gid in cl.index:
            close_s = close_s if close_s is not None else _f(cl.loc[gid, "cfbd_spread"])
            close_t = close_t if close_t is not None else _f(cl.loc[gid, "cfbd_total"])
        res = dict(
            home_pts=hp, away_pts=ap,
            err_home=(p["model_total"] - p["model_spread"]) / 2 - hp,  # the projected score shown on the site
            err_away=(p["model_total"] + p["model_spread"]) / 2 - ap,
            err_margin=-p["model_spread"] - (hp - ap),
            err_total=p["model_total"] - (hp + ap),
            close_spread=close_s, close_total=close_t,
        )
        if close_s is not None:
            res["mkt_err_margin"] = -close_s - (hp - ap)
        if close_t is not None:
            res["mkt_err_total"] = close_t - (hp + ap)
        # Every game, regardless of flags: the model's side against the closing line.
        # A model number exactly on the line is "no pick" and isn't counted.
        if close_s is not None and close_s - p["model_spread"] != 0:
            res["ats_all"] = _ats(hp - ap, "home" if close_s - p["model_spread"] > 0 else "away", close_s)
        if close_t is not None and p["model_total"] - close_t != 0:
            res["ou_all"] = _ou(hp + ap, "over" if p["model_total"] > close_t else "under", close_t)
        same = lambda f: f and f.get("line") is not None and f.get("version") is not None \
            and f.get("version") == p.get("version")
        fs = p.get("first_spread_flag")
        if same(fs):
            res["ats"] = _ats(hp - ap, fs["side"], fs["line"])
            if close_s is not None:
                res["clv_spread"] = (fs["line"] - close_s) if fs["side"] == "home" else (close_s - fs["line"])
        ft = p.get("first_total_flag")
        if same(ft):
            res["ou"] = _ou(hp + ap, ft["side"], ft["line"])
            if close_t is not None:
                res["clv_total"] = (close_t - ft["line"]) if ft["side"] == "over" else (ft["line"] - close_t)
        p["result"] = res
    return store
