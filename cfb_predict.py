"""Live predictions for the current week, edge flags, freezing at kickoff, and grading."""
import json
import math

import numpy as np
import pandas as pd

import cfb_config as C
import cfb_lines as lines
import cfb_fetch as fetch
import cfb_model as model
import cfb_ratings as Rt
import cfb_sim as sim
from cfb_lines import _f

STORE = C.DATA / "predictions.json"   # everything recorded before Oct 2026 (read only)
STORE_DIR = C.DATA / "predictions"    # one file per season and week: past weeks stop changing
OVERRIDES = C.DATA / "overrides.csv"
if not OVERRIDES.exists() and (C.DATA.parent / "overrides.csv").exists():
    OVERRIDES = C.DATA.parent / "overrides.csv"  # also accept the file at the repo root


def load_store():
    """Every recorded prediction, keyed by game id: the older single file, then the per-week files
    (newer, so they win when both have a game)."""
    s = json.loads(STORE.read_text()) if STORE.exists() else {}
    if STORE_DIR.exists():
        for f in sorted(STORE_DIR.glob("*/week-*.json")):
            s.update(json.loads(f.read_text()))
    return s


def _week_file(p):
    t = pd.Timestamp(p.get("start")) if p.get("start") else None
    season = (t.year if t.month >= 7 else t.year - 1) if t is not None else 0
    return STORE_DIR / str(season) / f"week-{int(p.get('wk') or 0):02d}.json"


def save_store(s):
    """Write each season-week's predictions to its own file, only when something in it changed."""
    groups = {}
    for k, p in s.items():
        groups.setdefault(_week_file(p), {})[k] = p
    for f, g in groups.items():
        txt = json.dumps(g, indent=1, sort_keys=True, default=str)
        if f.exists() and f.read_text() == txt:
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(txt)


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
    soon = up.start <= now + pd.Timedelta(days=6)
    keep = (games.wk == wk) | games.game_id.isin(up[soon].game_id)
    return games[keep].copy(), wk


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


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def predict_game(r, state, R, mkt, wx, ovr, tiers, gp, shrink=(1.0, 1.0), gctx_row=None):
    """One game's projection. The spread and total are the expected points (the numbers the backtest
    validated); the score simulation supplies the shape of the outcome distribution."""
    f = model.game_feats(R, r.home, r.away, r.neutral)
    conv = np.array(state["conv"])
    groups = state.get("groups", [])
    gx = dict(gctx_row or {})
    bt1 = pd.DataFrame([dict(season=r.season, game_id=r.game_id, home_tier=tiers.get(r.home, "FCS"),
                             away_tier=tiers.get(r.away, "FCS"), **f)])
    pr = model.predict_bt(bt1, conv, groups, state.get("total_model"))
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
    # Service-academy correction (see cfb_config.ACADEMIES): learned from out-of-sample errors.
    auto = []
    ac = state.get("academy") or {}
    ah, aa = r.home in (ac.get("teams") or []), r.away in (ac.get("teams") or [])
    if (ah or aa) and ac.get("n"):
        madj = ac["margin"] * (int(ah) - int(aa))
        tadj = ac["total"]
        eh += (madj + tadj) / 2
        ea += (tadj - madj) / 2
        who = " and ".join(t for t, x in ((r.home, ah), (r.away, aa)) if x)
        auto.append(f"Service-academy correction for {who}: margin {madj:+.1f} (home view), total {tadj:+.1f}"
                    if madj else f"Service-academy correction: total {tadj:+.1f}")
    drives = (f["p_h"] + f["p_a"]) / 2 / state["plays_per_drive"]
    spread = mkt.get("spread")
    total = mkt.get("total")
    # Fixed seed per game: identical inputs always give identical numbers (no run-to-run flicker)
    rng = np.random.default_rng([int(r.game_id), 7])
    h, a = sim.simulate(eh, ea, drives, state["sim_sd"], state["rho"], n=C.SIMS, rng=rng)
    # The simulation's averages drift slightly from the expected points (overtime adds about half a
    # point to totals on average). Shift its outcomes so it agrees with the expected points, which is
    # what the backtest scored.
    sm = (eh - ea) - float((h - a).mean())
    st_ = (eh + ea) - float((h + a).mean())
    s = sim.summarize(h, a, None if spread is None else spread + sm, None if total is None else total - st_)
    s.update(margin=eh - ea, total_mean=eh + ea, home_mean=eh, away_mean=ea,
             p_home_win=float(((h - a) + sm > 0).mean()))
    if spread is not None:
        s["p_push_spread"] = float(((h - a) + spread == 0).mean())
    if total is not None:
        s["p_push_total"] = float(((h + a) == total).mean())
    # Win, cover and over chances and the 80% range come from a normal curve around the expected points
    # with the out-of-sample error size. In the backtest this was better calibrated than the simulation
    # (Brier 0.1674 vs 0.1678; its 80% range held 80% of results vs the simulation's 75%), and it's the
    # same method the Matchup page and the backtest use. The simulation still gives likely scores and pushes.
    msd, tsd = state.get("margin_sd") or 16.0, state.get("total_sd") or 16.5
    nm = _ncdf((eh - ea) / msd)
    s["p_home_win"] = nm
    s["range80"] = [round((eh - ea) - 1.2816 * msd, 1), round((eh - ea) + 1.2816 * msd, 1)]
    if spread is not None:
        s["p_home_cover"] = _ncdf((eh - ea + spread) / msd) - s["p_push_spread"] / 2
    if total is not None:
        s["p_over"] = _ncdf((eh + ea - total) / tsd) - s["p_push_total"] / 2
    # Cautions with a switch per market (see cfb_config). FCS games always block totals.
    tot_only, spr_only = [], []
    if "FCS" in (tiers.get(r.home), tiers.get(r.away)):
        (low if getattr(C, "FCS_BLOCKS_SPREADS", True) else tot_only).append("FCS opponent")
    if gp < 3:
        es, et = getattr(C, "EARLY_SEASON_BLOCKS_SPREADS", True), getattr(C, "EARLY_SEASON_BLOCKS_TOTALS", True)
        if es and et:
            low.append("early season")
        elif es:
            spr_only.append("early season")
        elif et:
            tot_only.append("early season")
    model_spread = round(-(eh - ea), 1)
    model_total = round(eh + ea, 1)
    # When the model and market differ a lot, the market usually knows something the model
    # doesn't (injury, QB, opt-outs): in the backtest the market was closer 2 times in 3.
    # Each caution applies only to its own bet: a big gap on the total doesn't block the spread.
    spread_caution, total_caution, soft = list(spr_only), list(tot_only), {}
    if spread is not None and abs(model_spread - spread) >= C.DISAGREE_SPREAD:
        spread_caution.append("big disagreement: check injury/QB news")
    elif spread is not None and abs(model_spread - spread) >= C.NEWS_NOTE:
        soft["spread"] = "8+ pts apart: worth a quick injury/QB news check"
    if total is not None and abs(model_total - total) >= C.DISAGREE_TOTAL:
        total_caution.append("big disagreement: check injury/QB news")
    elif total is not None and abs(model_total - total) >= C.NEWS_NOTE:
        soft["total"] = "8+ pts apart: worth a quick injury/QB news check"
    nb = len(mkt.get("books") or [])
    if 0 < nb < 3:
        for k in ("spread", "total"):
            if mkt.get(k) is not None and k not in soft:
                soft[k] = f"thin market: only {nb} book{'s' if nb > 1 else ''} quoting"
    w = wx or {}
    if not w.get("dome") and ((w.get("wind") or 0) >= C.WEATHER_WIND or (w.get("precip") or 0) >= C.WEATHER_RAIN):
        total_caution.append("wind/rain forecast: the model doesn't use weather, the market does")
    out = dict(
        game_id=int(r.game_id), start=r.start.isoformat(), wk=int(r.wk),
        home=r.home, away=r.away, neutral=bool(r.neutral),
        home_tier=tiers.get(r.home, "FCS"), away_tier=tiers.get(r.away, "FCS"),
        exp_home=round(eh, 1), exp_away=round(ea, 1),
        model_spread=model_spread, model_total=model_total,
        market=mkt, weather=wx or {}, notes=notes, auto_adj=auto, low_conf=sorted(set(low)), spread_caution=spread_caution,
        total_caution=total_caution, soft_notes=soft,
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
    # Expected value per unit risked at -110 for the model's side, and the best single book by
    # expected value (line and price together), using calibrated probabilities.
    sd_m, sd_t = state.get("margin_sd") or 16.0, state.get("total_sd") or 16.5
    if spread is not None and out.get("spread_side"):
        out["ev_spread"] = round(lines.ev(out["p_side_cover"], s.get("p_push_spread", 0), None), 3)
        out["best_spread"] = lines.best_line(mkt.get("books") or [], "spread", out["spread_side"],
                                             model=model_spread, sd=sd_m, shrink=shrink[0])
    if total is not None and out.get("total_side"):
        out["ev_total"] = round(lines.ev(out["p_side_total"], s.get("p_push_total", 0), None), 3)
        out["best_total"] = lines.best_line(mkt.get("books") or [], "total", out["total_side"],
                                            model=model_total, sd=sd_t, shrink=shrink[1])
    return out


def bet_eligible(p, kind, now):
    """A recorded bet needs a real sportsbook quote for that market (not the CollegeFootballData
    backup line) taken within the last C.BET_MAX_QUOTE_AGE_H hours."""
    m = p.get("market") or {}
    if m.get(f"{kind}_src", "odds") != "odds" or not m.get("books"):
        return False
    ts = m.get("last_ts")
    if not ts:
        return False
    return now - pd.Timestamp(ts) <= pd.Timedelta(hours=getattr(C, "BET_MAX_QUOTE_AGE_H", 36))


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
        # The first projection made for this game (by this model version), to show how much it has moved
        fm = old.get("first_model") if old.get("version") == version else None
        new["first_model"] = fm or dict(ts=now.isoformat(), spread=p["model_spread"], total=p["model_total"])
        for kind, line_key in (("spread", "spread"), ("total", "total")):
            fk = f"first_{kind}_flag"
            if old.get(fk) and old[fk].get("version") == version:
                new[fk] = old[fk]  # already bet at an earlier line: keep that line
            elif p.get(f"{kind}_flag") and not p["low_conf"] and not p.get(f"{kind}_caution") and bet_eligible(p, kind, now):
                # Graded at the consensus line (conservative); the best single book is kept for reference
                m = p.get("market") or {}
                new[fk] = dict(ts=now.isoformat(), side=p[f"{kind}_side"], version=version,
                               line=m.get(line_key), model=p[f"model_{kind}"], best=p.get(f"best_{kind}"),
                               quote_ts=m.get("last_ts"), books=len(m.get("books") or []),
                               p=p.get("p_side_cover" if kind == "spread" else "p_side_total"),
                               ev=p.get(f"ev_{kind}"), fit=p.get("fit_id"))
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
            close_spread=close_s, close_total=close_t, close_ts=mv.get("last_ts"),
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
        # Shadow record: flags that a caution kept from being bets, scored at the line the final
        # pre-kickoff prediction saw. Shown separately; never counted as bets.
        m = p.get("market") or {}
        if p.get("spread_flag") and not same(fs) and m.get("spread") is not None and p.get("spread_side"):
            res["ats_flagonly"] = _ats(hp - ap, p["spread_side"], m["spread"])
        if p.get("total_flag") and not same(ft) and m.get("total") is not None and p.get("total_side"):
            res["ou_flagonly"] = _ou(hp + ap, p["total_side"], m["total"])
        p["result"] = res
    return store
