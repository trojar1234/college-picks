"""Model 2.0 live run: projections for the current slate, frozen records, grading, and the site page.

Runs after the current model (v1) on every update, inside a guard so it can never stop v1 from
publishing. Full runs ("update", "retune") rebuild everything (about 2-3 minutes); line-only runs
("odds") reuse the last projections and only refresh the sportsbook comparison.

Records are append-only: each game's projection is updated until kickoff, then frozen and never
changed. Sportsbook lines are stored next to each projection for grading and comparison only.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import cfb_config as C
import v2_backtest as B
import v2_ensemble as VE
import v2_models as VM
import v2_sim as SIM

DIR = C.DATA / "v2"
SLATE = DIR / "slate.json"
BACKTEST = DIR / "backtest.json"
WEIGHTS = DIR / "weights.json"


def _rec_path(season):
    return DIR / f"predictions_{season}.json"


def _load(p, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _f(x, nd=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def _v1_frame(preds, store):
    """v1's projection for every game it has one for: live (preds) first, else its frozen record."""
    rows = {}
    for gid, p in (store or {}).items():
        if p.get("model_spread") is not None and p.get("model_total") is not None:
            rows[int(gid)] = (-float(p["model_spread"]), float(p["model_total"]))
    for p in preds or []:
        if p.get("model_spread") is not None and p.get("model_total") is not None:
            rows[int(p["game_id"])] = (-float(p["model_spread"]), float(p["model_total"]))
    return rows


def project(R, slate_ids, v1_live, mkt):
    """Projections for the slate from a backtest build R (whose last season is the current one)."""
    Gm = R["Gm"]
    g = Gm[Gm.game_id.isin(slate_ids)].copy()
    if g.empty:
        return []
    # v1's live number replaces its walk-forward number for upcoming games (same model, today's fit)
    for i, r in g.iterrows():
        if int(r.game_id) in v1_live:
            g.at[i, "m_v1"], g.at[i, "t_v1"] = v1_live[int(r.game_id)]
    season = int(g.season.max())
    w = R["weights"].get(season)
    if w:
        has = g[[f"m_{m}" for m in VE.MEMBERS if f"m_{m}" in g]].notna().all(axis=1).to_numpy()
        g["m_ens"] = np.where(has, VE.apply_weights(g.fillna({"m_v1": 0.0}), w["margin"]),
                              VE.apply_weights(g, w["margin_without_v1"]))
        g["t_ens"] = np.where(has, VE.apply_weights(g.fillna({"t_v1": 0.0}), w["total"]),
                              VE.apply_weights(g, w["total_without_v1"]))
        # error sizes recomputed with the projection actually published (v1's live number included)
        tr = Gm[(Gm.season < season) & Gm.m_ens.notna() & Gm.margin.notna()]
        if tr.season.nunique() >= 2:
            g["sd_m"] = VE.pred_sd(g, "margin", VE.fit_conf(tr, "margin", False), False)
            g["sd_t"] = VE.pred_sd(g, "total", VE.fit_conf(tr, "total", False), False)
    out = []
    for r in g.itertuples():
        m, t = float(r.m_ens), float(r.t_ens)
        sd_m = float(r.sd_m) if np.isfinite(r.sd_m) else 16.0
        sd_t = float(r.sd_t) if np.isfinite(r.sd_t) else 16.5
        mu_h, mu_a = max((t + m) / 2, 1.0), max((t - m) / 2, 1.0)
        drives = np.nanmean([r.drives_h, r.drives_a]) if np.isfinite(r.drives_h) or np.isfinite(r.drives_a) else 11.5
        mk = mkt.get(int(r.game_id), {})
        h, a = SIM.simulate(mu_h, mu_a, drives, sd_m, sd_t, n=20000, seed=int(r.game_id) % 100000)
        sm = SIM.summarize(h, a, (mk.get("spread"), mk.get("total")))
        comp = {k: [_f(getattr(r, f"m_{k}")), _f(getattr(r, f"t_{k}"))] for k in VE.MEMBERS if hasattr(r, f"m_{k}")}
        out.append(dict(game_id=int(r.game_id), start=pd.Timestamp(r.start).isoformat(), home=r.home, away=r.away,
                        drives=_f(drives, 2),
                        sim=[float(mu_h), float(mu_a), float(drives), float(sd_m), float(sd_t)],  # exact simulator inputs
                        neutral=bool(r.neutral), wk=int(r.wk), mismatch=int(r.mismatch),
                        margin=_f(m), spread=_f(-m), total=_f(t), sd_m=_f(sd_m), sd_t=_f(sd_t),
                        exp_home=_f(mu_h), exp_away=_f(mu_a), agree=str(r.agree),
                        models_sd=_f(np.nanstd([getattr(r, f"m_{k}") for k in VM.MODELS])),
                        comp=comp, market={k: _f(mk.get(k)) for k in ("spread", "total", "open_spread", "open_total")},
                        **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in sm.items()}))
    return out


def _grade(rec, G, closing=None):
    """closing: {game_id: (spread, total)} true closing lines (CollegeFootballData consensus) when
    available; otherwise the last line this run tracked before kickoff is used and marked as such."""
    res = G.set_index("game_id")
    closing = closing or {}
    for gid, p in rec.items():
        if p.get("result") or int(gid) not in res.index:
            continue
        r = res.loc[int(gid)]
        if not bool(r.completed) or not np.isfinite(r.home_pts):
            continue
        mar, tot = float(r.home_pts - r.away_pts), float(r.home_pts + r.away_pts)
        out = dict(home_pts=int(r.home_pts), away_pts=int(r.away_pts), err_margin=_f(mar - p["margin"]),
                   err_total=_f(tot - p["total"]))
        if p.get("v1"):
            out.update(v1_err_margin=_f(mar - p["v1"][0]), v1_err_total=_f(tot - p["v1"][1]))
        mk = dict(p.get("market") or {})
        cs, ct = closing.get(int(gid), (None, None))
        out["close_src"] = "cfbd" if cs is not None or ct is not None else "last tracked"
        if cs is not None:
            mk["spread"] = cs
        if ct is not None:
            mk["total"] = ct
        for lab, sk, tk in (("close", "spread", "total"), ("open", "open_spread", "open_total")):
            sp, tl = mk.get(sk), mk.get(tk)
            if sp is not None:
                out[f"{lab}_err_margin"] = _f(mar + sp)
                # sp is the home spread (negative = home favored); the model's home spread is -margin.
                # The model prefers home when its spread is below the line (home -10 vs a -3.5 line).
                side = 1 if -p["margin"] < sp else -1 if -p["margin"] > sp else 0  # 1 = home side
                v = (mar + sp) * side
                out[f"ats_{lab}"] = "win" if v > 0 else "loss" if v < 0 else "push" if side else None
            if tl is not None:
                out[f"{lab}_err_total"] = _f(tot - tl)
                side = 1 if p["total"] > tl else -1 if p["total"] < tl else 0
                v = (tot - tl) * side
                out[f"ou_{lab}"] = "win" if v > 0 else "loss" if v < 0 else "push" if side else None
        out["p_home"] = p.get("p_home")
        out["home_won"] = mar > 0
        p["result"] = out


def scorecard(rec):
    """Live record since Model 2.0 started (frozen pre-kickoff projections only)."""
    g = [p for p in rec.values() if p.get("result") and not p.get("mismatch")]
    if not g:
        return dict(n=0)

    def mae(k):
        v = [abs(p["result"][k]) for p in g if p["result"].get(k) is not None]
        return (round(float(np.mean(v)), 2), len(v)) if v else (None, 0)

    def rec_(k):
        v = [p["result"].get(k) for p in g]
        w, l, pu = v.count("win"), v.count("loss"), v.count("push")
        return dict(w=w, l=l, p=pu, pct=round(w / (w + l), 3) if w + l else None)

    br = [(p["result"]["p_home"] - float(p["result"]["home_won"])) ** 2 for p in g if p["result"].get("p_home") is not None]
    return dict(n=len(g), since=min(p["first_ts"] for p in g),
                v2=dict(margin=mae("err_margin"), total=mae("err_total"), brier=round(float(np.mean(br)), 4) if br else None),
                v1=dict(margin=mae("v1_err_margin"), total=mae("v1_err_total")),
                close=dict(margin=mae("close_err_margin"), total=mae("close_err_total")),
                open=dict(margin=mae("open_err_margin"), total=mae("open_err_total")),
                leans=dict(spread_open=rec_("ats_open"), spread_close=rec_("ats_close"),
                           total_open=rec_("ou_open"), total_close=rec_("ou_close")))


def run(mode, SEASONS, pfeat, passing, v1_wf, preds, store, season, wk, now, mview):
    DIR.mkdir(parents=True, exist_ok=True)
    full = mode in ("update", "retune") or not SLATE.exists()
    games = SEASONS[season]["games_all"]
    upcoming = games[(pd.to_datetime(games.start, utc=True) > now + pd.Timedelta(minutes=5))]
    slate_ids = set(int(p["game_id"]) for p in preds or [])
    if not slate_ids and len(upcoming):  # v1 had nothing (e.g. all lines missing): use the next week
        slate_ids = set(upcoming[upcoming.wk == upcoming.wk.min()].game_id.astype(int))
    mkt = {}
    for gid in slate_ids:
        m = (mview or {}).get(int(gid), {}) or {}
        p = next((x for x in preds or [] if int(x["game_id"]) == gid), None)
        pm = (p or {}).get("market") or {}
        mkt[gid] = dict(spread=pm.get("spread", m.get("spread")), total=pm.get("total", m.get("total")),
                        open_spread=pm.get("open_spread", m.get("open_spread")), open_total=pm.get("open_total", m.get("open_total")))
    v1_live = _v1_frame(preds, None)
    if full:
        R = B.build(SEASONS, pfeat, passing, v1_wf)
        proj = project(R, slate_ids, v1_live, mkt)
        rep = B.report(R["Gm"])
        BACKTEST.write_text(json.dumps(dict(made=now.isoformat(), report=rep), default=str))
        WEIGHTS.write_text(json.dumps({str(k): v for k, v in R["weights"].items()}, default=str))
        G = R["G"]
    else:  # odds-only: keep the projections, refresh the sportsbook comparison
        proj = [p for p in _load(SLATE, {}).get("games", []) if p["game_id"] in slate_ids]
        for p in proj:
            p["market"] = {k: _f(v) for k, v in mkt.get(p["game_id"], {}).items()}
            # identical football inputs and seed as the full run; only the chances at the new lines change
            mu_h, mu_a, dr, sm_, st_ = p.get("sim") or [p["exp_home"], p["exp_away"], p.get("drives") or 11.5, p["sd_m"], p["sd_t"]]
            h, a = SIM.simulate(mu_h, mu_a, dr, sm_, st_, n=20000, seed=p["game_id"] % 100000)
            sm = SIM.summarize(h, a, (p["market"].get("spread"), p["market"].get("total")))
            for k in ("p_home_cover", "p_push_spread", "p_over", "p_push_total"):
                p.pop(k, None)
                if k in sm:
                    p[k] = round(sm[k], 3)
        G = None
    SLATE.write_text(json.dumps(dict(made=now.isoformat(), season=season, week=wk, games=proj), default=str))

    # frozen record: update until kickoff, then never again
    rec = _load(_rec_path(season), {})
    for p in proj:
        if pd.Timestamp(p["start"]) <= now + pd.Timedelta(minutes=5):
            continue
        k = str(p["game_id"])
        first = (rec.get(k) or {}).get("first_ts") or now.isoformat()
        rec[k] = dict(p, first_ts=first, ts=now.isoformat(), v1=list(v1_live.get(p["game_id"], [])) or None)
        if rec[k].get("result"):
            rec[k].pop("result")
    if G is None:
        import v2_data as VD
        G = VD.games_table({season: SEASONS[season]})
    closing = {}
    try:
        L = SEASONS[season]["lines"]
        for r in L.itertuples():
            sp = r.cfbd_spread if np.isfinite(r.cfbd_spread) else None
            tl = r.cfbd_total if np.isfinite(r.cfbd_total) else None
            closing[int(r.game_id)] = (sp, tl)
    except Exception as e:
        print("Model 2.0: closing lines unavailable:", e)
    _grade(rec, G, closing)
    _rec_path(season).write_text(json.dumps(rec, default=str))

    bt = _load(BACKTEST, {})
    w = _load(WEIGHTS, {}).get(str(season))
    import v2_site
    v2_site.build(dict(generated=now.isoformat(), season=season, week=wk, games=sorted(proj, key=lambda p: p["start"]),
                       live=scorecard(rec), backtest=bt.get("report"), backtest_made=bt.get("made"), weights=w,
                       record=[dict(game_id=int(k), **{x: v.get(x) for x in ("home", "away", "start", "spread", "total", "p_home",
                                                                            "market", "result", "v1")})
                               for k, v in rec.items() if v.get("result")]))
    return proj
