"""Entry point.

    python run.py update   # full run: pull results, update ratings, grade, predict, rebuild site
    python run.py odds     # lines-only run: new line snapshot, re-predict, rebuild site (no CFBD calls)
    python run.py retune   # full run that also re-tunes the prior strength
"""
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

import cfb_config as C
import cfb_data as Dt
import cfb_extra as extra
import cfb_fetch as fetch
import cfb_lines as lines
import cfb_model as model
import cfb_predict as predict
import cfb_report as report
import cfb_site as site

STATE = C.DATA / "model_state.json"
EXPERIMENTS = C.DATA / "experiments.json"


def current_season(now):
    return now.year if now.month >= 3 else now.year - 1


def main(mode):
    now = pd.Timestamp(os.environ.get("CFB_NOW") or datetime.now(timezone.utc))  # CFB_NOW: testing only
    season = current_season(now)
    full = mode in ("update", "retune")
    state_old = json.loads(STATE.read_text()) if STATE.exists() else {}

    print(f"Loading seasons {C.FIRST_SEASON}-{season} ...")
    SEASONS = {}
    for y in range(C.FIRST_SEASON, season + 1):
        sd = fetch.season_data(y, refresh=(full and y == season))
        SEASONS[y] = Dt.load_season(sd, y)
    games = SEASONS[season]["games"]
    if games.empty:
        raise SystemExit(f"No {season} games found. Check the CFBD key.")

    # Seasons used for tuning, and one untouched holdout season for an honest check.
    tune_seasons = list(range(C.FIRST_SEASON + 2, season - 1))
    holdout = season - 1

    # Extra data for the optional feature groups (downloads capped; odds-only runs use cache only).
    if not full:
        extra.CFBD_BUDGET, extra.WEATHER_BUDGET = 0, 0
    print("Preparing extra features ...")
    gctx, prior_extra, available, xstatus = extra.build(SEASONS, season, tune_seasons[0])
    available["score"] = available["direct"] = True  # built from data the model already has
    for k, v in xstatus.items():
        print(f"  {k}: {v}")
    avail_key = sorted(k for k in available if not k.startswith("_"))

    # Learning step 1: tune prior strength (first run, monthly, or on request).
    lam = state_old.get("lam")
    tuned_at = pd.Timestamp(state_old["tuned_at"]) if state_old.get("tuned_at") else None
    retuned = False
    if lam is None or mode == "retune" or (full and tuned_at is not None and now - tuned_at > pd.Timedelta(days=30)):
        print("Tuning prior strength on the tuning seasons ...")
        prev = state_old.get("features") or {}
        lam, tune_res = model.tune(SEASONS, C.LAMBDA_GRID, tune_seasons,
                                   groups=[g for g in prev.get("groups", ["score"]) if g in available], gctx=gctx)
        tuned_at, retuned = now, True
    else:
        tune_res = state_old.get("tune_results", {})

    # Learning step 2: test each optional feature; keep only those that improve the tuning seasons.
    features = state_old.get("features")
    exp_old = json.loads(EXPERIMENTS.read_text()) if EXPERIMENTS.exists() else {}
    if full and (features is None or retuned or exp_old.get("available") != avail_key):
        print("Testing optional features ...")
        features, results = model.select_features(SEASONS, lam, tune_seasons, holdout, gctx, prior_extra, available)
        exp_old = dict(run=now.isoformat(), available=avail_key, status=xstatus, chosen=features,
                       tune_seasons=[tune_seasons[0], tune_seasons[-1]], **results)
        EXPERIMENTS.write_text(json.dumps(exp_old, indent=1, default=str))
    features = features or {}
    # Drop any feature whose data has since become unavailable.
    features = dict(recency=features.get("recency"),
                    priors=bool(features.get("priors")) and "priors" in available,
                    groups=[g for g in features.get("groups", []) if g in available])
    print("Features in use:", features)

    # Learning step 3: refit ratings, points conversion, and simulator variance on all data.
    print(f"Fitting model (prior strength {lam}) ...")
    state, finals, bt, wf, info = model.fit_state(SEASONS, lam, season, features, gctx, prior_extra)
    state.update(lam=lam, tuned_at=tuned_at.isoformat(), tune_results={str(k): v for k, v in tune_res.items()},
                 updated=now.isoformat(), features=features)
    STATE.write_text(json.dumps(state, indent=1))

    # Lines
    snap, unmatched = lines.snapshot_rows(fetch.odds_snapshot(), games, now)
    hist = lines.append(snap)
    mview = lines.market_view(hist, games)
    cfbd_ln = SEASONS[season]["lines"].set_index("game_id") if len(SEASONS[season]["lines"]) else None

    # Backtest report first: its calibration tells us how far to trust cover probabilities.
    bt_rep = report.backtest(wf[wf.season < season] if len(wf) else wf, SEASONS)
    for srow in bt_rep.get("seasons", []):
        srow["holdout"] = srow["season"] == holdout
    shrink = (bt_rep.get("shrink_spread", 1.0), bt_rep.get("shrink_total", 1.0))

    # Predictions for this week's games that haven't kicked off
    slate, wk = predict.current_slate(games, now)
    R = finals[season]
    tiers = finals[season]["tiers"]
    ovr = predict.load_overrides()
    wx = predict.game_weather(slate, now) if len(slate) else {}
    gp = int(games.completed.sum() / max(1, len(set(games.home) | set(games.away))) * 2)
    gidx = gctx.set_index("game_id") if len(gctx) else None
    preds = []
    for r in slate.itertuples():
        if r.start <= now:
            continue
        m = dict(mview.get(int(r.game_id), {}))
        if m.get("spread") is None and cfbd_ln is not None and r.game_id in cfbd_ln.index:
            m["spread"] = lines._f(cfbd_ln.loc[r.game_id, "cfbd_spread"])
            m["open_spread"] = m.get("open_spread") or lines._f(cfbd_ln.loc[r.game_id, "cfbd_spread_open"])
        if m.get("total") is None and cfbd_ln is not None and r.game_id in cfbd_ln.index:
            m["total"] = lines._f(cfbd_ln.loc[r.game_id, "cfbd_total"])
            m["open_total"] = m.get("open_total") or lines._f(cfbd_ln.loc[r.game_id, "cfbd_total_open"])
        grow = gidx.loc[r.game_id].to_dict() if gidx is not None and r.game_id in gidx.index else {}
        preds.append(predict.predict_game(r, state, R, m, wx.get(r.game_id), ovr, tiers, gp, shrink, grow))

    store = predict.load_store()
    store = predict.update_store(store, preds, now)
    store = predict.grade(store, games, mview, SEASONS[season]["lines"])
    predict.save_store(store)

    # Report card
    rep = dict(
        backtest=bt_rep,
        live=report.live(store, season),
        state={k: state[k] for k in ("lam", "rho", "sim_sd", "margin_sd", "total_sd", "updated", "tuned_at")},
        tune=state["tune_results"],
        experiments=exp_old,
        features=features,
    )
    site.build(season, wk, slate, store, rep, finals[season], now, unmatched)
    print(f"Done. Week {wk}: {len(preds)} games projected, {sum(p.get('spread_flag', False) for p in preds)} spread edges.")
    if unmatched:
        print("Unmatched odds games:", unmatched[:10])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "update")
