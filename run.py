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
import cfb_fetch as fetch
import cfb_lines as lines
import cfb_model as model
import cfb_predict as predict
import cfb_report as report
import cfb_site as site

STATE = C.DATA / "model_state.json"


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

    # Learning step 1: tune prior strength (first run, monthly, or on request).
    lam = state_old.get("lam")
    tuned_at = pd.Timestamp(state_old["tuned_at"]) if state_old.get("tuned_at") else None
    if lam is None or mode == "retune" or (full and tuned_at is not None and now - tuned_at > pd.Timedelta(days=30)):
        print("Tuning prior strength on the backtest ...")
        lam, tune_res = model.tune(SEASONS, C.LAMBDA_GRID, season)
        tuned_at = now
    else:
        tune_res = state_old.get("tune_results", {})

    # Learning step 2: refit ratings, points conversion, and simulator variance on all data.
    print(f"Fitting model (prior strength {lam}) ...")
    state, finals, bt, wf, info = model.fit_state(SEASONS, lam, season)
    state.update(lam=lam, tuned_at=tuned_at.isoformat(), tune_results={str(k): v for k, v in tune_res.items()},
                 updated=now.isoformat())
    STATE.write_text(json.dumps(state, indent=1))

    # Lines
    snap, unmatched = lines.snapshot_rows(fetch.odds_snapshot(), games, now)
    hist = lines.append(snap)
    mview = lines.market_view(hist, games)
    cfbd_ln = SEASONS[season]["lines"].set_index("game_id") if len(SEASONS[season]["lines"]) else None

    # Backtest report first: its calibration tells us how far to trust cover probabilities.
    bt_rep = report.backtest(wf[wf.season < season] if len(wf) else wf, SEASONS)
    shrink = (bt_rep.get("shrink_spread", 1.0), bt_rep.get("shrink_total", 1.0))

    # Predictions for this week's games that haven't kicked off
    slate, wk = predict.current_slate(games, now)
    R = finals[season]
    tiers = finals[season]["tiers"]
    ovr = predict.load_overrides()
    wx = predict.game_weather(slate, now) if len(slate) else {}
    gp = int(games.completed.sum() / max(1, len(set(games.home) | set(games.away))) * 2)
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
        preds.append(predict.predict_game(r, state, R, m, wx.get(r.game_id), ovr, tiers, gp, shrink))

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
    )
    site.build(season, wk, slate, store, rep, finals[season], now, unmatched)
    print(f"Done. Week {wk}: {len(preds)} games projected, {sum(p.get('spread_flag', False) for p in preds)} spread edges.")
    if unmatched:
        print("Unmatched odds games:", unmatched[:10])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "update")
