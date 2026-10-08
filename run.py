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
import cfb_players as players
import cfb_fetch as fetch
import cfb_lines as lines
import cfb_model as model
import cfb_predict as predict
import cfb_report as report
import cfb_site as site

STATE = C.DATA / "model_state.json"
EXPERIMENTS = C.DATA / "experiments.json"
MODEL_VERSION = 5  # bump when the model's structure changes, to force re-tuning and re-testing


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

    # Extra data (downloads capped; line-only runs use what's cached).
    if not full:
        extra.CFBD_BUDGET, extra.PBP_BUDGET, players.PLAYER_BUDGET = 0, 0, 0
    print("Preparing extra data ...")
    qb_ctx, available, xstatus, pbp = extra.build(SEASONS, season, tune_seasons[0])
    # Attach play-by-play metrics to each team-game row (NaN where not downloaded yet).
    for y, S in SEASONS.items():
        a = S["adv"]
        if len(a):
            a = a.drop(columns=[c for c in ("ppd", "fin", "fpos") if c in a]).merge(pbp, on=["game_id", "off"], how="left")
            S["adv"] = a
    # Player-level preseason data (QB quality, returning production counting transfers).
    pseasons = list(range(C.FIRST_SEASON + 1, season + 1))
    pmissing = players.ensure_data(pseasons, season, C.FIRST_SEASON + 1)
    pfeat, pstat, pcount = players.build(pseasons)
    need = len(pseasons) - 1  # every season but the newest must have data for a fair test
    def _short(lst):
        return ", ".join(lst[:4]) + (" ..." if len(lst) > 4 else "")
    xstatus["players"] = pstat
    if pmissing["core"]:
        available["_why_players"] = f"partial: still downloading {_short(pmissing['core'])}"
    elif pcount["qb"] >= need - 1 and pcount["defense"] >= need:
        available["players"] = True
    else:
        available["_why_players"] = "not enough seasons of QB/defense data"
    if pmissing["off"]:
        available["_why_players_off"] = f"partial: still downloading {_short(pmissing['off'])}"
    elif "players" in available and pcount["offense"] >= need:
        available["players_off"] = True
    else:
        available["_why_players_off"] = available.get("_why_players") or "player EPA data not available"
    for k in ("players", "players_off"):
        if f"_why_{k}" in available:
            xstatus[k] = available[f"_why_{k}"]
    for k, v in xstatus.items():
        print(f"  {k}: {v}")
    avail_key = sorted(k for k in available if not k.startswith("_"))
    exp_old = json.loads(EXPERIMENTS.read_text()) if EXPERIMENTS.exists() else {}
    new_version = exp_old.get("version") != MODEL_VERSION

    # Learning step 1: tune prior strength (first run, new model version, monthly, or on request).
    lam = None if new_version else state_old.get("lam")
    tuned_at = pd.Timestamp(state_old["tuned_at"]) if state_old.get("tuned_at") else None
    retuned = False
    if lam is None or mode == "retune" or (full and tuned_at is not None and now - tuned_at > pd.Timedelta(days=30)):
        print("Tuning prior strength on the tuning seasons ...")
        prev = (state_old.get("features") or {}) if not new_version else {}
        lam, tune_res = model.tune(SEASONS, C.LAMBDA_GRID, tune_seasons,
                                   [g for g in prev.get("groups", model.BASE_GROUPS) if g in model.BASE_GROUPS or g in available],
                                   pfeat, tuple(prev.get("players") or ()))
        tuned_at, retuned = now, True
    else:
        tune_res = state_old.get("tune_results", {})

    # Learning step 2: test optional features; keep only those that clearly improve the tuning seasons.
    features = None if new_version else state_old.get("features")
    if full and (features is None or retuned or exp_old.get("available") != avail_key):
        print("Testing optional features ...")
        features, results = model.select_features(SEASONS, lam, tune_seasons, holdout, available, pfeat)
        exp_old = dict(version=MODEL_VERSION, run=now.isoformat(), available=avail_key, status=xstatus,
                       chosen=features, tune_seasons=[tune_seasons[0], tune_seasons[-1]], **results)
        EXPERIMENTS.write_text(json.dumps(exp_old, indent=1, default=str))
    features = features or {}
    groups = [g for g in features.get("groups", []) if g in model.BASE_GROUPS or g in available]
    use = list(features.get("players") or [])
    if use and not ("players_off" in available if "off" in use else "players" in available):
        use = []  # data for the chosen player option is no longer available
    features = dict(groups=groups or list(model.BASE_GROUPS), players=use)
    print("Model inputs in use:", features["groups"], "player data:", use or "none")

    # Learning step 3: refit ratings, points conversion, and simulator variance on all data.
    print(f"Fitting model (prior strength {lam}) ...")
    state, finals, bt, wf, info = model.fit_state(SEASONS, lam, season, features, pfeat)
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
    gidx = qb_ctx.set_index("game_id") if len(qb_ctx) else None
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
