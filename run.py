"""Entry point.

    python run.py update   # full run: pull results, update ratings, grade, predict, rebuild site
    python run.py odds     # lines-only run: new line snapshot, re-predict, rebuild site (no CFBD calls)
    python run.py retune   # full run that also re-tunes the prior strength
"""
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
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


DECISIONS = C.DATA / "decisions.csv"
DEC_COLS = ["ts", "game_id", "game", "kickoff", "market", "status", "side", "line", "model", "edge",
            "p", "ev", "best_book", "best_line", "best_price", "line_source", "quote_ts", "reasons", "fit_id"]


def log_decisions(preds, store, old_flags, ts, fit_id):
    """Append-only record of every candidate (bets and flags a caution blocked), written the first time
    each game/market reaches that status. Never edited afterward."""
    seen = set()
    if DECISIONS.exists():
        with DECISIONS.open() as f:
            seen = {(r["game_id"], r["market"], r["status"]) for r in csv.DictReader(f)}
    rows = []
    for p in preds:
        gid = str(p["game_id"])
        rec = store.get(gid, {})
        for i, kind in enumerate(("spread", "total")):
            if not p.get(f"{kind}_flag"):
                continue
            is_bet = bool(rec.get(f"first_{kind}_flag")) and not old_flags.get(gid, (False, False))[i]
            reasons = [*p.get("low_conf", []), *p.get(f"{kind}_caution", [])]
            m = p.get("market") or {}
            if not reasons and not predict.bet_eligible(p, kind, ts):
                reasons = ["no fresh sportsbook quote"]
            status = "bet" if is_bet else ("flag" if reasons else None)
            if status is None or (gid, kind, status) in seen:
                continue
            b = p.get(f"best_{kind}") or {}
            rows.append(dict(ts=ts.isoformat(), game_id=gid, game=f"{p['away']} @ {p['home']}", kickoff=p["start"],
                             market=kind, status=status, side=p.get(f"{kind}_side"), line=m.get(kind),
                             model=p.get(f"model_{kind}"), edge=p.get(f"{kind}_edge"),
                             p=p.get("p_side_cover" if kind == "spread" else "p_side_total"), ev=p.get(f"ev_{kind}"),
                             best_book=b.get("book"), best_line=b.get("line"), best_price=b.get("price"),
                             line_source=m.get(f"{kind}_src"), quote_ts=m.get("last_ts"), reasons="; ".join(reasons),
                             fit_id=fit_id))
    if rows:
        new = not DECISIONS.exists()
        with DECISIONS.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DEC_COLS)
            if new:
                w.writeheader()
            w.writerows(rows)


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
    if games.empty and season - 1 in SEASONS and not SEASONS[season - 1]["games"].empty:
        # Offseason before the new schedule is published: keep showing last season.
        print(f"No {season} schedule yet; showing {season - 1}.")
        SEASONS.pop(season)
        season -= 1
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
    if C.TOTALS_MODEL:
        available["totals"] = True  # see cfb_config.TOTALS_MODEL
    else:
        available["_why_totals"] = "switched off: more accurate totals, but its over/under bets did worse against opening lines"
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
    if full and (features is None or retuned or exp_old.get("available") != avail_key
                 or exp_old.get("keep_rule") != model.KEEP_RULE):  # re-test when the keep rule changes
        print("Testing optional features ...")
        features, results = model.select_features(SEASONS, lam, tune_seasons, holdout, available, pfeat)
        exp_old = dict(version=MODEL_VERSION, keep_rule=model.KEEP_RULE, run=now.isoformat(), available=avail_key, status=xstatus,
                       chosen=features, tune_seasons=[tune_seasons[0], tune_seasons[-1]], **results)
        EXPERIMENTS.write_text(json.dumps(exp_old, indent=1, default=str))
    for t in exp_old.get("tests", []):  # keep "not tested" reasons current between full re-tests
        why = available.get(f"_why_{t.get('group')}")
        if t.get("skipped") and why:
            t["skipped"] = why
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
    # Fingerprint of the fitted model (coefficients, error sizes, corrections) and the code version,
    # saved with every prediction so any number can be traced to the exact fit that made it.
    fit_id = hashlib.sha1(json.dumps({k: state.get(k) for k in ("conv", "groups", "lam", "sim_sd", "rho", "margin_sd",
                                                                   "total_sd", "academy", "features")},
                                     sort_keys=True, default=str).encode()).hexdigest()[:10]
    (C.DATA / "states").mkdir(exist_ok=True)
    (C.DATA / "states" / f"{season}-{now.strftime('%m-%d')}.json").write_text(
        json.dumps(dict(fit_id=fit_id, saved=now.isoformat(), **{k: state.get(k) for k in (
            "conv", "groups", "lam", "sim_sd", "rho", "margin_sd", "total_sd", "academy", "features", "plays_per_drive")}),
            default=str))

    # Lines
    odds = fetch.odds_snapshot()
    # Stamp lines with the time they were actually fetched (the model fit takes minutes), so a quote
    # taken after kickoff is never treated as pre-game.
    fetched = now if os.environ.get("CFB_NOW") else max(now, pd.Timestamp(datetime.now(timezone.utc)))
    snap, unmatched = lines.snapshot_rows(odds, games, fetched)
    hist = lines.append(snap)
    mview = lines.market_view(hist, games)
    cfbd_ln = SEASONS[season]["lines"].set_index("game_id") if len(SEASONS[season]["lines"]) else None

    # Backtest report first: its calibration tells us how far to trust cover probabilities.
    # QB changes for every past season (cached box scores only), so the backtest can replay that caution
    try:
        ag = pd.concat([S["games"] for S in SEASONS.values() if len(S["games"])], ignore_index=True)
        qtab, _ = extra.qb_table(ag, extra.Budget(0), C.FIRST_SEASON, season)
        qb_hist = extra.qb_deltas(ag, qtab) if qtab is not None else None
    except Exception as e:  # the backtest still runs without it
        print("QB history for the backtest skipped:", e)
        qb_hist = None
    bt_rep = report.backtest(wf[wf.season < season] if len(wf) else wf, SEASONS, qb_hist)
    for srow in bt_rep.get("seasons", []):
        srow["holdout"] = srow["season"] == holdout
    shrink = (bt_rep.get("shrink_spread", 1.0), bt_rep.get("shrink_total", 1.0))

    # Predictions for this week's games that haven't kicked off
    slate, wk = predict.current_slate(games, now)
    R = finals[season]
    tiers = finals[season]["tiers"]
    ovr = predict.load_overrides()
    wx = predict.game_weather(slate, now) if len(slate) else {}
    # Games played by the typical FBS team (FCS opponents excluded from the count of teams)
    played = pd.concat([games[games.completed].home, games[games.completed].away]).value_counts()
    fbs_teams = [t for t, v in tiers.items() if v != "FCS"]
    gp = int(np.median([played.get(t, 0) for t in fbs_teams])) if fbs_teams else 0
    gidx = qb_ctx.set_index("game_id") if len(qb_ctx) else None
    preds = []
    for r in slate.itertuples():
        if r.start <= fetched + pd.Timedelta(minutes=5):
            continue  # kicked off (or about to): its line may already be live
        m = dict(mview.get(int(r.game_id), {}))
        # Where each line came from: "odds" = a sportsbook snapshot; "cfbd" = the backup line from
        # CollegeFootballData, shown for reference but never allowed to become a bet.
        m["spread_src"] = "odds" if m.get("spread") is not None else None
        m["total_src"] = "odds" if m.get("total") is not None else None
        if m.get("spread") is None and cfbd_ln is not None and r.game_id in cfbd_ln.index:
            m["spread"] = lines._f(cfbd_ln.loc[r.game_id, "cfbd_spread"])
            m["spread_src"] = "cfbd" if m["spread"] is not None else None
            if m.get("open_spread") is None:  # 0 is a real (pick'em) line, so test for missing explicitly
                m["open_spread"] = lines._f(cfbd_ln.loc[r.game_id, "cfbd_spread_open"])
        if m.get("total") is None and cfbd_ln is not None and r.game_id in cfbd_ln.index:
            m["total"] = lines._f(cfbd_ln.loc[r.game_id, "cfbd_total"])
            m["total_src"] = "cfbd" if m["total"] is not None else None
            if m.get("open_total") is None:
                m["open_total"] = lines._f(cfbd_ln.loc[r.game_id, "cfbd_total_open"])
        grow = gidx.loc[r.game_id].to_dict() if gidx is not None and r.game_id in gidx.index else {}
        pg = predict.predict_game(r, state, R, m, wx.get(r.game_id), ovr, tiers, gp, shrink, grow)
        pg["fit_id"] = fit_id
        preds.append(pg)

    # Publish gate: never overwrite the last good site with broken numbers. A failure here stops
    # the run before anything is saved, so the workflow publishes nothing and GitHub emails you.
    def _ok(p):
        try:
            sp, tot = float(p["model_spread"]), float(p["model_total"])
        except (KeyError, TypeError, ValueError):
            return False
        return np.isfinite(sp) and np.isfinite(tot) and 10 <= tot <= 130 and abs(sp) <= 90
    bad = [f"{p['away']} @ {p['home']}" for p in preds if not _ok(p)]
    if bad:
        raise SystemExit(f"Refusing to publish: impossible projections for {', '.join(bad[:5])}")
    if not tiers or not any(v != "FCS" for v in tiers.values()):
        raise SystemExit("Refusing to publish: no team ratings")

    store = predict.load_store()
    version = f"{MODEL_VERSION}|spread {C.SPREAD_EDGE}|total {C.TOTAL_EDGE}"
    # Decisions are stamped when the odds they used were fetched (the model fit takes minutes)
    old_flags = {k: (bool(v.get("first_spread_flag")), bool(v.get("first_total_flag"))) for k, v in store.items()}
    store = predict.update_store(store, preds, fetched, version)
    log_decisions(preds, store, old_flags, fetched, fit_id)
    store = predict.grade(store, games, mview, SEASONS[season]["lines"])
    predict.save_store(store)

    # Report card
    rep = dict(
        backtest=bt_rep,
        live=dict(report.live(store, season, state.get("margin_sd"), state.get("total_sd")), value_decay=report.value_decay(store, hist, season),
                  flag_clv=report.flag_clv(store, pd.read_csv(DECISIONS) if DECISIONS.exists() else None, season)),
        state={k: state[k] for k in ("lam", "rho", "sim_sd", "margin_sd", "total_sd", "updated", "tuned_at")},
        tune=state["tune_results"],
        experiments=exp_old,
        features=features,
    )
    # Ratings as of a week ago (before the latest completed week), for the rating-movement column
    prev_final = None
    done = games[games.completed]
    if len(done):
        last = done.wk.max()
        cut = done[done.wk == last].start.min()
        if (done.start < cut).any():
            prev_final = dict(model.ratings_from(model._rows_before(SEASONS[season], cut), info[season], lam),
                              tiers=finals[season]["tiers"])
    lines_ts = hist.ts.max().isoformat() if len(hist) and "ts" in hist else None

    # Each team's power rating after every completed week (for the rating-history chart)
    rating_hist = {}
    try:
        if len(done):
            cuts = [(0, games.start.min())]
            for w in sorted(done.wk.unique()):
                nxt = games[games.wk > w].start.min()
                cuts.append((int(w), nxt if nxt == nxt else done.start.max() + pd.Timedelta(days=1)))
            for w, cut in cuts:
                Rw = dict(model.ratings_from(model._rows_before(SEASONS[season], cut), info[season], lam),
                          tiers=finals[season]["tiers"])
                for row in site.team_table(Rw, state):
                    rating_hist.setdefault(row["team"], []).append([w, round(float(row["power"]), 1)])
    except Exception as e:  # a chart is never worth failing the run over
        print("Rating history skipped:", e)
        rating_hist = {}

    # Data health: what this run had to work with, shown on the Performance page
    graded_season = [p for p in store.values() if p.get("result") and p.get("start")
                     and report.season_of(p["start"]) == season]
    health = dict(
        run=now.isoformat(), mode=mode, build=(os.environ.get("GITHUB_SHA") or "")[:7],
        model_version=MODEL_VERSION, slate_games=int(len(slate)), projected=len(preds),
        skipped_started=int(len(slate)) - len(preds),
        with_spread=sum(1 for p in preds if (p.get("market") or {}).get("spread") is not None),
        with_total=sum(1 for p in preds if (p.get("market") or {}).get("total") is not None),
        odds_events=len(odds or []), odds_rows=int(len(snap)), unmatched_odds=len(unmatched),
        lines_fetched=lines_ts, rated_teams=sum(1 for v in tiers.values() if v != "FCS"),
        schedule_games=int(len(games)), completed_games=int(games.completed.sum()),
        graded=len(graded_season),
        backup_lines=sum(1 for p in preds if "cfbd" in ((p.get("market") or {}).get("spread_src"), (p.get("market") or {}).get("total_src"))),
        stale_quotes=sum(1 for p in preds if (p.get("market") or {}).get("last_ts") and fetched - pd.Timestamp(p["market"]["last_ts"]) > pd.Timedelta(hours=C.BET_MAX_QUOTE_AGE_H)),
        oddities=(snap.attrs.get("oddities") or [])[:10], academy=state.get("academy") or {}, fit_id=fit_id,
        cautions=dict(early_season_blocks_spreads=getattr(C, "EARLY_SEASON_BLOCKS_SPREADS", True),
                      fcs_blocks_spreads=getattr(C, "FCS_BLOCKS_SPREADS", True)),
        bets_graded=sum(1 for p in graded_season for k in ("ats", "ou") if (p["result"] or {}).get(k)),
    )
    print("Health:", health)
    site.build(season, wk, slate, store, rep, finals[season], now, unmatched, state, prev_final, lines_ts, games,
               rating_hist=rating_hist, health=health)
    print(f"Done. Week {wk}: {len(preds)} games projected, {sum(p.get('spread_flag', False) for p in preds)} spread edges.")
    if unmatched:
        print("Unmatched odds games:", unmatched[:10])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "update")
