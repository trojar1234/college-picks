"""Season-by-season rating chain, walk-forward backtest, and the fitted pieces
that turn ratings into points. Re-running this is how the model learns:
every completed game updates the ratings, and every graded prediction feeds
the refit of the points conversion and the simulator's variance.

Optional feature groups (each kept only if it improves the backtest):
  recency  - recent games count more in the ratings (half-life in weeks)
  priors   - richer preseason priors: QB/OL recruiting, transfer portal,
             coaching changes, defensive returning production
  qb       - adjustment when a team's expected starting QB has changed
  travel   - rest-day advantage, travel distance, time zones crossed
  weather  - wind, rain, and cold at kickoff, fitted from past games
"""
import numpy as np
import pandas as pd

import cfb_data as Dt
import cfb_ratings as Rt
import cfb_sim as sim


def season_tiers(games):
    t = {}
    for r in games.itertuples():
        t[r.home] = Dt.team_tier(r.home, r.home_conf, r.home_cls, r.season)
        t[r.away] = Dt.team_tier(r.away, r.away_conf, r.away_cls, r.season)
    return t


def _completed_rows(S, games_subset, half_life=None):
    ids = set(games_subset.loc[games_subset.completed, "game_id"])
    a = S["adv"]
    rows = a[a.game_id.isin(ids)] if len(a) else a
    if half_life and len(rows):
        rows = rows.copy()
        age = (rows.start.max() - rows.start).dt.total_seconds() / (7 * 86400)
        rows["w"] = 0.5 ** (age / half_life)
    return rows


def game_feats(R, home, away, neutral):
    h = 0 if neutral else 1
    out = dict(h=h)
    for key, m in (("q", "ppa"), ("s", "sr"), ("p", "pace")):
        out[f"{key}_h"] = Rt.expect(R[m], home, away, h)
        out[f"{key}_a"] = Rt.expect(R[m], away, home, -h)
    return out


FIRST_G = {"ppa": (-0.15, 0.15), "sr": (-0.05, 0.05), "pace": (0.0, 0.0)}
FIRST_HFA = {"ppa": 0.03, "sr": 0.01, "pace": 0.5}


def run_chain(SEASONS, lam, backtest=True, opts=None, prior_extra=None):
    """Walk through seasons in order. Returns final ratings per season, the
    walk-forward backtest rows, and the context (priors) used per season.
    opts: {"recency": half-life in weeks or None, "priors": bool}."""
    opts = opts or {}
    hl = opts.get("recency")
    use_extra = bool(opts.get("priors")) and prior_extra
    seasons = sorted(SEASONS)
    first = seasons[0]
    finals, pairs, bt, info = {}, [], [], {}
    for s in seasons:
        S = SEASONS[s]
        g = S["games"]
        if g.empty or S["adv"].empty:
            continue
        tiers = season_tiers(g)
        ctx = dict(tiers=tiers, fcs=frozenset(t for t, v in tiers.items() if v == "FCS"), m={})
        feats = None
        if s == first or (s - 1) not in finals:
            for m, col in Rt.METRICS.items():
                ctx["m"][m] = dict(prior={}, mu0=float(S["adv"][col].mean()), hfa0=FIRST_HFA[m],
                                   default=(0.0, 0.0), g0=FIRST_G[m])
        else:
            prev = finals[s - 1]
            extra = prior_extra.get(s) if use_extra else None
            feats = Rt.prior_features(prev["abs"], S["talent"], S["ret"], extra)
            priors, defaults = Rt.make_prior(prev["abs"], feats, Rt.fit_prior_coefs(pairs, use_extra),
                                             prev["tiers"], use_extra)
            for m in Rt.METRICS:
                ctx["m"][m] = dict(prior=priors[m], mu0=prev[m]["mu"], hfa0=prev[m]["hfa"],
                                   default=defaults[m], g0=prev[m]["g"])
        if backtest and s != first:
            done = g[g.completed]
            for wk in sorted(done.wk.unique()):
                cut = done.loc[done.wk == wk, "start"].min()
                R = ratings_from(_completed_rows(S, g[(g.start < cut) & (g.wk < wk)], hl), ctx, lam)
                gp = int((g.completed & (g.start < cut)).sum())
                for r in done[done.wk == wk].itertuples():
                    bt.append(dict(season=s, wk=wk, game_id=r.game_id, home=r.home, away=r.away,
                                   home_pts=r.home_pts, away_pts=r.away_pts,
                                   home_tier=tiers.get(r.home), away_tier=tiers.get(r.away),
                                   games_played=gp, **game_feats(R, r.home, r.away, r.neutral)))
        R = ratings_from(_completed_rows(S, g, hl), ctx, lam)
        finals[s] = dict(R, tiers=tiers, abs={m: Rt.absolute(R[m]) for m in Rt.METRICS})
        info[s] = ctx
        if feats is not None:
            pairs.append((feats, finals[s]["abs"]))
    return finals, pd.DataFrame(bt), info


def ratings_from(rows, ctx, lam):
    out = {}
    for m, col in Rt.METRICS.items():
        c = ctx["m"][m]
        out[m] = Rt.fit(rows, col, c["prior"], c["mu0"], c["hfa0"], lam, c["default"], ctx["fcs"], c["g0"])
    return out


# ---------- Points conversion: ratings (+ game context) -> expected points ----------

# Extra columns each optional group adds to the conversion, built per side (team on offense).
GROUP_COLS = {
    "qb": ["qbd"],
    "travel": ["rest_adv", "trav", "tz"],
    "weather": ["wind_over", "precip", "cold"],
}
CONV_DECAY = 0.6  # weight per season of age: scoring environments drift (rule changes, pace)


def side_frame(bt, gctx=None):
    """Two rows per game (home offense, away offense) with every column the conversion may use."""
    hs = pd.DataFrame(dict(season=bt.season.values, game_id=bt.game_id.values, P=bt.p_h.values,
                           Q=bt.q_h.values, S=bt.s_h.values, h=bt.h.values,
                           pts=bt.home_pts.values if "home_pts" in bt else np.nan, side="h"))
    aw = pd.DataFrame(dict(season=bt.season.values, game_id=bt.game_id.values, P=bt.p_a.values,
                           Q=bt.q_a.values, S=bt.s_a.values, h=-bt.h.values,
                           pts=bt.away_pts.values if "away_pts" in bt else np.nan, side="a"))
    df = pd.concat([hs, aw], ignore_index=True)
    cols = [c for g in GROUP_COLS.values() for c in g]
    if gctx is not None and len(gctx):
        gx = gctx.set_index("game_id")
        for c in cols:
            if c in ("wind_over", "precip", "cold"):
                df[c] = df.game_id.map(gx[c]) if c in gx else 0.0
            else:
                own = df.game_id.map(gx[f"{c}_h"]) if f"{c}_h" in gx else np.nan
                opp = df.game_id.map(gx[f"{c}_a"]) if f"{c}_a" in gx else np.nan
                df[c] = np.where(df.side == "h", own, opp)
    for c in cols:
        if c not in df:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def conv_X(df, groups=()):
    P, Q, S, h = (df[c].to_numpy(dtype=float) for c in ("P", "Q", "S", "h"))
    cols = [np.ones_like(P), P, P * Q, P * S, h]
    for g in groups:
        for c in GROUP_COLS[g]:
            cols.append(df[c].to_numpy(dtype=float))
    return np.column_stack(cols)


def fit_conv(df, groups=(), decay=None):
    """Least squares from ratings to points, weighting recent seasons more."""
    decay = CONV_DECAY if decay is None else decay
    X = conv_X(df, groups)
    y = df.pts.to_numpy(float)
    w = np.sqrt(decay ** (df.season.max() - df.season.to_numpy()))
    return np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]


def predict_bt(bt, conv, groups=(), gctx=None):
    out = bt.copy()
    df = side_frame(bt, gctx)
    pred = conv_X(df, groups) @ np.asarray(conv)
    n = len(bt)
    out["pred_h"] = pred[:n]
    out["pred_a"] = pred[n:]
    return out


def walk_forward(bt, eval_from, groups=(), gctx=None):
    """Score each season with a points conversion fit only on earlier seasons."""
    parts = []
    for s in sorted(bt.season.unique()):
        if s < eval_from:
            continue
        train = bt[bt.season < s]
        if len(train) < 300:
            continue
        conv = fit_conv(side_frame(train, gctx), groups)
        tp = predict_bt(train, conv, groups, gctx)
        sd_m = float(((tp.home_pts - tp.away_pts) - (tp.pred_h - tp.pred_a)).std())
        sd_t = float(((tp.home_pts + tp.away_pts) - (tp.pred_h + tp.pred_a)).std())
        p = predict_bt(bt[bt.season == s], conv, groups, gctx)
        p["sd_m"], p["sd_t"] = sd_m, sd_t
        parts.append(p)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def score(wf, seasons):
    """Margin and total mean absolute error over the given seasons."""
    w = wf[wf.season.isin(seasons)]
    if w.empty:
        return dict(margin=None, total=None, n=0)
    return dict(
        margin=round(float(((w.home_pts - w.away_pts) - (w.pred_h - w.pred_a)).abs().mean()), 3),
        total=round(float(((w.home_pts + w.away_pts) - (w.pred_h + w.pred_a)).abs().mean()), 3),
        n=int(len(w)),
    )


def tune(SEASONS, grid, tune_seasons, opts=None, prior_extra=None):
    """Pick the prior strength with the lowest margin error on the tuning seasons."""
    results = {}
    for lam in grid:
        _, bt, _ = run_chain(SEASONS, lam, opts=opts, prior_extra=prior_extra)
        wf = walk_forward(bt, min(tune_seasons))
        sc = score(wf, tune_seasons)
        if sc["margin"] is None:
            continue
        results[lam] = sc["margin"]
        print(f"  prior strength {lam:>3}: margin MAE {sc['margin']:.2f}")
    best = min(results, key=results.get) if results else grid[len(grid) // 2]
    return best, results


# ---------- Feature selection: keep only what improves the tuning seasons ----------

MIN_GAIN = 0.02  # points of MAE a feature must save to be kept


def _better(sc, base):
    dm = base["margin"] - sc["margin"]
    dt = base["total"] - sc["total"]
    return (dm >= MIN_GAIN or dt >= MIN_GAIN) and dm > -0.01 and dt > -0.01, round(dm, 3), round(dt, 3)


def select_features(SEASONS, lam, tune_seasons, holdout, gctx, prior_extra, available):
    """Test each feature group against the baseline on the tuning seasons only.
    Groups that help are combined; the combination is checked on the tuning
    seasons again and reported once on the untouched holdout season."""
    start = min(tune_seasons)
    log = []
    _, bt0, _ = run_chain(SEASONS, lam)
    base = score(walk_forward(bt0, start), tune_seasons)
    print(f"  baseline: margin {base['margin']}  total {base['total']}")
    chosen = dict(recency=None, priors=False, groups=[])

    # Chain-level features
    best_rec = None
    for hl in (4, 8, 16):
        _, b, _ = run_chain(SEASONS, lam, opts=dict(recency=hl))
        sc = score(walk_forward(b, start), tune_seasons)
        ok, dm, dt = _better(sc, base)
        log.append(dict(feature=f"Recency weighting (half-life {hl} weeks)", group="recency", margin=sc["margin"],
                        total=sc["total"], gain_margin=dm, gain_total=dt, kept=False, ok=ok))
        if ok and (best_rec is None or sc["margin"] < best_rec[1]):
            best_rec = (hl, sc["margin"], len(log) - 1)
    if best_rec:
        chosen["recency"] = best_rec[0]
        log[best_rec[2]]["kept"] = True
    if "priors" in available:
        _, b, _ = run_chain(SEASONS, lam, opts=dict(priors=True), prior_extra=prior_extra)
        sc = score(walk_forward(b, start), tune_seasons)
        ok, dm, dt = _better(sc, base)
        log.append(dict(feature="Preseason priors (QB/OL recruiting, transfer portal, coaching changes, defensive returning production)",
                        group="priors", margin=sc["margin"], total=sc["total"],
                        gain_margin=dm, gain_total=dt, kept=ok, ok=ok))
        chosen["priors"] = ok
    else:
        log.append(dict(feature="Preseason priors (QB/OL recruiting, transfer portal, coaching changes, defensive returning production)", group="priors", skipped=available.get("_why_priors", "data not available yet")))

    # Conversion-level features (reuse the baseline ratings)
    names = {"qb": "QB change detection", "travel": "Travel and rest", "weather": "Weather at kickoff"}
    for grp in ("qb", "travel", "weather"):
        if grp not in available:
            log.append(dict(feature=names[grp], group=grp, skipped=available.get(f"_why_{grp}", "data not available yet")))
            continue
        sc = score(walk_forward(bt0, start, [grp], gctx), tune_seasons)
        ok, dm, dt = _better(sc, base)
        log.append(dict(feature=names[grp], group=grp, margin=sc["margin"], total=sc["total"],
                        gain_margin=dm, gain_total=dt, kept=ok, ok=ok))
        if ok:
            chosen["groups"].append(grp)

    # Combined check
    opts = dict(recency=chosen["recency"], priors=chosen["priors"])
    _, btc, _ = run_chain(SEASONS, lam, opts=opts, prior_extra=prior_extra)
    wfc = walk_forward(btc, start, chosen["groups"], gctx)
    wf0 = walk_forward(bt0, start)
    comb, hold_c, hold_0 = score(wfc, tune_seasons), score(wfc, [holdout]), score(wf0, [holdout])
    if comb["margin"] is not None and comb["margin"] > base["margin"] + 0.005:
        chosen = dict(recency=None, priors=False, groups=[])  # combination didn't hold up
        comb = base
    return chosen, dict(baseline=base, combined=comb, holdout=dict(season=holdout, baseline=hold_0, selected=hold_c),
                        tests=log)


def fit_state(SEASONS, lam, current_season, features=None, gctx=None, prior_extra=None):
    """Fit everything needed for live predictions. Run every update."""
    features = features or {}
    groups = features.get("groups", [])
    opts = dict(recency=features.get("recency"), priors=features.get("priors"))
    finals, bt, info = run_chain(SEASONS, lam, opts=opts, prior_extra=prior_extra)
    seasons = sorted(SEASONS)
    wf = walk_forward(bt, seasons[0] + 2, groups, gctx)
    conv = fit_conv(side_frame(bt, gctx), groups)
    allp = predict_bt(bt, conv, groups, gctx)
    rh = allp.home_pts - allp.pred_h
    ra = allp.away_pts - allp.pred_a
    team_var = float(np.concatenate([rh, ra]).var())
    rho = float(np.clip(np.corrcoef(rh, ra)[0, 1], 0.0, 0.6))
    adv = pd.concat([SEASONS[s]["adv"] for s in seasons if len(SEASONS[s]["adv"])])
    ppd = float((adv.plays / adv.drives.replace(0, np.nan)).median()) if "drives" in adv else 5.5
    mean_pts = float(np.concatenate([allp.home_pts, allp.away_pts]).mean())
    mean_plays = float(adv.plays.mean())
    sd = sim.calibrate(team_var, rho, e=mean_pts, drives=mean_plays / ppd)
    state = dict(
        lam=lam,
        conv=conv.tolist(),
        groups=groups,
        team_var=team_var,
        rho=rho,
        sim_sd=sd,
        plays_per_drive=ppd,
        margin_sd=float((rh - ra).std()),
        total_sd=float((rh + ra).std()),
    )
    return state, finals, bt, wf, info
