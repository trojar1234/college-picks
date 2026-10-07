"""Season-by-season rating chain, walk-forward backtest, and the fitted pieces
that turn ratings into points. Re-running this is how the model learns:
every completed game updates the ratings, and every graded prediction feeds
the refit of the points conversion and the simulator's variance.
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


def _completed_rows(S, games_subset):
    ids = set(games_subset.loc[games_subset.completed, "game_id"])
    a = S["adv"]
    return a[a.game_id.isin(ids)] if len(a) else a


def game_feats(R, home, away, neutral):
    h = 0 if neutral else 1
    out = dict(h=h)
    for key, m in (("q", "ppa"), ("s", "sr"), ("p", "pace")):
        out[f"{key}_h"] = Rt.expect(R[m], home, away, h)
        out[f"{key}_a"] = Rt.expect(R[m], away, home, -h)
    return out


FIRST_G = {"ppa": (-0.15, 0.15), "sr": (-0.05, 0.05), "pace": (0.0, 0.0)}
FIRST_HFA = {"ppa": 0.03, "sr": 0.01, "pace": 0.5}


def run_chain(SEASONS, lam, backtest=True):
    """Walk through seasons in order. Returns final ratings per season, the
    walk-forward backtest rows, and the context (priors) used per season."""
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
            feats = Rt.prior_features(prev["abs"], S["talent"], S["ret"])
            priors, defaults = Rt.make_prior(prev["abs"], feats, Rt.fit_prior_coefs(pairs), prev["tiers"])
            for m in Rt.METRICS:
                ctx["m"][m] = dict(prior=priors[m], mu0=prev[m]["mu"], hfa0=prev[m]["hfa"],
                                   default=defaults[m], g0=prev[m]["g"])
        if backtest and s != first:
            done = g[g.completed]
            for wk in sorted(done.wk.unique()):
                cut = done.loc[done.wk == wk, "start"].min()
                R = ratings_from(_completed_rows(S, g[(g.start < cut) & (g.wk < wk)]), ctx, lam)
                gp = int((g.completed & (g.start < cut)).sum())
                for r in done[done.wk == wk].itertuples():
                    bt.append(dict(season=s, wk=wk, game_id=r.game_id, home=r.home, away=r.away,
                                   home_pts=r.home_pts, away_pts=r.away_pts,
                                   home_tier=tiers.get(r.home), away_tier=tiers.get(r.away),
                                   games_played=gp, **game_feats(R, r.home, r.away, r.neutral)))
        R = ratings_from(_completed_rows(S, g), ctx, lam)
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


# ---------- Points conversion: ratings -> expected points ----------

def _conv_X(P, Q, S, h):
    P, Q, S, h = (np.asarray(x, dtype=float) for x in (P, Q, S, h))
    return np.column_stack([np.ones_like(P), P, P * Q, P * S, h])


def side_rows(bt):
    """Two rows per game (home offense, away offense)."""
    hs = pd.DataFrame(dict(season=bt.season, game_id=bt.game_id, P=bt.p_h, Q=bt.q_h, S=bt.s_h, h=bt.h, pts=bt.home_pts))
    aw = pd.DataFrame(dict(season=bt.season, game_id=bt.game_id, P=bt.p_a, Q=bt.q_a, S=bt.s_a, h=-bt.h, pts=bt.away_pts))
    return pd.concat([hs, aw], ignore_index=True)


CONV_DECAY = 0.6  # weight per season of age: scoring environments drift (rule changes, pace)


def fit_conv(sides, decay=None):
    """Least squares from ratings to points, weighting recent seasons more."""
    decay = CONV_DECAY if decay is None else decay
    X = _conv_X(sides.P, sides.Q, sides.S, sides.h)
    y = sides.pts.to_numpy(float)
    w = np.sqrt(decay ** (sides.season.max() - sides.season.to_numpy()))
    return np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]


def points(conv, P, Q, S, h):
    return _conv_X(P, Q, S, h) @ conv


def predict_bt(bt, conv):
    out = bt.copy()
    out["pred_h"] = points(conv, bt.p_h, bt.q_h, bt.s_h, bt.h)
    out["pred_a"] = points(conv, bt.p_a, bt.q_a, bt.s_a, -bt.h)
    return out


def walk_forward(bt, eval_from):
    """Score each season with a points conversion fit only on earlier seasons."""
    parts = []
    for s in sorted(bt.season.unique()):
        if s < eval_from:
            continue
        train = bt[bt.season < s]
        if len(train) < 300:
            continue
        conv = fit_conv(side_rows(train))
        tp = predict_bt(train, conv)
        sd_m = float(((tp.home_pts - tp.away_pts) - (tp.pred_h - tp.pred_a)).std())
        sd_t = float(((tp.home_pts + tp.away_pts) - (tp.pred_h + tp.pred_a)).std())
        p = predict_bt(bt[bt.season == s], conv)
        p["sd_m"], p["sd_t"] = sd_m, sd_t
        parts.append(p)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def tune(SEASONS, grid, current_season):
    """Pick the prior strength that gives the lowest backtest margin error."""
    seasons = sorted(SEASONS)
    eval_from = seasons[0] + 2
    results = {}
    for lam in grid:
        _, bt, _ = run_chain(SEASONS, lam)
        wf = walk_forward(bt, eval_from)
        wf = wf[wf.season < current_season]
        if wf.empty:
            continue
        mae = float(((wf.home_pts - wf.away_pts) - (wf.pred_h - wf.pred_a)).abs().mean())
        results[lam] = mae
        print(f"  prior strength {lam:>3}: margin MAE {mae:.2f}")
    best = min(results, key=results.get) if results else grid[len(grid) // 2]
    return best, results


def fit_state(SEASONS, lam, current_season):
    """Fit everything needed for live predictions. Run every update."""
    finals, bt, info = run_chain(SEASONS, lam)
    seasons = sorted(SEASONS)
    wf = walk_forward(bt, seasons[0] + 2)
    conv = fit_conv(side_rows(bt))
    allp = predict_bt(bt, conv)
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
        team_var=team_var,
        rho=rho,
        sim_sd=sd,
        plays_per_drive=ppd,
        margin_sd=float((rh - ra).std()),
        total_sd=float((rh + ra).std()),
    )
    return state, finals, bt, wf, info
