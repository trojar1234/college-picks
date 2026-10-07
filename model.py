"""Season-by-season rating chain, walk-forward backtest, and the fitted pieces
that turn ratings into points. Re-running this is how the model learns:
every completed game updates the ratings, and every graded prediction feeds
the refit of the points conversion and the simulator's variance.
"""
import numpy as np
import pandas as pd

from . import data as Dt
from . import ratings as Rt
from . import sim


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


def game_feats(Rp, Rc, home, away, neutral):
    h = 0 if neutral else 1
    return dict(
        q_h=Rt.expect(Rp, home, away, h),
        p_h=Rt.expect(Rc, home, away, h),
        q_a=Rt.expect(Rp, away, home, -h),
        p_a=Rt.expect(Rc, away, home, -h),
        h=h,
    )


def run_chain(SEASONS, lam, backtest=True):
    """Walk through seasons in order. Returns final ratings per season, the
    walk-forward backtest rows, and the priors used for the latest season."""
    seasons = sorted(SEASONS)
    first = seasons[0]
    finals, pairs, bt, info = {}, [], [], {}
    for s in seasons:
        S = SEASONS[s]
        g = S["games"]
        if g.empty or S["adv"].empty:
            continue
        tiers = season_tiers(g)
        if s == first or (s - 1) not in finals:
            ppa_prior, pace_prior = {}, {}
            mu0 = float(S["adv"].ppa.mean())
            pmu0 = float(S["adv"].plays.mean())
            hfa0, phfa0 = 0.03, 0.5
            defaults = {"ppa": (-0.05, 0.05), "pace": (0.0, 0.0)}
            feats = None
        else:
            prev = finals[s - 1]
            feats = Rt.prior_features(prev, S["talent"], S["ret"])
            coefs = Rt.fit_prior_coefs(pairs)
            ppa_prior, pace_prior, defaults = Rt.make_prior(prev, feats, coefs, prev["tiers"])
            mu0, hfa0 = prev["ppa"]["mu"], prev["ppa"]["hfa"]
            pmu0, phfa0 = prev["pace"]["mu"], prev["pace"]["hfa"]
        ctx = dict(ppa_prior=ppa_prior, pace_prior=pace_prior, mu0=mu0, hfa0=hfa0,
                   pmu0=pmu0, phfa0=phfa0, defaults=defaults, tiers=tiers)
        if backtest and s != first:
            done = g[g.completed]
            for wk in sorted(done.wk.unique()):
                cut = done.loc[done.wk == wk, "start"].min()
                train = _completed_rows(S, g[(g.start < cut) & (g.wk < wk)])
                Rp, Rc = ratings_from(train, ctx, lam)
                gp = int((g.completed & (g.start < cut)).sum())
                for r in done[done.wk == wk].itertuples():
                    f = game_feats(Rp, Rc, r.home, r.away, r.neutral)
                    bt.append(dict(season=s, wk=wk, game_id=r.game_id, home=r.home, away=r.away,
                                   home_pts=r.home_pts, away_pts=r.away_pts,
                                   home_tier=tiers.get(r.home), away_tier=tiers.get(r.away),
                                   games_played=gp, **f))
        Rp, Rc = ratings_from(_completed_rows(S, g), ctx, lam)
        finals[s] = dict(ppa=Rp, pace=Rc, tiers=tiers)
        info[s] = ctx
        if feats is not None:
            pairs.append((feats, finals[s]))
    return finals, pd.DataFrame(bt), info


def ratings_from(rows, ctx, lam):
    Rp = Rt.fit(rows, "ppa", ctx["ppa_prior"], ctx["mu0"], ctx["hfa0"], lam, ctx["defaults"]["ppa"])
    Rc = Rt.fit(rows, "plays", ctx["pace_prior"], ctx["pmu0"], ctx["phfa0"], lam, ctx["defaults"]["pace"])
    return Rp, Rc


# ---------- Points conversion: ratings -> expected points ----------

def _conv_X(P, Q, h):
    P, Q, h = map(np.asarray, (P, Q, h))
    return np.column_stack([np.ones_like(P), P, P * Q, h])


def side_rows(bt):
    """Two rows per game (home offense, away offense)."""
    hs = pd.DataFrame(dict(season=bt.season, game_id=bt.game_id, P=bt.p_h, Q=bt.q_h, h=bt.h, pts=bt.home_pts))
    aw = pd.DataFrame(dict(season=bt.season, game_id=bt.game_id, P=bt.p_a, Q=bt.q_a, h=-bt.h, pts=bt.away_pts))
    return pd.concat([hs, aw], ignore_index=True)


def fit_conv(sides):
    X = _conv_X(sides.P, sides.Q, sides.h)
    y = sides.pts.to_numpy(float)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def points(conv, P, Q, h):
    return _conv_X(P, Q, h) @ conv


def predict_bt(bt, conv):
    out = bt.copy()
    out["pred_h"] = points(conv, bt.p_h, bt.q_h, bt.h)
    out["pred_a"] = points(conv, bt.p_a, bt.q_a, -bt.h)
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
