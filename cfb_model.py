"""Season-by-season rating chain, walk-forward backtest, and the fitted pieces
that turn ratings into points. Re-running this is how the model learns:
every completed game updates the ratings, and every graded prediction feeds
the refit of the points conversion and the simulator's variance.

Core model (always on):
  - opponent-adjusted efficiency ratings (EPA/play, success rate, plays per game)
  - opponent-adjusted scoring-margin rating (captures special teams, finishing,
    field position and turnovers that per-play efficiency misses)
  - FCS teams rated individually from all their games, shrunk toward the FCS average

Optional (kept only if it improves the tuning seasons by at least MIN_GAIN):
  pbp - play-by-play: non-garbage points per drive, finishing drives, field position
"""
import numpy as np
import pandas as pd

import cfb_data as Dt
import cfb_ratings as Rt
import cfb_sim as sim

BASE_GROUPS = ["score"]  # always in the conversion
OPTIONAL = {"pbp": "Play-by-play: non-garbage points per drive, finishing drives, field position"}


def season_tiers(games):
    t = {}
    for r in games.itertuples():
        t[r.home] = Dt.team_tier(r.home, r.home_conf, r.home_cls, r.season)
        t[r.away] = Dt.team_tier(r.away, r.away_conf, r.away_cls, r.season)
    return t


def _rows_before(S, cut=None):
    """Every team-game played before the cutoff (FCS-vs-FCS games included, which is
    how FCS teams get individual ratings)."""
    a = S["adv"]
    if cut is None or not len(a):
        return a
    return a[a.start < cut]


def game_feats(R, home, away, neutral):
    h = 0 if neutral else 1
    out = dict(h=h)
    for key, m in FEAT_KEYS:
        out[f"{key}_h"] = Rt.expect(R[m], home, away, h)
        out[f"{key}_a"] = Rt.expect(R[m], away, home, -h)
    return out


FEAT_KEYS = (("q", "ppa"), ("s", "sr"), ("p", "pace"), ("r", "pts"), ("d", "ppd"), ("fn", "fin"), ("fp", "fpos"))
FIRST_G = {"ppa": (-0.15, 0.15), "sr": (-0.05, 0.05), "pace": (0.0, 0.0), "pts": (-14.0, 14.0),
           "ppd": (-1.0, 1.0), "fin": (-1.0, 1.0), "fpos": (5.0, -5.0)}
FIRST_HFA = {"ppa": 0.03, "sr": 0.01, "pace": 0.5, "pts": 1.5, "ppd": 0.15, "fin": 0.1}


def _mu0(adv, col):
    v = adv[col].mean() if col in adv else np.nan
    return float(v) if v == v else 0.0


def _history(finals, s, years=4):
    """Average of each team's final ratings over the previous `years` seasons."""
    past = [finals[y]["abs"] for y in range(s - years, s) if y in finals]
    out = {}
    for m in Rt.METRICS:
        out[m] = {}
        for side in ("off", "deff"):
            acc = {}
            for p in past:
                for t, v in p[m][side].items():
                    acc.setdefault(t, []).append(v)
            out[m][side] = {t: float(np.mean(v)) for t, v in acc.items()}
    return out


def run_chain(SEASONS, lam, backtest=True, players=None, use=()):
    """Walk through seasons in order. Returns final ratings per season, the
    walk-forward backtest rows, and the context (priors) used per season.
    players: {season: {team: {...}}} player-level preseason features; use: which of
    them to include ("qb", "def", "off")."""
    use = tuple(use or ())
    seasons = sorted(SEASONS)
    first = seasons[0]
    finals, pairs, bt, info = {}, [], [], {}
    for s in seasons:
        S = SEASONS[s]
        g = S["games"]
        if g.empty or S["adv"].empty:
            continue
        tiers = season_tiers(S.get("games_all", g))
        ctx = dict(tiers=tiers, fcs=frozenset(t for t, v in tiers.items() if v == "FCS"), m={})
        feats = None
        if s == first or (s - 1) not in finals:
            for m, col in Rt.METRICS.items():
                ctx["m"][m] = dict(prior={}, mu0=_mu0(S["adv"], col), hfa0=FIRST_HFA.get(m, 0.0),
                                   default=(0.0, 0.0), g0=FIRST_G.get(m, (0.0, 0.0)))
        else:
            prev = finals[s - 1]
            feats = Rt.prior_features(prev["abs"], S["talent"], S["ret"], _history(finals, s),
                                      (players or {}).get(s) if use else None)
            priors, defaults = Rt.make_prior(prev["abs"], feats, Rt.fit_prior_coefs(pairs, use), prev["tiers"], use)
            for m in Rt.METRICS:
                ctx["m"][m] = dict(prior=priors[m], mu0=prev[m]["mu"], hfa0=prev[m]["hfa"],
                                   default=defaults[m], g0=prev[m]["g"])
        if backtest and s != first:
            done = g[g.completed]
            for wk in sorted(done.wk.unique()):
                cut = done.loc[done.wk == wk, "start"].min()
                R = ratings_from(_rows_before(S, cut), ctx, lam)
                gp = int((g.completed & (g.start < cut)).sum())
                for r in done[done.wk == wk].itertuples():
                    bt.append(dict(season=s, wk=wk, game_id=r.game_id, home=r.home, away=r.away,
                                   home_pts=r.home_pts, away_pts=r.away_pts,
                                   home_tier=tiers.get(r.home), away_tier=tiers.get(r.away),
                                   games_played=gp, **game_feats(R, r.home, r.away, r.neutral)))
        R = ratings_from(_rows_before(S), ctx, lam)
        finals[s] = dict(R, tiers=tiers, abs={m: Rt.absolute(R[m]) for m in Rt.METRICS})
        info[s] = ctx
        if feats is not None:
            pairs.append((feats, finals[s]["abs"]))
    return finals, pd.DataFrame(bt), info


def ratings_from(rows, ctx, lam):
    out = {}
    for m, col in Rt.METRICS.items():
        c = ctx["m"][m]
        r = rows[rows[col].notna()] if col in rows else rows.iloc[0:0]
        out[m] = Rt.fit(r, col, c["prior"], c["mu0"], c["hfa0"], lam, c["default"], ctx["fcs"], c["g0"])
    return out


# ---------- Points conversion: ratings -> expected points ----------

GROUP_COLS = {
    "score": ["R"],             # scoring-margin rating
    "pbp": ["DP", "FN", "FP"],  # non-garbage pts/drive x pace, finishing, field position
}
CONV_DECAY = 0.6  # weight per season of age: scoring environments drift (rule changes, pace)


def side_frame(bt):
    """Two rows per game (home offense, away offense) with every column the conversion may use."""
    names = {"p": "P", "q": "Q", "s": "S", "r": "R", "d": "D", "fn": "FN", "fp": "FP"}

    def side(sfx, sign):
        out = dict(season=bt.season.values, game_id=bt.game_id.values, h=sign * bt.h.values,
                   pts=bt[f"{'home' if sfx == 'h' else 'away'}_pts"].values if "home_pts" in bt else np.nan,
                   side=sfx)
        for k, n in names.items():
            col = f"{k}_{sfx}"
            out[n] = bt[col].values if col in bt else np.nan
        return pd.DataFrame(out)

    df = pd.concat([side("h", 1), side("a", -1)], ignore_index=True)
    df["DP"] = df.D * df.P / 10
    for c in ("R", "DP", "FN", "FP"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def conv_X(df, groups=()):
    P, Q, S, h = (df[c].to_numpy(dtype=float) for c in ("P", "Q", "S", "h"))
    cols = [np.ones_like(P), P, P * Q, P * S, h]
    for g in groups:
        for c in GROUP_COLS.get(g, []):
            cols.append(df[c].to_numpy(dtype=float))
    return np.column_stack(cols)


def fit_conv(df, groups=(), decay=None):
    """Least squares from ratings to points, weighting recent seasons more."""
    decay = CONV_DECAY if decay is None else decay
    X = conv_X(df, groups)
    y = df.pts.to_numpy(float)
    w = np.sqrt(decay ** (df.season.max() - df.season.to_numpy()))
    return np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]


def predict_bt(bt, conv, groups=()):
    out = bt.copy()
    pred = conv_X(side_frame(bt), groups) @ np.asarray(conv)
    n = len(bt)
    out["pred_h"] = pred[:n]
    out["pred_a"] = pred[n:]
    return out


def walk_forward(bt, eval_from, groups=()):
    """Score each season with a points conversion fit only on earlier seasons."""
    parts = []
    for s in sorted(bt.season.unique()):
        if s < eval_from:
            continue
        train = bt[bt.season < s]
        if len(train) < 300:
            continue
        conv = fit_conv(side_frame(train), groups)
        tp = predict_bt(train, conv, groups)
        sd_m = float(((tp.home_pts - tp.away_pts) - (tp.pred_h - tp.pred_a)).std())
        sd_t = float(((tp.home_pts + tp.away_pts) - (tp.pred_h + tp.pred_a)).std())
        p = predict_bt(bt[bt.season == s], conv, groups)
        p["sd_m"], p["sd_t"] = sd_m, sd_t
        parts.append(p)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def score(wf, seasons):
    """Margin and total mean absolute error over the given seasons."""
    w = wf[wf.season.isin(seasons)] if len(wf) else wf
    if w.empty:
        return dict(margin=None, total=None, n=0)
    return dict(
        margin=round(float(((w.home_pts - w.away_pts) - (w.pred_h - w.pred_a)).abs().mean()), 3),
        total=round(float(((w.home_pts + w.away_pts) - (w.pred_h + w.pred_a)).abs().mean()), 3),
        n=int(len(w)),
    )


def tune(SEASONS, grid, tune_seasons, groups=None, players=None, use=()):
    """Pick the prior strength with the lowest margin error on the tuning seasons."""
    groups = groups or BASE_GROUPS
    results = {}
    for lam in grid:
        _, bt, _ = run_chain(SEASONS, lam, players=players, use=use)
        sc = score(walk_forward(bt, min(tune_seasons), groups), tune_seasons)
        if sc["margin"] is None:
            continue
        results[lam] = sc["margin"]
        print(f"  prior strength {lam:>3}: margin MAE {sc['margin']:.2f}")
    best = min(results, key=results.get) if results else grid[len(grid) // 2]
    return best, results


# ---------- Feature selection: keep only what clearly improves the tuning seasons ----------

MIN_GAIN = 0.05  # points of MAE a feature must save to be kept (smaller gains were noise)


def _better(sc, base):
    dm = base["margin"] - sc["margin"]
    dt = base["total"] - sc["total"]
    return (dm >= MIN_GAIN or dt >= MIN_GAIN) and dm > -0.01 and dt > -0.01, round(dm, 3), round(dt, 3)


PLAYER_OPTIONS = {
    "players": (("qb", "def"), "Player-level preseason data: QB quality (transfers included) and defensive returning production counting transfers"),
    "players_off": (("qb", "def", "off"), "Same, plus offensive returning production counting transfers (player EPA)"),
}


def select_features(SEASONS, lam, tune_seasons, holdout, available, players=None):
    """Test each optional input against the core model on the tuning seasons only,
    and report the result once on the untouched holdout season."""
    start = min(tune_seasons)
    _, bt0, _ = run_chain(SEASONS, lam)
    wf0 = walk_forward(bt0, start, BASE_GROUPS)
    base = score(wf0, tune_seasons)
    print(f"  core model: margin {base['margin']}  total {base['total']}")
    log, use, bt, ref = [], (), bt0, base

    # 1) player-level preseason data (changes the ratings, so each option re-runs the chain)
    best = None
    for key, (u, name) in PLAYER_OPTIONS.items():
        if key not in available:
            log.append(dict(feature=name, group=key, skipped=available.get(f"_why_{key}", "data not available yet")))
            continue
        _, b, _ = run_chain(SEASONS, lam, players=players, use=u)
        sc = score(walk_forward(b, start, BASE_GROUPS), tune_seasons)
        ok, dm, dt = _better(sc, base)
        log.append(dict(feature=name, group=key, margin=sc["margin"], total=sc["total"],
                        gain_margin=dm, gain_total=dt, kept=False, ok=ok))
        if ok and (best is None or sc["margin"] < best[1]["margin"]):
            best = (u, sc, b, len(log) - 1)
    if best:
        use, ref, bt = best[0], best[1], best[2]
        log[best[3]]["kept"] = True

    # 2) optional conversion inputs, tested on top of whatever was kept above
    chosen = list(BASE_GROUPS)
    for grp, name in OPTIONAL.items():
        if grp not in available:
            log.append(dict(feature=name, group=grp, skipped=available.get(f"_why_{grp}", "data not available yet")))
            continue
        sc = score(walk_forward(bt, start, BASE_GROUPS + [grp]), tune_seasons)
        ok, dm, dt = _better(sc, ref)
        log.append(dict(feature=name, group=grp, margin=sc["margin"], total=sc["total"],
                        gain_margin=dm, gain_total=dt, kept=ok, ok=ok))
        if ok:
            chosen.append(grp)
    wfc = walk_forward(bt, start, chosen)
    return dict(groups=chosen, players=list(use)), dict(
        baseline=base, combined=score(wfc, tune_seasons),
        holdout=dict(season=holdout, baseline=score(wf0, [holdout]), selected=score(wfc, [holdout])),
        tests=log)


def fit_state(SEASONS, lam, current_season, features=None, players=None):
    """Fit everything needed for live predictions. Run every update."""
    groups = (features or {}).get("groups") or BASE_GROUPS
    use = tuple((features or {}).get("players") or ())
    finals, bt, info = run_chain(SEASONS, lam, players=players, use=use)
    seasons = sorted(SEASONS)
    wf = walk_forward(bt, seasons[0] + 2, groups)
    conv = fit_conv(side_frame(bt), groups)
    allp = predict_bt(bt, conv, groups)
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
