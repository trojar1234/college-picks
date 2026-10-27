"""Season-by-season rating chain, walk-forward backtest, and the fitted pieces
that turn ratings into points. Re-running this is how the model learns:
every completed game updates the ratings, and every graded prediction feeds
the refit of the points conversion and the simulator's variance.

Core model (always on):
  - opponent-adjusted efficiency ratings (EPA/play, success rate, plays per game)
  - opponent-adjusted scoring-margin rating (captures special teams, finishing,
    field position and turnovers that per-play efficiency misses)
  - FCS teams rated individually from all their games, shrunk toward the FCS average

Optional (kept only if the improvement on the tuning seasons is real and it does not cost bets: see _better):
  pbp - play-by-play: non-garbage points per drive, finishing drives, field position
"""
import numpy as np
import pandas as pd

import cfb_config as C
import cfb_data as Dt
import cfb_ratings as Rt
import cfb_sim as sim

BASE_GROUPS = ["score"]  # always in the conversion
OPTIONAL = {
    "totals": "Separate totals model: game-level fit with matchup type (Power 4 vs Power 4, FCS), recalibrated on past out-of-sample totals",
    "pbp": "Play-by-play: non-garbage points per drive, finishing drives, field position",
}


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
        if g.empty:
            continue
        no_stats = S["adv"].empty  # preseason: the schedule is out but no games played yet
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
        if backtest and s != first and not no_stats:
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
        if feats is not None and not no_stats:  # prior-only ratings must not train the priors
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


def predict_bt(bt, conv, groups=(), tm=None):
    """Expected points for each side. With a totals model (tm), the margin comes from the
    team-points conversion and the total from the separate game-level totals model."""
    out = bt.copy()
    pred = conv_X(side_frame(bt), groups) @ np.asarray(conv)
    n = len(bt)
    out["pred_h"] = pred[:n]
    out["pred_a"] = pred[n:]
    if tm:
        out = apply_totals(out, tm)
    return out


# ---------- Totals model ----------
# Backtests showed the team-points conversion's totals were too spread out (actual totals moved
# only ~0.8 points per point of predicted total). A game-level regression, then a straight-line
# recalibration fit on earlier seasons' out-of-sample totals, fixes that.

TOT_DECAY = 0.6


def total_X(bt):
    P = bt.p_h + bt.p_a
    PQ = bt.p_h * bt.q_h + bt.p_a * bt.q_a
    PS = bt.p_h * bt.s_h + bt.p_a * bt.s_a
    R = bt.r_h + bt.r_a
    if "home_tier" in bt and "away_tier" in bt:
        m4 = ((bt.home_tier == "P4") & (bt.away_tier == "P4")).astype(float)
        mf = ((bt.home_tier == "FCS") | (bt.away_tier == "FCS")).astype(float)
    else:
        m4 = mf = pd.Series(0.0, index=bt.index)
    return np.column_stack([np.ones(len(bt)), P, PQ, PS, R, m4, mf]).astype(float)


def fit_totals(train):
    X = total_X(train)
    y = (train.home_pts + train.away_pts).to_numpy(float)
    w = np.sqrt(TOT_DECAY ** (train.season.max() - train.season.to_numpy()))
    return np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]


def fit_recal(pred_total, actual_total):
    """actual = a + b * predicted, from out-of-sample predictions."""
    b, a = np.polyfit(np.asarray(pred_total, float), np.asarray(actual_total, float), 1)
    return [float(a), float(b)]


def apply_totals(out, tm):
    t = total_X(out) @ np.asarray(tm["beta"])
    if tm.get("recal"):
        a, b = tm["recal"]
        t = a + b * t
    m = (out.pred_h - out.pred_a).to_numpy()
    out = out.copy()
    out["pred_h"], out["pred_a"] = (t + m) / 2, (t - m) / 2
    return out


def walk_forward(bt, eval_from, groups=()):
    """Score each season with a points conversion fit only on earlier seasons."""
    parts, oos = [], []  # oos: earlier seasons' out-of-sample raw totals, for the recalibration
    errs = []  # earlier seasons' out-of-sample (margin, total) errors, for honest error sizes
    use_tot = "totals" in groups
    for s in sorted(bt.season.unique()):
        train = bt[bt.season < s]
        if len(train) < 300:
            continue
        conv = fit_conv(side_frame(train), groups)
        tm = None
        if use_tot:
            tm = dict(beta=fit_totals(train).tolist())
            cur = bt[bt.season == s]
            raw = total_X(cur) @ np.asarray(tm["beta"])
            prev = pd.concat(oos) if oos else None
            if prev is not None and prev.season.nunique() >= 2:
                tm["recal"] = fit_recal(prev.raw, prev.actual)
            oos.append(pd.DataFrame(dict(season=s, raw=raw, actual=(cur.home_pts + cur.away_pts).to_numpy(float))))
        if s < eval_from:
            continue
        p = predict_bt(bt[bt.season == s], conv, groups, tm)
        # The live projections correct the service academies (see academy_adjust); the backtest applies the
        # same correction, estimated only from earlier seasons' misses, so its record matches the live system.
        p["pred_h_raw"], p["pred_a_raw"] = p.pred_h.to_numpy(), p.pred_a.to_numpy()
        if parts:
            p = apply_academy(p, academy_adjust(pd.concat(parts)))
        if errs:  # error size measured on earlier seasons' genuinely out-of-sample predictions
            e = pd.concat(errs[-3:])
            sd_m, sd_t = float(e.m.std()), float(e.t.std())
        else:  # first season scored: nothing out of sample yet, so use the training fit
            tp = predict_bt(train, conv, groups, tm)
            sd_m = float(((tp.home_pts - tp.away_pts) - (tp.pred_h - tp.pred_a)).std())
            sd_t = float(((tp.home_pts + tp.away_pts) - (tp.pred_h + tp.pred_a)).std())
        p["sd_m"], p["sd_t"] = sd_m, sd_t
        parts.append(p)
        done = p[p.home_pts.notna()]
        errs.append(pd.DataFrame(dict(m=(done.home_pts - done.away_pts) - (done.pred_h - done.pred_a),
                                      t=(done.home_pts + done.away_pts) - (done.pred_h + done.pred_a))))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fit_total_model(bt):
    """Totals model for live use: fit on all seasons; recalibration from every season's
    walk-forward (out-of-sample) raw totals."""
    beta = fit_totals(bt)
    raws, acts = [], []
    for s in sorted(bt.season.unique()):
        train = bt[bt.season < s]
        if len(train) < 300:
            continue
        cur = bt[bt.season == s]
        cur = cur[cur.home_pts.notna()]
        raws.append(total_X(cur) @ fit_totals(train))
        acts.append((cur.home_pts + cur.away_pts).to_numpy(float))
    tm = dict(beta=beta.tolist())
    if len(raws) >= 2:
        tm["recal"] = fit_recal(np.concatenate(raws), np.concatenate(acts))
    return tm


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

# Keep rule. A fixed 0.05-point bar was replaced by a test of whether the gain is real: on the tuning
# seasons, the 95% range of the paired improvement (resampling whole weeks, since games in a week share
# conditions) must sit above zero, the gain must be at least MIN_GAIN (smaller is too small to matter),
# and the other measure must not get worse by more than 0.01. In the walk-forward, 24 pure-noise
# inputs never improved accuracy at all, so a gain whose range clears zero is not chance.
MIN_GAIN = 0.01
KEEP_RULE = 2   # 1 = fixed 0.05-point bar (before Oct 23, 2026); 2 = real-gain test, bet check, confirmation season
CONFIRM_TOL = 0.02  # points of extra error allowed on the confirmation season (it's one season, so noisier)
BET_TOL = 1.0   # minimum slack (units) on the bet check; the actual slack is the larger of this and the chance wobble


def _paired(wf_new, wf_ref, seasons, col, reps=2000, seed=5):
    """Gain in mean absolute error (positive = new is better) and its 95% range, same games."""
    a = wf_ref[wf_ref.season.isin(seasons) & wf_ref.home_pts.notna()]
    b = wf_new[wf_new.season.isin(seasons) & wf_new.home_pts.notna()]
    m = a[["game_id", "season", "wk", "home_pts", "away_pts", "pred_h", "pred_a"]].merge(
        b[["game_id", "pred_h", "pred_a"]], on="game_id", suffixes=("", "_n"))
    if m.empty:
        return 0.0, (0.0, 0.0)
    if col == "margin":
        act, p0, p1 = m.home_pts - m.away_pts, m.pred_h - m.pred_a, m.pred_h_n - m.pred_a_n
    else:
        act, p0, p1 = m.home_pts + m.away_pts, m.pred_h + m.pred_a, m.pred_h_n + m.pred_a_n
    d = (act - p0).abs() - (act - p1).abs()
    g = d.groupby([m.season, m.wk]).agg(["sum", "count"]).to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(reps, len(g)))
    bs = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return float(d.mean()), (float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))


def _open_bets(wf, lines, seasons):
    """Every 7+ point disagreement with the opening line (spreads and totals, FCS games left out) on
    the given seasons, with its result in units (1 unit at -110). Grading only: lines never enter the model."""
    if lines is None or len(lines) == 0 or "cfbd_spread_open" not in lines:
        return None
    d = wf[wf.season.isin(seasons) & wf.home_pts.notna() & wf.home_tier.isin(["P4", "G5"])
           & wf.away_tier.isin(["P4", "G5"])].merge(
        lines[["game_id", "cfbd_spread_open", "cfbd_total_open"]], on="game_id")
    pm, am = d.pred_h - d.pred_a, d.home_pts - d.away_pts
    pt, at = d.pred_h + d.pred_a, d.home_pts + d.away_pts
    out = []
    for kind, e, v, thr in (("spread", d.cfbd_spread_open + pm, am + d.cfbd_spread_open, C.SPREAD_EDGE),
                            ("total", pt - d.cfbd_total_open, at - d.cfbd_total_open, C.TOTAL_EDGE)):
        r = np.sign(e) * v
        k = e.notna() & (e.abs() >= thr)
        out.append(pd.DataFrame(dict(game_id=d.game_id[k], season=d.season[k], wk=d.wk[k], kind=kind,
                                     u=np.where(r[k] > 0, 100 / 110, np.where(r[k] < 0, -1.0, 0.0)))))
    return pd.concat(out, ignore_index=True)


def open_units(wf, lines, seasons):
    b = _open_bets(wf, lines, seasons)
    if b is None:
        return None
    return dict(units=round(float(b.u.sum()), 1), w=int((b.u > 0).sum()), l=int((b.u < 0).sum()))


def _bet_change(wf_new, wf_ref, lines, seasons, reps=2000, seed=7):
    """Change in units (new minus reference) on 7+ point bets against opening lines, and how much that
    change would wobble by chance (spread of a week-resampled bootstrap). Most bets are the same in
    both versions, so only the bets that differ count."""
    a, b = _open_bets(wf_new, lines, seasons), _open_bets(wf_ref, lines, seasons)
    if a is None or b is None:
        return None, None
    k = ["game_id", "season", "wk", "kind"]
    m = a.merge(b, on=k, how="outer", suffixes=("_n", "_r")).fillna({"u_n": 0.0, "u_r": 0.0})
    if m.empty:
        return 0.0, 0.0
    g = (m.u_n - m.u_r).groupby([m.season, m.wk]).sum().to_numpy()
    tot = g[np.random.default_rng(seed).integers(0, len(g), size=(reps, len(g)))].sum(1)
    return float(g.sum()), float(tot.std())


def _better(wf_new, wf_ref, seasons, lines=None):
    """Keep a feature only if (1) the accuracy gain is real and (2) it doesn't cost bets. (2) exists
    because extra totals terms tested in Oct 2026 (squared terms) cut totals error for real, yet on the
    unseen 2025 season won fewer units on 7+ point bets against opening lines (+13 to +21 vs +27):
    being closer on every game is not the same as beating the line."""
    dm, rm = _paired(wf_new, wf_ref, seasons, "margin")
    dt, rt = _paired(wf_new, wf_ref, seasons, "total")
    real = (dm >= MIN_GAIN and rm[0] > 0) or (dt >= MIN_GAIN and rt[0] > 0)
    bn, br = open_units(wf_new, lines, seasons), open_units(wf_ref, lines, seasons)
    du, sd = _bet_change(wf_new, wf_ref, lines, seasons)
    bets_ok = du is None or du >= -max(BET_TOL, sd)   # rejected only if bets drop by more than chance alone would explain
    if bn is not None and du is not None:
        bn = dict(bn, change=round(du, 1), noise=round(sd, 1))
    return (real and dm > -0.01 and dt > -0.01 and bets_ok, round(dm, 3), round(dt, 3),
            [round(x, 3) for x in rm], [round(x, 3) for x in rt], bn, br)


PLAYER_OPTIONS = {
    "players": (("qb", "def"), "Player-level preseason data: QB quality (transfers included) and defensive returning production counting transfers"),
    "players_off": (("qb", "def", "off"), "Same, plus offensive returning production counting transfers (player EPA)"),
}


def _confirm(wf_new, wf_ref, season, lines):
    """Second check on the newest complete season, which played no part in the first test: a feature
    that passed must not make that season clearly worse (error up by more than CONFIRM_TOL, or bet
    units down by more than chance). A CFBD Elo input tested in Oct 2026 passed on 2017-2024, then
    cut 2025's units on 7+ point bets against opening lines from +26 to +12."""
    dm, _ = _paired(wf_new, wf_ref, [season], "margin")
    dt, _ = _paired(wf_new, wf_ref, [season], "total")
    du, sd = _bet_change(wf_new, wf_ref, lines, [season])
    bad = dm < -CONFIRM_TOL or dt < -CONFIRM_TOL or (du is not None and du < -max(BET_TOL, sd))
    return not bad, dict(season=season, gain_margin=round(dm, 3), gain_total=round(dt, 3),
                         bets_change=None if du is None else round(du, 1), noise=None if sd is None else round(sd, 1))


def select_features(SEASONS, lam, tune_seasons, holdout, available, players=None):
    """Test each optional input against the core model on the tuning seasons only,
    and report the result once on the untouched holdout season."""
    start = min(tune_seasons)
    _, bt0, _ = run_chain(SEASONS, lam)
    wf0 = walk_forward(bt0, start, BASE_GROUPS)
    base = score(wf0, tune_seasons)
    print(f"  core model: margin {base['margin']}  total {base['total']}")
    log, use, bt, ref, ref_wf = [], (), bt0, base, wf0
    lines = pd.concat([S["lines"] for S in SEASONS.values() if len(S.get("lines", []))], ignore_index=True) \
        if any(len(S.get("lines", [])) for S in SEASONS.values()) else None

    # 1) player-level preseason data (changes the ratings, so each option re-runs the chain)
    best = None
    for key, (u, name) in PLAYER_OPTIONS.items():
        if key not in available:
            log.append(dict(feature=name, group=key, skipped=available.get(f"_why_{key}", "data not available yet")))
            continue
        _, b, _ = run_chain(SEASONS, lam, players=players, use=u)
        wfb = walk_forward(b, start, BASE_GROUPS)
        sc = score(wfb, tune_seasons)
        ok, dm, dt, rm, rt, bn, br = _better(wfb, wf0, tune_seasons, lines)
        conf_ok, conf = _confirm(wfb, wf0, holdout, lines)
        log.append(dict(feature=name, group=key, margin=sc["margin"], total=sc["total"],
                        gain_margin=dm, gain_total=dt, range_margin=rm, range_total=rt,
                        bets=bn, bets_ref=br, confirm=conf, confirm_ok=conf_ok, kept=False, ok=ok and conf_ok))
        ok = ok and conf_ok
        if ok and (best is None or sc["margin"] < best[1]["margin"]):
            best = (u, sc, b, len(log) - 1, wfb)
    if best:
        use, ref, bt, ref_wf = best[0], best[1], best[2], best[4]
        log[best[3]]["kept"] = True

    # 2) optional conversion inputs, tested on top of whatever was kept above
    chosen = list(BASE_GROUPS)
    for grp, name in OPTIONAL.items():
        if grp not in available:
            log.append(dict(feature=name, group=grp, skipped=available.get(f"_why_{grp}", "data not available yet")))
            continue
        wfg = walk_forward(bt, start, BASE_GROUPS + [grp])
        sc = score(wfg, tune_seasons)
        ok, dm, dt, rm, rt, bn, br = _better(wfg, ref_wf, tune_seasons, lines)
        conf_ok, conf = _confirm(wfg, ref_wf, holdout, lines)
        ok = ok and conf_ok
        log.append(dict(feature=name, group=grp, margin=sc["margin"], total=sc["total"],
                        gain_margin=dm, gain_total=dt, range_margin=rm, range_total=rt,
                        bets=bn, bets_ref=br, confirm=conf, confirm_ok=conf_ok, kept=ok, ok=ok))
        if ok:
            chosen.append(grp)
    wfc = walk_forward(bt, start, chosen)
    return dict(groups=chosen, players=list(use)), dict(
        baseline=base, combined=score(wfc, tune_seasons), bets_baseline=open_units(wf0, lines, tune_seasons),
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
    done = bt[bt.home_pts.notna()]
    tm = fit_total_model(done) if "totals" in groups else None
    allp = predict_bt(bt, conv, groups, tm)
    rh = allp.home_pts - allp.pred_h
    ra = allp.away_pts - allp.pred_a
    # Error sizes for probabilities: from genuinely out-of-sample predictions (the last three seasons
    # of the walk-forward) rather than the fit's errors on its own training games.
    o = wf[wf.home_pts.notna() & (wf.season >= current_season - 3)] if len(wf) else wf
    if len(o) >= 500:
        rh = o.home_pts - o.pred_h
        ra = o.away_pts - o.pred_a
    rh, ra = rh.dropna(), ra.dropna()
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
        total_model=tm,
        team_var=team_var,
        rho=rho,
        sim_sd=sd,
        plays_per_drive=ppd,
        margin_sd=float((rh - ra).std()),
        total_sd=float((rh + ra).std()),
    )
    state["academy"] = academy_adjust(wf)
    state["rating_tau2"] = rating_uncertainty(wf, lam)
    try:  # monitoring only: never let it stop a run
        state["home_field"] = home_field_check(bt, finals, conv, groups, SEASONS, current_season)
    except Exception as e:
        print("Home-field check skipped:", e)
    return state, finals, bt, wf, info


def home_field_check(bt, finals, conv, groups, SEASONS, season):
    """Monitoring: the home-field edge the model applies this season (projected margin at home minus the
    same game on a neutral field) next to the edge this season's results show (least squares on
    FBS-vs-FBS margins, capped at 35 points so blowouts don't dominate)."""
    R = finals.get(season)
    out = {}
    sched = SEASONS[season]["games"]
    sched = sched[~sched.neutral.astype(bool) & (sched.home_cls == "fbs") & (sched.away_cls == "fbs")]
    if len(sched) and R:  # every home game this season, with today's ratings
        rows = [(dict(season=season, game_id=0, **game_feats(R, r.home, r.away, False)),
                 dict(season=season, game_id=0, **game_feats(R, r.home, r.away, True))) for r in sched.itertuples()]
        p1 = predict_bt(pd.DataFrame([a for a, _ in rows]), conv, groups)
        p0 = predict_bt(pd.DataFrame([b for _, b in rows]), conv, groups)
        out["model"] = round(float(((p1.pred_h - p1.pred_a) - (p0.pred_h - p0.pred_a)).mean()), 2)
    g = SEASONS[season]["games"]
    g = g[g.completed & (g.home_cls == "fbs") & (g.away_cls == "fbs")]
    if len(g) >= 100:
        teams = sorted(set(g.home) | set(g.away))
        ix = {t: i for i, t in enumerate(teams)}
        X = np.zeros((len(g), len(teams) + 1))
        for r, (h, a, n) in enumerate(zip(g.home, g.away, g.neutral)):
            X[r, ix[h]], X[r, ix[a]], X[r, -1] = 1, -1, 0 if n else 1
        y = np.clip((g.home_pts - g.away_pts).to_numpy(float), -35, 35)
        A = X.T @ X + np.diag([0.01] * len(teams) + [0.0])
        b = np.linalg.solve(A, X.T @ y)
        resid = y - X @ b
        dof = max(1, len(y) - len(teams) - 1)
        se = float(np.sqrt(resid @ resid / dof * np.linalg.inv(A)[-1, -1]))
        out.update(realized=round(float(b[-1]), 2), se=round(se, 2), n=int(len(g)))
    return out


def apply_academy(p, ac):
    """Add the service-academy correction to a frame of predictions (pred_h, pred_a)."""
    if not ac or not ac.get("n"):
        return p
    p = p.copy()
    side = p.home.isin(ac["teams"]).astype(int) - p.away.isin(ac["teams"]).astype(int)
    m, t = ac["margin"] * side, ac["total"] * (side != 0)
    p["pred_h"] = p.pred_h + (m + t) / 2
    p["pred_a"] = p.pred_a + (t - m) / 2
    return p


def rating_uncertainty(wf, lam):
    """How uncertain a team's rating is, from how much bigger the out-of-sample misses are early in the
    season: miss variance = base + tau2 * (1/(n_home+lam) + 1/(n_away+lam)) with n = games played.
    Returns tau2 (one team's rating variance is about tau2 / (games + lam))."""
    d = wf[wf.home_pts.notna()] if len(wf) else wf
    if len(d) < 1000:
        return None
    em = (d.home_pts - d.away_pts) - (d.pred_h - d.pred_a)
    v = em.groupby(d.wk).var()
    n = (v.index.to_numpy(float) - 1).clip(0)
    keep = v.index <= 15
    X = np.column_stack([np.ones(keep.sum()), 2 / (n[keep] + lam)])
    b = np.linalg.lstsq(X, v[keep].to_numpy(), rcond=None)[0]
    return round(float(max(b[1], 0.0)), 1)


def academy_adjust(wf):
    """How much the service academies beat their projections, from out-of-sample walk-forward
    errors (before any academy correction), shrunk toward zero. Returns {} when switched off or with
    too few games."""
    teams = list(getattr(C, "ACADEMIES", []) or [])
    if not teams or wf is None or not len(wf):
        return {}
    d = wf[wf.home_pts.notna()]
    if "pred_h_raw" in d:
        d = d.assign(pred_h=d.pred_h_raw.fillna(d.pred_h), pred_a=d.pred_a_raw.fillna(d.pred_a))
    side = d.home.isin(teams).astype(int) - d.away.isin(teams).astype(int)
    g = d[side != 0]
    if len(g) < 30:
        return {}
    k = float(getattr(C, "ACADEMY_SHRINK", 50))
    em = ((g.home_pts - g.away_pts) - (g.pred_h - g.pred_a)) * side[side != 0]
    et = (g.home_pts + g.away_pts) - (g.pred_h + g.pred_a)
    return dict(teams=teams, margin=round(float(em.sum() / (len(g) + k)), 2),
                total=round(float(et.sum() / (len(g) + k)), 2), n=int(len(g)))
