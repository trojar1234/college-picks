"""Opponent-adjusted team ratings with Bayesian priors.

Each game gives two observations (one per offense). For each we model

    y = mu + off[offense] + def[defense] + hfa * h

where y is EPA per play (efficiency) or plays per game (pace), and h is +1
home / -1 away / 0 neutral. Ratings are solved by ridge regression that pulls
each team toward its preseason prior. `lam` is the prior's weight in
"games' worth" of evidence: early in the season the prior dominates, by
midseason the data does. Re-solving after every week is the weekly update.
"""
import numpy as np
import pandas as pd


def fit(rows, ycol, prior, mu0, hfa0, lam, default=(0.0, 0.0), fcs=frozenset(), g0=(0.0, 0.0)):
    """rows: DataFrame with off, deff, h, and ycol. prior: {team: (off, def)}.

    FCS teams share a group offset (g_off, g_def) estimated from all their games,
    and their own ratings shrink toward that group. Games involving an FCS team
    don't inform home field (FCS teams almost always play away, which would
    otherwise inflate home field)."""
    teams = sorted(set(prior) | (set(rows["off"]) | set(rows["deff"]) if len(rows) else set()))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    k = 4 + 2 * n  # mu, off..., def..., hfa, g_off, g_def
    HI, GO, GD = 1 + 2 * n, 2 + 2 * n, 3 + 2 * n
    A = np.zeros((k, k))
    b = np.zeros(k)
    if len(rows):
        m = len(rows)
        o = rows["off"].map(idx).to_numpy()
        d = rows["deff"].map(idx).to_numpy()
        of = rows["off"].isin(fcs).to_numpy()
        df = rows["deff"].isin(fcs).to_numpy()
        h = np.where(of | df, 0.0, rows["h"].to_numpy(dtype=float))
        X = np.zeros((m, k))
        r = np.arange(m)
        X[:, 0] = 1
        X[r, 1 + o] = 1
        X[r, 1 + n + d] = 1
        X[:, HI] = h
        X[:, GO] = of
        X[:, GD] = df
        y = rows[ycol].to_numpy(dtype=float)
        w = rows["w"].to_numpy(dtype=float) if "w" in rows else np.ones(m)
        A += (X * w[:, None]).T @ X
        b += (X * w[:, None]).T @ y
    b0 = np.zeros(k)
    b0[0], b0[HI], b0[GO], b0[GD] = mu0, hfa0, g0[0], g0[1]
    for t, i in idx.items():
        po, pd_ = (0.0, 0.0) if t in fcs else prior.get(t, default)
        b0[1 + i] = po
        b0[1 + n + i] = pd_
    pen = np.full(k, float(lam))
    pen[0] = 0.5    # weak pull on the league average
    pen[HI] = 20.0  # home field is stable; pull it toward its prior
    pen[GO] = pen[GD] = 2.0
    A[np.diag_indices(k)] += pen
    b += pen * b0
    sol = np.linalg.solve(A, b)
    return dict(
        mu=float(sol[0]),
        hfa=float(sol[HI]),
        g=(float(sol[GO]), float(sol[GD])),
        fcs=set(fcs),
        off={t: float(sol[1 + i]) for t, i in idx.items()},
        deff={t: float(sol[1 + n + i]) for t, i in idx.items()},
        default=default,
    )


def _side(R, team, j):
    book = R["off"] if j == 0 else R["deff"]
    if team in book:
        return book[team] + (R["g"][j] if team in R["fcs"] else 0.0)
    return R["g"][j]  # unknown team: almost always a small FCS school


def expect(R, off_team, def_team, h):
    return R["mu"] + _side(R, off_team, 0) + _side(R, def_team, 1) + R["hfa"] * h


def absolute(R):
    """Ratings with the FCS group offset folded in (used to build next year's priors)."""
    return dict(mu=R["mu"], hfa=R["hfa"],
                off={t: _side(R, t, 0) for t in R["off"]},
                deff={t: _side(R, t, 1) for t in R["deff"]})


# ---------- Preseason priors ----------

def _z(d):
    # A talent score of 0 means "not rated" (the service academies aren't in the recruiting
    # composites), not "worst roster in the country". Treat it as unknown.
    d = {k: x for k, x in d.items() if x == x and x > 50}
    v = np.array([x for x in d.values() if x == x])
    if len(v) < 5:
        return {}
    m, s = v.mean(), v.std() or 1.0
    return {k: (x - m) / s for k, x in d.items() if x == x}


# rating name -> column in game rows. "pts" is opponent-adjusted points scored, which captures
# what per-play efficiency misses: special teams, red-zone finishing, field position, turnovers.
METRICS = {"ppa": "ppa", "sr": "sr", "pace": "plays", "pts": "pts",
           # from raw play-by-play (when downloaded): non-garbage points per drive,
           # points per scoring opportunity (finishing), and average starting field position
           "ppd": "ppd", "fin": "fin", "fpos": "fpos"}
CORE = ("ppa", "sr", "pace", "pts")  # every team has these; the rest may be missing
NEUTRAL = ("pace", "fpos")            # tendencies: unknown teams default to league average


def _ret(ret, t, rmean):
    v = ret.get(t, rmean)
    return (v if v == v else rmean) - rmean


PX = ("qbd", "qbl", "dret", "oret")  # player-level preseason features (see cfb_players.py)


def prior_features(prev, talent, ret, hist=None, players=None):
    """Feature rows per team for building next season's prior (prev = absolute ratings):
    last season's ratings, the average of the last four seasons (program history, as
    FPI and SP+ use), returning production, the roster talent composite, and optional
    player-level information (QB quality, returning production counting transfers)."""
    tz = _z(talent)
    rv = [x for x in ret.values() if x == x]
    rmean = float(np.mean(rv)) if rv else 0.5
    teams = set(prev["ppa"]["off"]) | set(tz)
    rows = []
    for t in teams:
        r = dict(team=t, has_prev=t in prev["ppa"]["off"], ret=_ret(ret, t, rmean),
                 tal=tz.get(t, 0.0), tal_na=float(t not in tz))
        for m in METRICS:
            for side in ("off", "deff"):
                r[f"p_{m}_{side}"] = prev[m][side].get(t, np.nan)
                r[f"h_{m}_{side}"] = (hist or {}).get(m, {}).get(side, {}).get(t, np.nan)
        px = (players or {}).get(t, {})
        for k in PX:
            r[k] = px.get(k, np.nan)
        rows.append(r)
    df = pd.DataFrame(rows)
    for k in PX:
        if k not in df:
            df[k] = np.nan
        df[k] = df[k].astype(float)
    for k in ("qbl", "dret", "oret"):  # center so "average" is 0
        df[k] = df[k] - df[k].mean()
    return df


def _cols(m, side, use=()):
    base = f"p_{m}_{side}"
    if m in NEUTRAL:
        return [base]
    cols = [base, f"{base}_ret", "tal"] if side == "off" else [base, "tal"]
    cols += ["tal_na", f"h_{m}_{side}"]
    if side == "off" and "qb" in use:
        cols += ["qbd", "qbl"]
    if side == "off" and "off" in use:
        cols += [f"{base}_oret", "oret"]
    if side == "deff" and "def" in use:
        cols += [f"{base}_dret", "dret"]
    return cols


def _design(df, m, side, use=()):
    df = df.copy()
    base = f"p_{m}_{side}"
    df[f"{base}_ret"] = (df[base] * df["ret"]).fillna(0.0)
    h = f"h_{m}_{side}"
    if h not in df:
        df[h] = np.nan
    df[h] = df[h].fillna(df[base]).fillna(0.0)
    if "tal_na" not in df:
        df["tal_na"] = 0.0
    for k in PX:
        df[k] = df[k].astype(float).fillna(0.0) if k in df else 0.0
    df[base] = df[base].fillna(0.0)
    df[f"{base}_dret"] = df[base] * df["dret"]
    df[f"{base}_oret"] = df[base] * df["oret"]
    return np.column_stack([np.ones(len(df)), df[_cols(m, side, use)].to_numpy(dtype=float)])


def _usable(f):
    need = [f"p_{m}_{s}" for m in CORE for s in ("off", "deff")]
    return f[f.has_prev & f[need].notna().all(axis=1)]


def fit_prior_coefs(pairs, use=()):
    """pairs: list of (features for season s, absolute final ratings of season s)."""
    coefs = {}
    for m in METRICS:
        for side in ("off", "deff"):
            Xs, ys = [], []
            for feats, final in pairs:
                f = _usable(feats).copy()
                f["y"] = f.team.map(final[m][side])
                f = f.dropna(subset=["y"])
                if len(f):
                    Xs.append(_design(f, m, side, use))
                    ys.append(f["y"].to_numpy())
            if not Xs or sum(len(y) for y in ys) < 150:
                coefs[(m, side)] = None
                continue
            X, y = np.vstack(Xs), np.concatenate(ys)
            reg = np.eye(X.shape[1])
            reg[0, 0] = 0
            coefs[(m, side)] = np.linalg.solve(X.T @ X + reg, X.T @ y)
    return coefs


def make_prior(prev, feats, coefs, tiers_prev, use=()):
    """Returns ({metric: {team: (off, def)}}, {metric: default (off, def)})."""
    def tier_mean(m, side):
        vals = [v for t, v in prev[m][side].items() if tiers_prev.get(t) == "FCS" and v == v]
        return float(np.mean(vals)) if vals else 0.0
    f = feats.copy()
    has = _usable(f)
    priors, defaults = {}, {}
    for m in METRICS:
        for side in ("off", "deff"):
            c = coefs.get((m, side))
            X = _design(has, m, side, use)
            pred = X @ c if c is not None else X[:, 1] * (0.5 if m in NEUTRAL else 0.6)
            col = f"pr_{m}_{side}"
            f[col] = np.nan
            f.loc[has.index, col] = pred
            f.loc[f[col].isna(), col] = 0.0 if m in NEUTRAL else tier_mean(m, side)
        priors[m] = {r.team: (float(getattr(r, f"pr_{m}_off")), float(getattr(r, f"pr_{m}_deff"))) for r in f.itertuples()}
        defaults[m] = (0.0, 0.0) if m in NEUTRAL else (tier_mean(m, "off"), tier_mean(m, "deff"))
    return priors, defaults
