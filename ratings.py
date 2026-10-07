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


def fit(rows, ycol, prior, mu0, hfa0, lam, default=(0.0, 0.0)):
    """rows: DataFrame with off, deff, h, and ycol. prior: {team: (off, def)}."""
    teams = sorted(set(prior) | set(rows["off"]) | set(rows["deff"])) if len(rows) else sorted(prior)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    k = 2 + 2 * n  # mu, off..., def..., hfa
    A = np.zeros((k, k))
    b = np.zeros(k)
    if len(rows):
        o = rows["off"].map(idx).to_numpy()
        d = rows["deff"].map(idx).to_numpy() + n
        h = rows["h"].to_numpy(dtype=float)
        y = rows[ycol].to_numpy(dtype=float)
        cols = [np.zeros(len(rows), int), o + 1, d + 1]
        # Build normal equations X'X and X'y without a dense design matrix.
        for ci in cols:
            np.add.at(b, ci, y)
            for cj in cols:
                np.add.at(A, (ci, cj), 1.0)
        hi = k - 1
        b[hi] += (h * y).sum()
        A[hi, hi] += (h * h).sum()
        for ci in cols:
            np.add.at(A, (ci, hi), h)
            np.add.at(A, (hi, ci), h)
    b0 = np.zeros(k)
    b0[0] = mu0
    b0[-1] = hfa0
    for t, i in idx.items():
        po, pd_ = prior.get(t, default)
        b0[1 + i] = po
        b0[1 + n + i] = pd_
    pen = np.full(k, float(lam))
    pen[0] = 0.5   # weak pull on the league average
    pen[-1] = 20.0  # home field is fairly stable; pull it toward its prior
    A[np.diag_indices(k)] += pen
    b += pen * b0
    sol = np.linalg.solve(A, b)
    return dict(
        mu=float(sol[0]),
        hfa=float(sol[-1]),
        off={t: float(sol[1 + i]) for t, i in idx.items()},
        deff={t: float(sol[1 + n + i]) for t, i in idx.items()},
        default=default,
    )


def expect(R, off_team, def_team, h):
    o = R["off"].get(off_team, R["default"][0])
    d = R["deff"].get(def_team, R["default"][1])
    return R["mu"] + o + d + R["hfa"] * h


# ---------- Preseason priors ----------

def _z(d):
    v = np.array([x for x in d.values() if x == x])
    if len(v) < 5:
        return {}
    m, s = v.mean(), v.std() or 1.0
    return {k: (x - m) / s for k, x in d.items() if x == x}


def prior_features(prev, talent, ret):
    """Feature rows per team for building next season's prior."""
    tz = _z(talent)
    rv = [x for x in ret.values() if x == x]
    rmean = float(np.mean(rv)) if rv else 0.5
    teams = set(prev["ppa"]["off"]) | set(tz)
    rows = []
    for t in teams:
        rows.append(
            dict(
                team=t,
                has_prev=t in prev["ppa"]["off"],
                p_off=prev["ppa"]["off"].get(t, np.nan),
                p_def=prev["ppa"]["deff"].get(t, np.nan),
                p_poff=prev["pace"]["off"].get(t, np.nan),
                p_pdef=prev["pace"]["deff"].get(t, np.nan),
                ret=(ret.get(t, rmean) if ret.get(t, rmean) == ret.get(t, rmean) else rmean) - rmean,
                tal=tz.get(t, -1.0),
            )
        )
    return pd.DataFrame(rows)


FEATS = {
    "off": ["p_off", "p_off_ret", "tal"],
    "def": ["p_def", "tal"],
    "poff": ["p_poff"],
    "pdef": ["p_pdef"],
}
FALLBACK = {"off": [0.6, 0.0, 0.0], "def": [0.6, 0.0], "poff": [0.5], "pdef": [0.5]}


def _design(df, comp):
    df = df.copy()
    df["p_off_ret"] = df["p_off"] * df["ret"]
    X = df[FEATS[comp]].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(df)), X])


def fit_prior_coefs(pairs):
    """pairs: list of (features_df for season s, final ratings of season s). Ridge OLS per component."""
    coefs = {}
    tgt_key = {"off": ("ppa", "off"), "def": ("ppa", "deff"), "poff": ("pace", "off"), "pdef": ("pace", "deff")}
    for comp, (m, side) in tgt_key.items():
        Xs, ys = [], []
        for feats, final in pairs:
            f = feats[feats.has_prev].copy()
            f["y"] = f.team.map(final[m][side])
            f = f.dropna(subset=["y", "p_off", "p_def", "p_poff", "p_pdef"])
            if len(f):
                Xs.append(_design(f, comp))
                ys.append(f["y"].to_numpy())
        if not Xs or sum(len(y) for y in ys) < 150:
            coefs[comp] = None
            continue
        X, y = np.vstack(Xs), np.concatenate(ys)
        reg = np.eye(X.shape[1]) * 1.0
        reg[0, 0] = 0
        coefs[comp] = np.linalg.solve(X.T @ X + reg, X.T @ y)
    return coefs


def make_prior(prev, feats, coefs, tiers_prev):
    """Returns ({team: (off, def)} for ppa, same for pace, defaults per tier)."""
    out_ppa, out_pace = {}, {}
    f = feats.copy()
    # Teams with no previous-season data start at their tier's average.
    def tier_mean(m, side, tier):
        vals = [v for t, v in prev[m][side].items() if tiers_prev.get(t) == tier]
        return float(np.mean(vals)) if vals else 0.0
    for comp, (m, side) in {"off": ("ppa", "off"), "def": ("ppa", "deff"), "poff": ("pace", "off"), "pdef": ("pace", "deff")}.items():
        has = f[f.has_prev & f[["p_off", "p_def", "p_poff", "p_pdef"]].notna().all(axis=1)]
        c = coefs.get(comp)
        if c is None:
            pred = _design(has, comp)[:, 1:] @ np.array(FALLBACK[comp])
        else:
            pred = _design(has, comp) @ c
        f[comp] = np.nan
        f.loc[has.index, comp] = pred
        f.loc[f[comp].isna(), comp] = tier_mean(m, side, "FCS")
    for _, r in f.iterrows():
        out_ppa[r.team] = (float(r["off"]), float(r["def"]))
        out_pace[r.team] = (float(r["poff"]), float(r["pdef"]))
    defaults = {
        "ppa": (tier_mean("ppa", "off", "FCS"), tier_mean("ppa", "deff", "FCS")),
        "pace": (0.0, 0.0),
    }
    return out_ppa, out_pace, defaults
