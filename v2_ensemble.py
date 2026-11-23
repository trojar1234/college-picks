"""Model 2.0 ensemble and confidence.

Weights: for each season, non-negative weights on the five models (plus an intercept) are fitted on
earlier seasons' out-of-sample predictions only, separately for margin and total. No market numbers.

Confidence: each game gets its own error size. A regression on earlier seasons' squared misses uses
the rating uncertainty from the Bayesian filter, how early in the season it is and the projected size
of the game. How much the five new models disagree was tested as a fourth input and did not help
(out-of-sample log score -4.1906 with it, -4.1907 without, on 2017-2022; identical on 2023-2024), so it
is shown on the site as information but does not change any probability. (v2_eval.confidence_test
keeps measuring it.)
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from v2_models import MODELS


def fit_weights(df, target, cols, ridge=0.0):
    d = df.dropna(subset=cols + [target])
    X = d[cols].to_numpy(float)
    y = d[target].to_numpy(float)
    xm, ym = X.mean(0), y.mean()
    if ridge:
        # shrink toward equal weights by augmenting with pseudo-observations
        k = len(cols)
        Xa = np.vstack([X - xm, np.sqrt(ridge) * np.eye(k)])
        ya = np.r_[y - ym, np.sqrt(ridge) * np.full(k, 1.0 / k)]
    else:
        Xa, ya = X - xm, y - ym
    w, _ = nnls(Xa, ya)
    a = ym - xm @ w
    return dict(w=dict(zip(cols, map(float, w))), a=float(a), n=int(len(d)))


def apply_weights(df, fit):
    out = np.full(len(df), fit["a"])
    for c, w in fit["w"].items():
        out = out + w * df[c].to_numpy(float)
    return out


# The ensemble's members: the five new models plus the current model (v1). Adding v1 cut the margin miss on
# 2017-2022 from 12.71 to 12.67 and on 2023-2024 from 12.47 to 12.43, and totals from 13.11 to 13.05 and
# 12.90 to 12.82, because its errors differ from the new models'. Context terms (both teams Power 4, late
# season, neutral site) were tested and changed nothing, so the stack has none.
MEMBERS = MODELS + ["v1"]
EXTRA = []


def _with_extra(df):
    df = df.copy()
    for c in EXTRA:
        df[c + "_pos"] = df[c].astype(float)
        df[c + "_neg"] = -df[c].astype(float)
    return df


def walk_ensemble(Gm, first_fit=2017, min_seasons=2, ridge=2000.0):
    """Gm: one row per game with actual margin/total and m_<model>, t_<model>. Adds m_ens, t_ens and
    the weights used each season."""
    Gm = Gm.copy()
    Gm["m_ens"] = np.nan
    Gm["t_ens"] = np.nan
    used = {}
    seasons = sorted(Gm.season.unique())
    for s in seasons:
        tr = Gm[(Gm.season < s) & (Gm.season >= first_fit) & Gm.margin.notna()]
        if tr.season.nunique() < min_seasons:
            continue
        cur = Gm.season == s
        ex = [f"{c}_{s}" for c in EXTRA for s in ("pos", "neg")]
        trx, curx = _with_extra(tr), _with_extra(Gm[cur])
        base = [m for m in MODELS if f"m_{m}" in Gm]
        full = [m for m in MEMBERS if f"m_{m}" in Gm and tr[f"m_{m}"].notna().any()]
        # with every member where all have a projection (v1 has none for FCS-vs-FCS games), else the new five
        fm = fit_weights(trx, "margin", [f"m_{m}" for m in full], ridge)
        ft = fit_weights(trx, "total", [f"t_{m}" for m in full] + ex, ridge)
        fm0 = fit_weights(trx, "margin", [f"m_{m}" for m in base], ridge)
        ft0 = fit_weights(trx, "total", [f"t_{m}" for m in base] + ex, ridge)
        has = curx[[f"m_{m}" for m in full]].notna().all(axis=1).to_numpy()
        Gm.loc[cur, "m_ens"] = np.where(has, apply_weights(curx.fillna({f"m_{m}": 0.0 for m in full}), fm),
                                        apply_weights(curx, fm0))
        Gm.loc[cur, "t_ens"] = np.where(has, apply_weights(curx.fillna({f"t_{m}": 0.0 for m in full}), ft),
                                        apply_weights(curx, ft0))
        used[int(s)] = dict(margin=fm, total=ft, margin_without_v1=fm0, total_without_v1=ft0)
    return Gm, used


def conf_X(df, kind, with_disagree=True):
    p = df[f"{kind[0]}_ens"].to_numpy(float)
    spread = df[[f"{kind[0]}_{m}" for m in MODELS]].std(axis=1).to_numpy(float)
    rv = df["pts_vm" if kind == "margin" else "pts_vt"].to_numpy(float)
    cols = [np.ones(len(df)), rv, 1.0 / (df.wk.clip(upper=20).to_numpy(float) + 1.0),
            np.abs(p) if kind == "margin" else p, df.mismatch.to_numpy(float)]
    if with_disagree:
        cols.append(spread ** 2)
    return np.column_stack(cols)


def fit_conf(df, kind, with_disagree=True):
    d = df.dropna(subset=[f"{kind[0]}_ens", kind, "pts_vm"])
    e2 = (d[kind] - d[f"{kind[0]}_ens"]).to_numpy(float) ** 2
    X = conf_X(d, kind, with_disagree)
    b = np.linalg.lstsq(X, e2, rcond=None)[0]
    return b, float(np.mean(e2))


def pred_sd(df, kind, fit, with_disagree=True):
    b, base = fit
    v = conf_X(df, kind, with_disagree) @ b
    return np.sqrt(np.clip(v, 0.4 * base, 3.0 * base))


def walk_conf(Gm, min_seasons=2, with_disagree=True):
    Gm = Gm.copy()
    for kind in ("margin", "total"):
        col = f"sd_{kind[0]}" + ("" if with_disagree else "_nodis")
        Gm[col] = np.nan
        for s in sorted(Gm.season.unique()):
            tr = Gm[(Gm.season < s) & Gm[f"{kind[0]}_ens"].notna() & Gm[kind].notna()]
            if tr.season.nunique() < min_seasons:
                continue
            cur = (Gm.season == s) & Gm[f"{kind[0]}_ens"].notna()
            f = fit_conf(tr, kind, with_disagree)
            Gm.loc[cur, col] = pred_sd(Gm[cur], kind, f, with_disagree)
    return Gm


def agreement(df):
    """Plain-language agreement label from the spread of the five models' margins."""
    sd = df[[f"m_{m}" for m in MODELS]].std(axis=1)
    return pd.cut(sd, [-1, 1.5, 3.0, 1e9], labels=["strong", "moderate", "weak"]).astype(str)
