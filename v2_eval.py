"""Model 2.0 evaluation: forecasting accuracy first, the market only as a benchmark.

Seasons: 2017-2022 tune (every design choice was made looking only at these), 2023-2024 validate,
2025 test (looked at once, after the design was fixed), 2026 research (it has shaped too many v1
decisions to count as untouched). Prospective games from the day Model 2.0 goes live are the real test.
"""
import math

import numpy as np
import pandas as pd

SPLITS = {"tune": list(range(2017, 2023)), "validate": [2023, 2024], "test": [2025], "2026": [2026]}


def _phi(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2)))


def crps_normal(y, mu, sd):
    z = (y - mu) / sd
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    return sd * (z * (2 * _phi(z) - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def block_boot(diff, weeks, reps=2000, seed=3):
    """95% range of a mean difference, resampling whole weeks."""
    d = pd.DataFrame(dict(d=diff, w=weeks)).dropna()
    g = d.groupby("w").d.agg(["sum", "count"]).to_numpy()
    if len(g) < 5:
        return None
    idx = np.random.default_rng(seed).integers(0, len(g), (reps, len(g)))
    bs = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return [round(float(np.quantile(bs, 0.025)), 3), round(float(np.quantile(bs, 0.975)), 3)]


def hist_sd(Gm, col, kind):
    """Error size for a benchmark, measured on earlier seasons only (walk-forward)."""
    out = pd.Series(np.nan, index=Gm.index)
    for s in sorted(Gm.season.unique()):
        prev = Gm[(Gm.season < s) & Gm[col].notna() & Gm[kind].notna()]
        if len(prev) > 300:
            out[Gm.season == s] = float(np.sqrt(((prev[kind] - prev[col]) ** 2).mean()))
    return out


def accuracy(Gm, models, split, fbs_only=True):
    d = Gm[Gm.season.isin(SPLITS[split]) & Gm.margin.notna()]
    if fbs_only:
        d = d[d.mismatch == 0]
    d = d.dropna(subset=[f"m_{m}" for m in models if f"m_{m}" in d])
    rows = []
    for m in models:
        if f"m_{m}" not in d:
            continue
        em, et = d.margin - d[f"m_{m}"], d.total - d[f"t_{m}"]
        r = dict(model=m, n=int(len(d)), margin_mae=round(float(em.abs().mean()), 3),
                 margin_rmse=round(float(np.sqrt((em ** 2).mean())), 3), margin_bias=round(float(em.mean()), 2))
        if d[f"t_{m}"].notna().all() and m not in ("elo",):
            r.update(total_mae=round(float(et.abs().mean()), 3), total_rmse=round(float(np.sqrt((et ** 2).mean())), 3),
                     total_bias=round(float(et.mean()), 2))
        if f"sd_m_{m}" in d and d[f"sd_m_{m}"].notna().all():
            p = np.clip(_phi(d[f"m_{m}"] / d[f"sd_m_{m}"]), 1e-4, 1 - 1e-4)
            y = (d.margin > 0).astype(float)
            r.update(brier=round(float(((p - y) ** 2).mean()), 4),
                     logloss=round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4),
                     crps_margin=round(float(crps_normal(d.margin, d[f"m_{m}"], d[f"sd_m_{m}"]).mean()), 3))
            lo, hi = d[f"m_{m}"] - 1.2816 * d[f"sd_m_{m}"], d[f"m_{m}"] + 1.2816 * d[f"sd_m_{m}"]
            r["cover80_margin"] = round(float(((d.margin >= lo) & (d.margin <= hi)).mean()), 3)
        if f"sd_t_{m}" in d and d[f"sd_t_{m}"].notna().all() and m not in ("elo",):
            lo, hi = d[f"t_{m}"] - 1.2816 * d[f"sd_t_{m}"], d[f"t_{m}"] + 1.2816 * d[f"sd_t_{m}"]
            r["cover80_total"] = round(float(((d.total >= lo) & (d.total <= hi)).mean()), 3)
            r["crps_total"] = round(float(crps_normal(d.total, d[f"t_{m}"], d[f"sd_t_{m}"]).mean()), 3)
        rows.append(r)
    return rows


def paired(Gm, a, b, split, kind="margin"):
    """Mean absolute error of b minus a (positive = a better), with a week-block bootstrap range."""
    d = Gm[Gm.season.isin(SPLITS[split]) & Gm.margin.notna() & (Gm.mismatch == 0)]
    k = kind[0]
    d = d.dropna(subset=[f"{k}_{a}", f"{k}_{b}"])
    if d.empty:
        return None
    diff = (d[kind] - d[f"{k}_{b}"]).abs() - (d[kind] - d[f"{k}_{a}"]).abs()
    return dict(a=a, b=b, kind=kind, n=int(len(d)), gain=round(float(diff.mean()), 3),
                range95=block_boot(diff.to_numpy(), (d.season * 100 + d.wk).to_numpy()))


def market_info(Gm, col, split):
    """How much of the model's disagreement with a line shows up in results:
    (result - line) = w * (model - line). w = 0: the line already knows everything the model knows."""
    out = {}
    d = Gm[Gm.season.isin(SPLITS[split]) & Gm.margin.notna() & (Gm.mismatch == 0)]
    for kind, line in (("margin", "line_m_open"), ("margin", "line_m_close"), ("total", "line_t_open"), ("total", "line_t_close")):
        k = kind[0]
        x = d.dropna(subset=[line, f"{k}_{col}"])
        if len(x) < 100:
            continue
        g = x[f"{k}_{col}"] - x[line]
        r = x[kind] - x[line]
        w = float((g * r).sum() / (g * g).sum())
        resid = r - w * g
        se = float(np.sqrt((resid ** 2).sum() / max(len(x) - 1, 1) / (g * g).sum()))
        out[line] = dict(w=round(w, 3), se=round(se, 3), n=int(len(x)),
                         model_mae=round(float((x[kind] - x[f"{k}_{col}"]).abs().mean()), 3),
                         line_mae=round(float((x[kind] - x[line]).abs().mean()), 3))
    return out


def picks_record(Gm, col, split, thresholds=(0, 3, 5, 7)):
    """Research only, never bets: how the side the model prefers did against each line, by gap size."""
    d = Gm[Gm.season.isin(SPLITS[split]) & Gm.margin.notna() & (Gm.mismatch == 0)]
    out = []
    for kind, line in (("margin", "line_m_open"), ("margin", "line_m_close"), ("total", "line_t_open"), ("total", "line_t_close")):
        k = kind[0]
        x = d.dropna(subset=[line, f"{k}_{col}"])
        gap = x[f"{k}_{col}"] - x[line]
        res = np.sign(gap) * (x[kind] - x[line])
        for t in thresholds:
            sel = gap.abs() >= max(t, 0.5)
            w, l = int((res[sel] > 0).sum()), int((res[sel] < 0).sum())
            row = dict(kind=kind, line=line, gap=t, w=w, l=l,
                       pct=round(w / (w + l), 3) if w + l else None, units=round(w * 100 / 110 - l, 1))
            if kind == "total":
                row["over_share"] = round(float((gap[sel] > 0).mean()), 2) if sel.any() else None
            else:
                row["home_share"] = round(float((gap[sel] > 0).mean()), 2) if sel.any() else None
            out.append(row)
    return out


def confidence_test(Gm, split):
    """Does model disagreement help predict how big the miss will be? Compare out-of-sample log scores of
    the confidence model with and without it, and show average misses by agreement level."""
    d = Gm[Gm.season.isin(SPLITS[split]) & Gm.margin.notna() & (Gm.mismatch == 0)].dropna(subset=["sd_m_dis", "sd_m_nodis"])
    if d.empty:
        return None
    def ls(sd):
        e = d.margin - d.m_ens
        return float((-0.5 * np.log(2 * np.pi * sd ** 2) - 0.5 * e ** 2 / sd ** 2).mean())
    by = d.groupby("agree").apply(lambda x: pd.Series(dict(n=len(x), mae=round(float((x.margin - x.m_ens).abs().mean()), 2))),
                                  include_groups=False)
    return dict(logscore_with=round(ls(d.sd_m_dis), 4), logscore_without=round(ls(d.sd_m_nodis), 4),
                by_agreement={k: dict(n=int(v.n), mae=float(v.mae)) for k, v in by.iterrows()})


def calibration(Gm, split, bins=(0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    d = Gm[Gm.season.isin(SPLITS[split]) & Gm.margin.notna() & (Gm.mismatch == 0)].dropna(subset=["sd_m"])
    p = _phi(d.m_ens / d.sd_m)
    y = (d.margin > 0).astype(float)
    c = pd.cut(p, list(bins), include_lowest=True)
    g = pd.DataFrame(dict(p=p, y=y, c=c)).groupby("c", observed=True).agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    return [dict(bin=str(i), n=int(r.n), predicted=round(float(r.pred), 3), actual=round(float(r.actual), 3)) for i, r in g.iterrows()]
