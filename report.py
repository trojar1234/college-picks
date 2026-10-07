"""Report card: backtest results, live results, calibration, and error diagnostics."""
import math

import numpy as np
import pandas as pd

from . import config as C


def _phi(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2)))


def _rec(results):
    w = int((results == "win").sum())
    l = int((results == "loss").sum())
    p = int((results == "push").sum())
    return dict(w=w, l=l, p=p, pct=round(w / (w + l), 3) if w + l else None)


def _matchup(r):
    t = sorted([r.home_tier or "FCS", r.away_tier or "FCS"])
    if "FCS" in t:
        return "FCS involved"
    return {("P4", "P4"): "P4 vs P4", ("G5", "P4"): "P4 vs G5", ("G5", "G5"): "G5 vs G5"}[tuple(t)]


def backtest(wf, SEASONS):
    if wf.empty:
        return {}
    ln = pd.concat([SEASONS[s]["lines"] for s in SEASONS if len(SEASONS[s]["lines"])], ignore_index=True)
    d = wf.merge(ln, on="game_id", how="left")
    d["pm"] = d.pred_h - d.pred_a
    d["am"] = d.home_pts - d.away_pts
    d["pt"] = d.pred_h + d.pred_a
    d["at"] = d.home_pts + d.away_pts
    d["model_spread"] = -d.pm
    d["matchup"] = [_matchup(r) for r in d.itertuples()]

    # Against the closing line (evaluation only)
    has = d.dropna(subset=["cfbd_spread"]).copy()
    e = has.cfbd_spread - has.model_spread
    has["side"] = np.where(e > 0, "home", "away")
    v = np.where(has.side == "home", has.am + has.cfbd_spread, -(has.am + has.cfbd_spread))
    has["res"] = np.where(v == 0, "push", np.where(v > 0, "win", "loss"))
    has["flag"] = e.abs() >= C.SPREAD_EDGE
    has["p_home_cover"] = _phi((has.pm + has.cfbd_spread) / has.sd_m)

    ht = d.dropna(subset=["cfbd_total"]).copy()
    et = ht.pt - ht.cfbd_total
    vt = np.where(et > 0, ht["at"] - ht.cfbd_total, ht.cfbd_total - ht["at"])
    ht["res"] = np.where(vt == 0, "push", np.where(vt > 0, "win", "loss"))
    ht["flag"] = et.abs() >= C.TOTAL_EDGE

    by_season = []
    for s, g in d.groupby("season"):
        hs, tt = has[has.season == s], ht[ht.season == s]
        by_season.append(dict(
            season=int(s), games=len(g),
            team_mae=round(float(pd.concat([(g.pred_h - g.home_pts).abs(), (g.pred_a - g.away_pts).abs()]).mean()), 2),
            margin_mae=round(float((g.pm - g.am).abs().mean()), 2),
            mkt_margin_mae=round(float((-hs.cfbd_spread - hs.am).abs().mean()), 2) if len(hs) else None,
            total_mae=round(float((g.pt - g["at"]).abs().mean()), 2),
            ats=_rec(hs[hs.flag].res), ou=_rec(tt[tt.flag].res),
        ))

    # Calibration: when the model says the home team covers X% of the time, how often does it?
    cal = []
    bins = [0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]
    nopush = has[has.res != "push"].copy()
    nopush["home_cov"] = ((nopush.side == "home") & (nopush.res == "win")) | ((nopush.side == "away") & (nopush.res == "loss"))
    nopush["bin"] = pd.cut(nopush.p_home_cover, bins)
    for b, g in nopush.groupby("bin", observed=True):
        if len(g) >= 20:
            cal.append(dict(bin=f"{b.left:.0%}–{b.right:.0%}", predicted=round(float(g.p_home_cover.mean()), 3),
                            actual=round(float(g.home_cov.mean()), 3), n=len(g)))

    # Learned shrink factors: how much of the model's confidence vs the line holds up.
    def shrink(p, hit):
        x, y = np.asarray(p) - 0.5, np.asarray(hit, float) - 0.5
        return float(np.clip((x * y).sum() / max((x * x).sum(), 1e-9), 0, 1)) if len(x) > 200 else 0.5
    shrink_spread = shrink(nopush.p_home_cover, nopush.home_cov)
    ht2 = ht[ht.res != "push"].copy()
    p_over = _phi((ht2.pt - ht2.cfbd_total) / ht2.sd_t)
    shrink_total = shrink(p_over, ht2["at"] > ht2.cfbd_total)

    return dict(
        shrink_spread=round(shrink_spread, 3),
        shrink_total=round(shrink_total, 3),
        seasons=by_season,
        overall=dict(
            games=len(d),
            team_mae=round(float(pd.concat([(d.pred_h - d.home_pts).abs(), (d.pred_a - d.away_pts).abs()]).mean()), 2),
            margin_mae=round(float((d.pm - d.am).abs().mean()), 2),
            mkt_margin_mae=round(float((-has.cfbd_spread - has.am).abs().mean()), 2) if len(has) else None,
            total_mae=round(float((d.pt - d["at"]).abs().mean()), 2),
            mkt_total_mae=round(float((ht.cfbd_total - ht["at"]).abs().mean()), 2) if len(ht) else None,
            ats=_rec(has[has.flag].res), ou=_rec(ht[ht.flag].res),
        ),
        calibration=cal,
        segments=segments(d),
    )


def _seg(d, col, err):
    rows = []
    for k, g in d.groupby(col, observed=True):
        n = len(g)
        if n < 30:
            continue
        m, se = float(g[err].mean()), float(g[err].std() / math.sqrt(n))
        rows.append(dict(group=str(k), n=n, bias=round(m, 2), mae=round(float(g[err].abs().mean()), 2),
                         flag=bool(n >= 100 and abs(m) > 2.5 * se)))
    return rows


def segments(d):
    """Where is the model systematically off? bias = average (predicted - actual)."""
    d = d.copy()
    d["margin_err"] = d.pm - d.am
    d["total_err"] = d.pt - d["at"]
    d["week_bucket"] = pd.cut(d.wk, [0, 3, 7, 12, 30], labels=["Weeks 1–3", "Weeks 4–7", "Weeks 8–12", "Late/bowls"])
    d["fav_size"] = pd.cut(d.pm.abs(), [-0.1, 3, 7, 14, 21, 100], labels=["0–3", "3–7", "7–14", "14–21", "21+"])
    # Home-side bias tells you if home field is mis-set.
    return dict(
        margin_by_week=_seg(d, "week_bucket", "margin_err"),
        margin_by_matchup=_seg(d, "matchup", "margin_err"),
        margin_by_fav_size=_seg(d, "fav_size", "margin_err"),
        total_by_week=_seg(d, "week_bucket", "total_err"),
        total_by_matchup=_seg(d, "matchup", "total_err"),
    )


def live(store, season):
    rows = [p for p in store.values() if p.get("result") and str(p.get("start", ""))[:4] in (str(season), str(season + 1))]
    if not rows:
        return dict(graded=0)
    r = pd.DataFrame([dict(**p["result"], wk=p["wk"]) for p in rows])
    out = dict(
        graded=len(r),
        team_mae=round(float(pd.concat([r.err_home.abs(), r.err_away.abs()]).mean()), 2),
        margin_mae=round(float(r.err_margin.abs().mean()), 2),
        total_mae=round(float(r.err_total.abs().mean()), 2),
        mkt_margin_mae=round(float(r.mkt_err_margin.abs().mean()), 2) if "mkt_err_margin" in r and r.mkt_err_margin.notna().any() else None,
        ats=_rec(r["ats"]) if "ats" in r else _rec(pd.Series([], dtype=str)),
        ou=_rec(r["ou"]) if "ou" in r else _rec(pd.Series([], dtype=str)),
    )
    clv = []
    for c in ("clv_spread", "clv_total"):
        if c in r:
            clv += r[c].dropna().tolist()
    out["clv_avg"] = round(float(np.mean(clv)), 2) if clv else None
    out["clv_beat_pct"] = round(float(np.mean([x > 0 for x in clv])), 3) if clv else None
    out["clv_n"] = len(clv)
    return out
