"""Report card: backtest results, live results, calibration, and error diagnostics."""
import math

import numpy as np
import pandas as pd

import cfb_config as C


def _phi(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2)))


def _rec(results):
    w = int((results == "win").sum())
    l = int((results == "loss").sum())
    p = int((results == "push").sum())
    return dict(w=w, l=l, p=p, pct=round(w / (w + l), 3) if w + l else None)


def _wilson(w, l, z=1.96):
    n = w + l
    if not n:
        return None
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


def _units(res):
    """Profit in units at -110 (risk 1.1 to win 1 is scaled here to: win +0.909, loss -1)."""
    return sum(100 / 110 if r == "win" else -1.0 if r == "loss" else 0.0 for r in res)


def _book(res):
    """Running record for a chronological list of results: units, ROI, worst drawdown, longest losing streak."""
    units = peak = dd = 0.0
    streak = worst = 0
    w = l = 0
    for r in res:
        if r == "win":
            units += 100 / 110
            w += 1
            streak = 0
        elif r == "loss":
            units -= 1
            l += 1
            streak += 1
            worst = max(worst, streak)
        peak = max(peak, units)
        dd = max(dd, peak - units)
    return dict(units=round(units, 2), roi=round(units / (w + l), 3) if w + l else None,
                max_drawdown=round(dd, 2), losing_streak=worst)


EDGE_BINS = [0, 2, 4, 6, 8, 10, 1000]
EDGE_LABELS = ["0–2", "2–4", "4–6", "6–8", "8–10", "10+"]


def _by_edge(x):
    """Record by size of the model-vs-line gap (x needs columns res and edge)."""
    if not len(x):
        return []
    cut = pd.cut(x.edge, EDGE_BINS, right=False, labels=EDGE_LABELS)
    return [dict(bucket=lab, **_rec(x.res[cut == lab])) for lab in EDGE_LABELS]


def _nested(x, cands, min_bets=30):
    """Honest test of a threshold rule: each season is bet with the threshold that made the most
    units on the seasons BEFORE it (needs min_bets there), so the season being scored never helped
    choose its own rule. x needs columns season, edge, res."""
    rows, used = [], []
    for s in sorted(x.season.unique()):
        prior = x[x.season < s]
        best = None
        for t in cands:
            r = prior[prior.edge >= t]
            if ((r.res == "win") | (r.res == "loss")).sum() < min_bets:
                continue
            u = _units(r.res)
            if best is None or u > best[1]:
                best = (t, u)
        if best is None:
            continue
        test = x[(x.season == s) & (x.edge >= best[0])]
        used.append(test)
        rows.append(dict(season=int(s), threshold=best[0], **_rec(test.res)))
    tot = pd.concat(used) if used else x.iloc[0:0]
    return dict(seasons=rows, total=_rec(tot.res), units=round(_units(tot.res), 2))


def _block_boot(x, reps=2000, seed=7):
    """95% range for a win rate when bets in the same week move together: resample whole
    season-weeks instead of single bets."""
    x = x[x.res.isin(["win", "loss"])]
    if len(x) < 20:
        return None
    g = x.assign(win=(x.res == "win").astype(int)).groupby(["season", "wk"]).win.agg(["sum", "count"]).to_numpy()
    if len(g) < 8:  # too few weeks to resample: the range would look far more certain than it is
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(reps, len(g)))
    rates = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return [round(float(np.quantile(rates, 0.025)), 3), round(float(np.quantile(rates, 0.975)), 3)]


def _logloss(p, y):
    p, y = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4), np.asarray(y, float)
    return round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4) if len(p) else None


def _units_range(x, reps=2000, seed=11):
    """95% range for total units from bets x (columns season, wk, res), resampling whole weeks."""
    x = x[x.res.isin(["win", "loss", "push"])]
    if len(x) < 20:
        return None
    u = x.res.map({"win": 100 / 110, "loss": -1.0, "push": 0.0})
    g = u.groupby([x.season, x.wk]).sum().to_numpy()
    if len(g) < 8:  # too few weeks to resample: the range would look far more certain than it is
        return None
    rng = np.random.default_rng(seed)
    tot = g[rng.integers(0, len(g), size=(reps, len(g)))].sum(1)
    return [round(float(np.quantile(tot, 0.025)), 1), round(float(np.quantile(tot, 0.975)), 1)]


def _accuracy(err):
    err = pd.Series(err).dropna()
    return dict(mae=round(float(err.abs().mean()), 2), rmse=round(float(np.sqrt((err ** 2).mean())), 2),
                bias=round(float(err.mean()), 2), n=int(len(err))) if len(err) else {}


def _brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return round(float(np.mean((p - y) ** 2)), 4) if len(p) else None


def _matchup(r):
    t = sorted([r.home_tier or "FCS", r.away_tier or "FCS"])
    if "FCS" in t:
        return "FCS involved"
    return {("P4", "P4"): "P4 vs P4", ("G5", "P4"): "P4 vs G5", ("G5", "G5"): "G5 vs G5"}[tuple(t)]


def _grade_spread(d, col, thr):
    x = d.dropna(subset=[col]).copy()
    e = x[col] + x.pm  # >0: home side has value vs this line
    v = np.where(e > 0, x.am + x[col], -(x.am + x[col]))
    x["res"] = np.where(v == 0, "push", np.where(v > 0, "win", "loss"))
    return x[e.abs() >= thr]


def _grade_total(d, col, thr):
    x = d.dropna(subset=[col]).copy()
    e = x.pt - x[col]
    v = np.where(e > 0, x["at"] - x[col], x[col] - x["at"])
    x["res"] = np.where(v == 0, "push", np.where(v > 0, "win", "loss"))
    return x[e.abs() >= thr]


def backtest(wf, SEASONS, qb=None):
    """qb: optional QB-change table (game_id, qbd_h, qbd_a) so the QB-change caution can be replayed."""
    if wf.empty:
        return {}
    ln = pd.concat([SEASONS[s]["lines"] for s in SEASONS if len(SEASONS[s]["lines"])], ignore_index=True)
    d = wf.merge(ln, on="game_id", how="left")
    has_qb = qb is not None and len(qb) > 0
    if has_qb:
        q = qb[["game_id", "qbd_h", "qbd_a"]].drop_duplicates("game_id")
        d = d.merge(q, on="game_id", how="left")
        d["qbc"] = np.maximum(d.qbd_h.abs().fillna(0), d.qbd_a.abs().fillna(0)) > 0.05  # same test as the live alert
    else:
        d["qbc"] = False
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

    sa = _grade_spread(d, "cfbd_spread", 0)   # every game vs the closing line
    sa = sa[(sa.pm + sa.cfbd_spread) != 0]   # model exactly on the line = no pick
    ta = _grade_total(d, "cfbd_total", 0)
    ta = ta[ta.pt != ta.cfbd_total]
    so = _grade_spread(d, "cfbd_spread_open", C.SPREAD_EDGE)
    to = _grade_total(d, "cfbd_total_open", C.TOTAL_EDGE)
    # The same bets under the live rules: gaps of 10+ ("check news") and QB changes never become bets;
    # the first three weeks and FCS games follow the switches in cfb_config (FCS always blocks totals).
    # (Weather can't be replayed historically.)
    early_s = getattr(C, "EARLY_SEASON_BLOCKS_SPREADS", True)
    early_t = getattr(C, "EARLY_SEASON_BLOCKS_TOTALS", True)
    fcs_s = getattr(C, "FCS_BLOCKS_SPREADS", True)
    def live_rule(x, gap, cap, spreads):
        ok = (gap.abs() < cap) & ~x.qbc
        if early_s if spreads else early_t:
            ok &= x.wk > 3
        if (not spreads or fcs_s) and "home_tier" in x and "away_tier" in x:
            ok &= (x.home_tier != "FCS") & (x.away_tier != "FCS")
        return x[ok]
    sol = live_rule(so, so.cfbd_spread_open + so.pm, C.DISAGREE_SPREAD, True)
    tol = live_rule(to, to.pt - to.cfbd_total_open, C.DISAGREE_TOTAL, False)
    # Does the line move toward the model? (opening -> closing, games with a real disagreement)
    mv = d.dropna(subset=["cfbd_spread", "cfbd_spread_open"])
    e = mv.cfbd_spread_open + mv.pm
    mv = mv[(e.abs() >= 5) & (mv.cfbd_spread != mv.cfbd_spread_open)]
    e = e.loc[mv.index]
    toward = (np.sign(mv.cfbd_spread_open - mv.cfbd_spread) == np.sign(e))
    move = dict(pct=round(float(toward.mean()), 3) if len(mv) else None, n=int(len(mv)))

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
            ats_open=_rec(so[so.season == s].res), ou_open=_rec(to[to.season == s].res),
            ats_all=_rec(sa[sa.season == s].res), ou_all=_rec(ta[ta.season == s].res),
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
    def spread_shrink(col):
        x = d.dropna(subset=[col])
        x = x[(x.am + x[col]) != 0]
        p = _phi((x.pm + x[col]) / x.sd_m)
        return shrink(p, (x.am + x[col]) > 0), len(x)

    def total_shrink(col):
        x = d.dropna(subset=[col])
        x = x[x["at"] != x[col]]
        p = _phi((x.pt - x[col]) / x.sd_t)
        return shrink(p, x["at"] > x[col]), len(x)

    ss, n_open = spread_shrink("cfbd_spread_open")
    shrink_spread = ss if n_open > 500 else spread_shrink("cfbd_spread")[0]
    st, n_open_t = total_shrink("cfbd_total_open")
    shrink_total = st if n_open_t > 500 else total_shrink("cfbd_total")[0]

    # --- Reporting extras (none of these change any rule) ---
    # Record by gap size, against the opening line (when bets are placed) and the closing line
    def graded(kind, col):
        if kind == "spread":
            x = _grade_spread(d, col, 0)
            x = x.assign(edge=(x[col] + x.pm).abs())
        else:
            x = _grade_total(d, col, 0)
            x = x.assign(edge=(x.pt - x[col]).abs())
        return x[x.edge > 0]
    gso, gsc, gto, gtc = graded("spread", "cfbd_spread_open"), graded("spread", "cfbd_spread"), \
        graded("total", "cfbd_total_open"), graded("total", "cfbd_total")
    edge_buckets = dict(spread_open=_by_edge(gso), spread_close=_by_edge(gsc),
                        total_open=_by_edge(gto), total_close=_by_edge(gtc))

    # How the cautions would have done: flagged bets (7+ vs open) split by each rule that blocks them
    def fcs(x):
        return (x.home_tier == "FCS") | (x.away_tier == "FCS")
    fs_, ft_ = gso[gso.edge >= C.SPREAD_EDGE], gto[gto.edge >= C.TOTAL_EDGE]
    cautions = []
    clean_s = (fs_.edge < C.DISAGREE_SPREAD) & ~fs_.qbc
    clean_t = (ft_.edge < C.DISAGREE_TOTAL) & ~ft_.qbc & ~fcs(ft_)
    for lab, fs_m, ft_m in (
            ("Gap 10+ points (blocked)", fs_.edge >= C.DISAGREE_SPREAD, ft_.edge >= C.DISAGREE_TOTAL),
            ("Weeks 1–3 (" + ("blocked" if early_s and early_t else "allowed from 2027" if not early_s and not early_t
                             else "spreads allowed from 2027" if not early_s else "totals allowed from 2027") + ")", fs_.wk <= 3, ft_.wk <= 3),
            ("FCS opponent (blocked)", fcs(fs_), fcs(ft_)),
            *((("QB change (blocked)", fs_.qbc, ft_.qbc),) if has_qb else ()),
            ("Weeks 1–3 with no other caution", (fs_.wk <= 3) & clean_s & ~fcs(fs_), (ft_.wk <= 3) & clean_t),
            ("After week 3 with no caution", (fs_.wk > 3) & clean_s & ~fcs(fs_), (ft_.wk > 3) & clean_t)):
        cautions.append(dict(label=lab, spread=_rec(fs_[fs_m].res), total=_rec(ft_[ft_m].res)))

    # Honest threshold test (no season helps pick its own rule), with and without the cautions
    def live_ok(x, cap, spreads):
        ok = (x.edge < cap) & ~x.qbc
        if early_s if spreads else early_t:
            ok &= x.wk > 3
        if not spreads or fcs_s:
            ok &= ~fcs(x)
        return x[ok]
    nested = dict(
        spread=_nested(gso, (5, 6, 7, 8, 9, 10)), total=_nested(gto, (5, 6, 7, 8, 9, 10)),
        spread_live=_nested(live_ok(gso, C.DISAGREE_SPREAD, True), (5, 6, 7, 8, 9)),
        total_live=_nested(live_ok(gto, C.DISAGREE_TOTAL, False), (5, 6, 7, 8, 9)),
    )

    # Uncertainty that allows for bets in the same week moving together
    boot = dict(ats_open=_block_boot(so), ou_open=_block_boot(to), ats_open_live=_block_boot(sol), ou_open_live=_block_boot(tol))

    # Win probability: accuracy (Brier score; lower is better) and calibration in equal-size groups
    wp = d.assign(p=_phi(d.pm / d.sd_m))
    wp = wp[wp.am != 0]
    wp["y"] = (wp.am > 0).astype(float)
    mk = wp.dropna(subset=["cfbd_spread"]).copy()
    # The line's own win probability: closing spread through a normal curve with the MARKET's error
    # size (from its own misses), not the model's, so neither side borrows the other's calibration.
    mkt_sd = float((mk.am + mk.cfbd_spread).std()) if len(mk) else None
    mk["pm_"] = _phi(-mk.cfbd_spread / mkt_sd) if mkt_sd else np.nan
    base_rate = float(wp.y.mean())  # naive forecast: the home team's historical win rate
    win_prob = dict(brier=_brier(wp.p, wp.y), logloss=_logloss(wp.p, wp.y), n=int(len(wp)),
                    brier_matched=_brier(mk.p, mk.y), logloss_matched=_logloss(mk.p, mk.y),
                    brier_market=_brier(mk.pm_, mk.y), logloss_market=_logloss(mk.pm_, mk.y),
                    market_sd=round(mkt_sd, 2) if mkt_sd else None,
                    brier_naive=_brier(np.full(len(wp), base_rate), wp.y), n_matched=int(len(mk)), bins=[])
    # Paired range for (model - market) Brier on the same games, resampling whole weeks
    if len(mk) > 200:
        dif = ((mk.p - mk.y) ** 2 - (mk.pm_ - mk.y) ** 2)
        g = dif.groupby([mk.season, mk.wk]).agg(["sum", "count"]).to_numpy()
        rng = np.random.default_rng(3)
        idx = rng.integers(0, len(g), size=(2000, len(g)))
        bs = g[idx, 0].sum(1) / g[idx, 1].sum(1)
        win_prob["brier_diff"] = round(float(dif.mean()), 4)
        win_prob["brier_diff_range"] = [round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)]
    win_prob["by_matchup"] = [dict(group=str(k), brier=_brier(g.p, g.y), n=int(len(g))) for k, g in wp.groupby("matchup")]
    win_prob["by_season"] = [dict(season=int(k), brier=_brier(g.p, g.y), logloss=_logloss(g.p, g.y), n=int(len(g)))
                             for k, g in wp.groupby("season")]
    if len(wp) >= 200:
        q = pd.qcut(wp.p, 10, duplicates="drop")
        for b, g in wp.groupby(q, observed=True):
            win_prob["bins"].append(dict(predicted=round(float(g.p.mean()), 3), actual=round(float(g.y.mean()), 3), n=int(len(g))))

    # Data coverage by season: how many games had each kind of line
    for row in by_season:
        g = d[d.season == row["season"]]
        row["coverage"] = {k: int(g[c].notna().sum()) for k, c in (
            ("open_spread", "cfbd_spread_open"), ("close_spread", "cfbd_spread"),
            ("open_total", "cfbd_total_open"), ("close_total", "cfbd_total"))}

    # Accuracy beyond the average miss: big misses (RMSE) and lean (bias = predicted minus actual)
    accuracy = dict(margin=_accuracy(d.pm - d.am), total=_accuracy(d.pt - d["at"]),
                    market_margin=_accuracy(-has.cfbd_spread - has.am), market_total=_accuracy(ht.cfbd_total - ht["at"]))
    # How often the actual result landed inside the model's 80% and 95% ranges
    coverage = dict(margin80=round(float(((d.am - d.pm).abs() <= 1.2816 * d.sd_m).mean()), 3),
                    margin95=round(float(((d.am - d.pm).abs() <= 1.96 * d.sd_m).mean()), 3),
                    total80=round(float(((d["at"] - d.pt).abs() <= 1.2816 * d.sd_t).mean()), 3),
                    total95=round(float(((d["at"] - d.pt).abs() <= 1.96 * d.sd_t).mean()), 3))
    profit = dict(ats_open=dict(units=round(_units(so.res), 1), range=_units_range(so)),
                  ou_open=dict(units=round(_units(to.res), 1), range=_units_range(to)),
                  ats_open_live=dict(units=round(_units(sol.res), 1), range=_units_range(sol)),
                  ou_open_live=dict(units=round(_units(tol.res), 1), range=_units_range(tol)))
    for row in by_season:
        g = d[d.season == row["season"]]
        row["margin_rmse"] = round(float(np.sqrt(((g.pm - g.am) ** 2).mean())), 2)
        row["margin_bias"] = round(float((g.pm - g.am).mean()), 2)
        row["total_bias"] = round(float((g.pt - g["at"]).mean()), 2)

    # Where the model leans against the market, by conference: model minus closing line (positive =
    # the model likes the team more than the market does) next to how teams actually did against it.
    conf = {}
    for s_ in SEASONS:
        g = SEASONS[s_]["games"]
        if len(g) and "home_conf" in g:
            conf.update(dict(zip(g.home, g.home_conf)))
            conf.update(dict(zip(g.away, g.away_conf)))
    lean = []
    if conf and len(has):
        rows_ = []
        for side, sign in (("home", 1), ("away", -1)):
            rows_.append(pd.DataFrame(dict(conf=has[side].map(conf), gap=sign * (has.pm + has.cfbd_spread),
                                           res=sign * (has.am + has.cfbd_spread))))
        t = pd.concat(rows_).dropna(subset=["conf"])
        t = t[t.conf != ""]
        for k, g in t.groupby("conf"):
            if len(g) >= 150:
                lean.append(dict(group=str(k), n=int(len(g)), model_vs_line=round(float(g.gap.mean()), 2),
                                 actual_vs_line=round(float(g.res.mean()), 2)))
        lean.sort(key=lambda r: r["model_vs_line"])

    return dict(
        conf_lean=lean, accuracy=accuracy, coverage=coverage, profit=profit,
        edge_buckets=edge_buckets, cautions=cautions, nested=nested, boot=boot, win_prob=win_prob,
        shrink_spread=round(shrink_spread, 3),
        shrink_total=round(shrink_total, 3),
        seasons=by_season,
        overall=dict(
            games=len(d),
            team_mae=round(float(pd.concat([(d.pred_h - d.home_pts).abs(), (d.pred_a - d.away_pts).abs()]).mean()), 2),
            margin_mae=round(float((d.pm - d.am).abs().mean()), 2),
            mkt_margin_mae=round(float((-has.cfbd_spread - has.am).abs().mean()), 2) if len(has) else None,
            matched_margin_mae=round(float((has.pm - has.am).abs().mean()), 2) if len(has) else None,
            matched_n=int(len(has)),
            total_mae=round(float((d.pt - d["at"]).abs().mean()), 2),
            mkt_total_mae=round(float((ht.cfbd_total - ht["at"]).abs().mean()), 2) if len(ht) else None,
            ats=_rec(has[has.flag].res), ou=_rec(ht[ht.flag].res),
            ats_open=_rec(so.res), ou_open=_rec(to.res),
            ats_open_live=_rec(sol.res), ou_open_live=_rec(tol.res),
            ats_all=_rec(sa.res), ou_all=_rec(ta.res),
        ),
        line_move=move,
        by_phase=[dict(label=lab, ats_open=_rec(so[(so.wk >= lo) & (so.wk <= hi)].res),
                       ou_open=_rec(to[(to.wk >= lo) & (to.wk <= hi)].res))
                  for lab, lo, hi in (("Weeks 1–4", 0, 4), ("Weeks 5–8", 5, 8), ("Weeks 9+", 9, 99))]
        if "wk" in so and "wk" in to else [],
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


def monitor(wf, season, state):
    """Drift checks for this season, from the walk-forward predictions (each week predicted before it
    was played): does the model lean too high or too low on margins, home teams or totals? A lean is
    flagged when it's more than 2.5 standard errors from zero."""
    out = {}
    d = wf[(wf.season == season) & wf.home_pts.notna()] if len(wf) else wf
    if len(d):
        d = d[d.home_tier.isin(["P4", "G5"]) & d.away_tier.isin(["P4", "G5"])]
    def lean(err, label):
        err = pd.Series(err).dropna()
        if len(err) < 30:
            return None
        m, se = float(err.mean()), float(err.std() / math.sqrt(len(err)))
        return dict(label=label, lean=round(m, 2), se=round(se, 2), n=int(len(err)), flag=bool(abs(m) > 2.5 * se))
    if len(d):
        pm, am = d.pred_h - d.pred_a, d.home_pts - d.away_pts
        out["leans"] = [x for x in (
            lean(pm - am, "Home team's margin (projected minus actual)"),
            lean((d.pred_h + d.pred_a) - (d.home_pts + d.away_pts), "Game total (projected minus actual)"),
            lean(((pm - am) * np.where(pm >= 0, 1, -1)), "Favorite's margin (projected minus actual)"),
        ) if x]
    hf = (state or {}).get("home_field") or {}
    if hf:
        out["home_field"] = hf
    return out


def season_of(start):
    """Season a game belongs to: January bowl games count toward the previous season."""
    t = pd.Timestamp(start)
    return t.year if t.month >= 7 else t.year - 1


def live(store, season, margin_sd=None, total_sd=None):
    """This season's prospectively recorded results (graded at kickoff from frozen predictions)."""
    rows = sorted([p for p in store.values() if p.get("result") and p.get("start") and season_of(p["start"]) == season],
                  key=lambda p: p["start"])
    if not rows:
        return dict(graded=0)
    r = pd.DataFrame([dict(**p["result"], wk=p["wk"], start=p["start"], game_id=p.get("game_id"), p_home_win=p.get("p_home_win"),
                           p_side_cover=p.get("p_side_cover"), p_side_total=p.get("p_side_total"),
                           sflag=bool(p.get("spread_flag")), tflag=bool(p.get("total_flag")),
                           model_spread=p.get("model_spread"), model_total=p.get("model_total"),
                           home_tier=p.get("home_tier"), away_tier=p.get("away_tier"),
                           fs=p.get("first_spread_flag"), ft=p.get("first_total_flag"),
                           fs_ts=(p.get("first_spread_flag") or {}).get("ts"),
                           ft_ts=(p.get("first_total_flag") or {}).get("ts"),
                           qb="QB change" in (p.get("low_conf") or []), ch_total=p.get("challenger_total"),
                           open_spread=(p.get("market") or {}).get("open_spread")) for p in rows])
    col = lambda c: r[c] if c in r else pd.Series([None] * len(r))
    out = dict(
        graded=len(r),
        team_mae=round(float(pd.concat([r.err_home.abs(), r.err_away.abs()]).mean()), 2),
        margin_mae=round(float(r.err_margin.abs().mean()), 2),
        total_mae=round(float(r.err_total.abs().mean()), 2),
        mkt_margin_mae=round(float(r.mkt_err_margin.abs().mean()), 2) if "mkt_err_margin" in r and r.mkt_err_margin.notna().any() else None,
        ats=_rec(col("ats")), ou=_rec(col("ou")), ats_all=_rec(col("ats_all")), ou_all=_rec(col("ou_all")),
    )
    # Model vs market on exactly the same games
    if "mkt_err_margin" in r:
        m = r[r.mkt_err_margin.notna()]
        out.update(matched_margin_mae=round(float(m.err_margin.abs().mean()), 2) if len(m) else None,
                   matched_mkt_margin_mae=round(float(m.mkt_err_margin.abs().mean()), 2) if len(m) else None,
                   matched_n=int(len(m)))
    if "mkt_err_total" in r:
        m = r[r.mkt_err_total.notna()]
        out.update(matched_total_mae=round(float(m.err_total.abs().mean()), 2) if len(m) else None,
                   matched_mkt_total_mae=round(float(m.mkt_err_total.abs().mean()), 2) if len(m) else None)
    # Spreads and totals scored separately: record, units, ROI, drawdown, streak, closing-line value
    for k, res_c, clv_c in (("spread", "ats", "clv_spread"), ("total", "ou", "clv_total")):
        res = [x for x in col(res_c) if isinstance(x, str)]
        c = col(clv_c).dropna().astype(float) if clv_c in r else pd.Series([], dtype=float)
        out[k] = dict(**_rec(pd.Series(res, dtype=str)), **_book(res),
                      clv_avg=round(float(c.mean()), 2) if len(c) else None,
                      clv_median=round(float(c.median()), 2) if len(c) else None,
                      clv_beat=round(float((c > 0).mean()), 3) if len(c) else None,
                      clv_push=round(float((c == 0).mean()), 3) if len(c) else None, clv_n=int(len(c)))
    both = [x for x in pd.concat([col("ats"), col("ou")]).tolist() if isinstance(x, str)]
    out["combined"] = dict(**_rec(pd.Series(both, dtype=str)))
    # Accuracy: average miss, big misses, lean (predicted minus actual), on all graded games
    out["accuracy"] = dict(margin=_accuracy(r.err_margin), total=_accuracy(r.err_total),
                           market_margin=_accuracy(col("mkt_err_margin")), market_total=_accuracy(col("mkt_err_total")))
    if margin_sd:
        out["coverage"] = dict(margin80=round(float((r.err_margin.abs() <= 1.2816 * margin_sd).mean()), 3),
                               total80=round(float((r.err_total.abs() <= 1.2816 * (total_sd or margin_sd)).mean()), 3))
    # Plausible range for each market's profit so far (resampling whole weeks)
    bets = []
    for _, x in r.iterrows():
        for k, c in (("spread", "ats"), ("total", "ou")):
            if isinstance(x.get(c), str):
                bets.append(dict(kind=k, res=x[c], season=season, wk=x.wk, start=x.start, gid=x.get("game_id"),
                                 tier=("P4" if "P4" in (x.home_tier, x.away_tier) else "G5"),
                                 day=pd.Timestamp(x.start).tz_convert("America/Chicago").strftime("%a")))
    B = pd.DataFrame(bets)
    if len(B):
        for k in ("spread", "total"):
            if k in out:
                out[k]["units_range"] = _units_range(B[B.kind == k])
        # Segments: Power 4 games vs the rest, Saturday vs weeknight
        out["segments"] = [dict(label=lab, **_rec(B.res[mask]), units=round(_units(B.res[mask]), 2))
                           for lab, mask in (("Power 4 involved", B.tier == "P4"), ("Group of 5 only", B.tier != "P4"),
                                             ("Saturday", B.day == "Sat"), ("Weeknight", B.day != "Sat"))]
        # Bets on both the spread and total of the same game (their results tend to move together)
        out["same_game"] = int((B.groupby(B.gid.astype(str)).size() > 1).sum())
    # Profit if every bet had been placed at the best single book's line and price when it was flagged
    best = []
    for _, x in r.iterrows():
        for f, kind, c in ((x.get("fs"), "spread", "ats"), (x.get("ft"), "total", "ou")):
            if not isinstance(f, dict) or not isinstance(x.get(c), str) or not isinstance(f.get("best"), dict):
                continue
            b = f["best"]
            if b.get("line") is None:
                continue
            if kind == "spread":
                home_line = b["line"] if f["side"] == "home" else -b["line"]
                v = (x.home_pts - x.away_pts) + home_line
                v = v if f["side"] == "home" else -v
            else:
                tot = x.home_pts + x.away_pts
                v = (tot - b["line"]) if f["side"] == "over" else (b["line"] - tot)
            pr = b.get("price")
            pay = (pr / 100 if pr and pr > 0 else 100 / -pr) if pr else 100 / 110
            best.append(pay if v > 0 else (-1.0 if v < 0 else 0.0))
    if best:
        out["best_price"] = dict(n=len(best), units=round(float(sum(best)), 2))
    # Shadow thresholds: what other gap sizes would have done this season against the last line before
    # kickoff (no cautions applied). The live rule is 7; these show whether nearby rules do better.
    sh = []
    for thr in (5, 6, 7, 8, 9):
        for k, line_c, mod_c, res_c in (("spread", "close_spread", "model_spread", "ats_all"), ("total", "close_total", "model_total", "ou_all")):
            if line_c in r and mod_c in r and res_c in r:
                x = r[r[line_c].notna() & r[mod_c].notna()]
                x = x[(x[line_c].astype(float) - x[mod_c].astype(float)).abs() >= thr]
                sh.append(dict(threshold=thr, kind=k, **_rec(x[res_c])))
    out["shadow_thresholds"] = sh
    # Kept for older pages: both markets' CLV pooled
    clv = pd.concat([col(c).dropna().astype(float) for c in ("clv_spread", "clv_total") if c in r]) if \
        any(c in r for c in ("clv_spread", "clv_total")) else pd.Series([], dtype=float)
    out["clv_avg"] = round(float(clv.mean()), 2) if len(clv) else None
    out["clv_beat_pct"] = round(float((clv > 0).mean()), 3) if len(clv) else None
    out["clv_n"] = int(len(clv))
    # Closing-line value by the day the bet was first flagged (Central time)
    days = []
    for ts_c, clv_c in (("fs_ts", "clv_spread"), ("ft_ts", "clv_total")):
        if clv_c in r:
            x = r[[ts_c, clv_c]].dropna()
            for ts, v in zip(x[ts_c], x[clv_c]):
                days.append((pd.Timestamp(ts).tz_convert("America/Chicago").strftime("%a"), float(v)))
    if days:
        dd = pd.DataFrame(days, columns=["day", "clv"])
        order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        out["clv_by_day"] = [dict(day=k, n=int(len(g)), avg=round(float(g.clv.mean()), 2), beat=round(float((g.clv > 0).mean()), 3))
                             for k, g in sorted(dd.groupby("day"), key=lambda kv: order.index(kv[0]))]
    # Flags the cautions kept from being bets, scored at the last line before kickoff (shadow record)
    out["blocked"] = dict(spread=_rec(col("ats_flagonly")), total=_rec(col("ou_flagonly")))
    # Win probabilities as published (from the simulation), and the line's implied ones on the same games
    w = r[r.p_home_win.notna() & (r.home_pts != r.away_pts)]
    if len(w):
        y = (w.home_pts > w.away_pts).astype(float)
        out["brier"] = _brier(w.p_home_win, y)
        out["brier_n"] = int(len(w))
        if margin_sd and "close_spread" in w:
            m = w.close_spread.notna()
            if m.any():
                out["brier_model_matched"] = _brier(w.p_home_win[m], y[m])
                # the line's win chance with the market's own error size from its misses this season
                # (fall back to the model's until there are enough games)
                cs = w.close_spread[m].astype(float)
                msd = float(((w.home_pts - w.away_pts)[m] + cs).std()) if m.sum() >= 100 else margin_sd
                out["brier_market"] = _brier(_phi(-cs / msd), y[m])
    out["paired"] = paired(r)
    out["challenger"] = challenger(r)
    # Published cover chances vs what happened (every game, the model's side vs the last line)
    for k, pc, res_c in (("spread", "p_side_cover", "ats_all"), ("total", "p_side_total", "ou_all")):
        if res_c in r:
            x = r[r[pc].notna() & r[res_c].isin(["win", "loss"])]
            if len(x):
                out[f"published_{k}"] = dict(avg_p=round(float(x[pc].mean()), 3), actual=round(float((x[res_c] == "win").mean()), 3),
                                             n=int(len(x)), brier=_brier(x[pc], (x[res_c] == "win").astype(float)))
    return out


def paired(r):
    """Model vs market on exactly the same games: the market's miss minus the model's miss on the final
    margin (positive = the model was closer), using the projection frozen at kickoff and the last line
    before kickoff, so both had the same information cutoff. Broken down by where the model disagreed,
    to show whether a weak stretch is concentrated (one kind of game) or broad (ordinary variance)."""
    need = ("close_spread", "model_spread", "home_pts", "away_pts")
    if not all(c in r for c in need):
        return None
    x = r[r.close_spread.notna() & r.model_spread.notna() & ~r.home_tier.eq("FCS") & ~r.away_tier.eq("FCS")].copy()
    if len(x) < 20:
        return None
    am = (x.home_pts - x.away_pts).astype(float)
    cs, ms = x.close_spread.astype(float), x.model_spread.astype(float)
    x["adv"] = (am + cs).abs() - (am + ms).abs()
    e = cs - ms  # >0: the model liked the home side against the line
    side_line = np.where(e > 0, cs, -cs)
    cover = np.where(e > 0, am + cs, -(am + cs))
    x["cov"] = np.where(cover > 0, 1.0, np.where(cover < 0, 0.0, np.nan))
    gap = e.abs()

    def row(label, m):
        g = x[m]
        if not len(g):
            return None
        a = g.adv
        c = g["cov"].dropna()
        return dict(label=label, n=int(len(g)), adv=round(float(a.mean()), 2),
                    se=round(float(a.std() / np.sqrt(len(g))), 2) if len(g) > 1 else None,
                    side_cover=round(float(c.mean()), 3) if len(c) else None)
    groups = [
        ("All games", np.ones(len(x), bool)),
        ("Model side at home", e > 0), ("Model side on the road", e < 0),
        ("Model side favored", side_line < 0), ("Model side the underdog", side_line > 0),
        ("Gap under 3", gap < 3), ("Gap 3 to 7", (gap >= 3) & (gap < 7)), ("Gap 7 or more", gap >= 7),
        ("Line 17 or more", cs.abs() >= 17), ("QB change flagged", x.qb.astype(bool)),
        ("Weeks 1-6", x.wk <= 6), ("Week 7 on", x.wk >= 7),
    ]
    rows = [r_ for r_ in (row(lab, np.asarray(m)) for lab, m in groups) if r_]
    out = dict(rows=rows)
    # Recorded bets: the model's miss next to the line's miss when the bet was flagged (the market at the
    # same moment, before later news) and the last line before kickoff
    bets = {}
    for k, fc, rc in (("spread", "fs", "ats"), ("total", "ft", "ou")):
        e = []
        for _, g in r.iterrows():
            f = g.get(fc)
            if not isinstance(g.get(rc), str):  # graded bets only
                continue
            res = g.get("close_spread" if k == "spread" else "close_total")
            if not isinstance(f, dict) or f.get("line") is None or f.get("model") is None or res is None or res != res:
                continue
            if k == "spread":
                act = float(g.home_pts - g.away_pts)
                e.append((abs(act + f["model"]), abs(act + f["line"]), abs(act + float(res))))
            else:
                act = float(g.home_pts + g.away_pts)
                e.append((abs(act - f["model"]), abs(act - f["line"]), abs(act - float(res))))
        if e:
            a_ = np.array(e)
            bets[k] = dict(n=len(e), model=round(float(a_[:, 0].mean()), 2), flagged=round(float(a_[:, 1].mean()), 2),
                           last=round(float(a_[:, 2].mean()), 2))
    if "open_spread" in x and x.open_spread.notna().sum() >= 20:  # timing: first recorded line vs last line
        y = x[x.open_spread.notna()]
        amy = (y.home_pts - y.away_pts).astype(float)
        out["timing"] = dict(n=int(len(y)), model=round(float((amy + y.model_spread.astype(float)).abs().mean()), 2),
                             first=round(float((amy + y.open_spread.astype(float)).abs().mean()), 2),
                             last=round(float((amy + y.close_spread.astype(float)).abs().mean()), 2))
    if bets:
        out["bets"] = bets
    return out


def challenger(r):
    """Shadow record for the separate totals model (never bet): accuracy on the same games as the live
    totals, and what a 7-point rule would have done against the last line before kickoff for each."""
    if "ch_total" not in r or r.ch_total.isna().all():
        return None
    x = r[r.ch_total.notna() & r.model_total.notna()].copy()
    if not len(x):
        return None
    at = (x.home_pts + x.away_pts).astype(float)
    out = dict(n=int(len(x)), mae_live=round(float((at - x.model_total.astype(float)).abs().mean()), 2),
               mae_challenger=round(float((at - x.ch_total.astype(float)).abs().mean()), 2),
               lean_live=round(float((x.model_total.astype(float) - at).mean()), 2),
               lean_challenger=round(float((x.ch_total.astype(float) - at).mean()), 2))
    if "close_total" in x:
        y = x[x.close_total.notna()]
        aty = (y.home_pts + y.away_pts).astype(float)
        out["mae_market"] = round(float((aty - y.close_total.astype(float)).abs().mean()), 2) if len(y) else None
        for k, col in (("live", "model_total"), ("challenger", "ch_total")):
            g = y[(y[col].astype(float) - y.close_total.astype(float)).abs() >= C.TOTAL_EDGE]
            v = np.where(g[col].astype(float) > g.close_total.astype(float), 1, -1) * ((g.home_pts + g.away_pts).astype(float) - g.close_total.astype(float))
            res = pd.Series(np.where(v > 0, "win", np.where(v < 0, "loss", "push")), dtype=str)
            out[f"rec_{k}"] = _rec(res)
    return out


def value_decay(store, hist, season):
    """For each recorded bet: hours from when it was flagged until the consensus line first moved half
    a point or more against it (the value starting to disappear), or None if it never did before kickoff."""
    if hist is None or not len(hist):
        return dict(n=0)
    h = hist.copy()
    h["ts"] = pd.to_datetime(h.ts, utc=True)
    out = []
    for gid, p in store.items():
        if not p.get("start") or season_of(p["start"]) != season:
            continue
        g = h[h.game_id == int(gid)]
        if g.empty:
            continue
        snaps = g.groupby("ts").agg(spread=("home_spread", "median"), total=("total", "median")).sort_index()
        start = pd.Timestamp(p["start"])
        for kind, col in (("spread", "spread"), ("total", "total")):
            f = p.get(f"first_{kind}_flag")
            if not f or f.get("line") is None or not f.get("ts"):
                continue
            t0 = pd.Timestamp(f["ts"])
            after = snaps[(snaps.index > t0) & (snaps.index < start)][col].dropna()
            after = np.floor(after * 2 + 0.5) / 2
            if kind == "spread":
                worse = (after < f["line"] - 0.25) if f["side"] == "home" else (after > f["line"] + 0.25)
            else:
                worse = (after > f["line"] + 0.25) if f["side"] == "over" else (after < f["line"] - 0.25)
            hit = after[worse]
            out.append(dict(kind=kind, hours=round((hit.index[0] - t0).total_seconds() / 3600, 1) if len(hit) else None))
    if not out:
        return dict(n=0)
    hrs = [o["hours"] for o in out if o["hours"] is not None]
    return dict(n=len(out), moved=len(hrs), never=len(out) - len(hrs),
                median_hours=round(float(np.median(hrs)), 1) if hrs else None,
                within_24h=sum(1 for x in hrs if x <= 24))


def flag_clv(store, decisions, season):
    """Closing-line value for flags the cautions blocked (from the decision log's line at the moment of
    flagging), next to the bets' own, so BET and FLAG can be compared."""
    if decisions is None or not len(decisions):
        return {}
    out = {}
    d = decisions[decisions.status == "flag"]
    # A later policy can log a fresh flag on the same game; measure from the first one.
    d = d.sort_values("ts").drop_duplicates(["game_id", "market"], keep="first")
    for kind in ("spread", "total"):
        vals = []
        for r in d[d.market == kind].itertuples():
            p = store.get(str(r.game_id)) or {}
            res = p.get("result") or {}
            if not p.get("start") or season_of(p["start"]) != season:
                continue
            close = res.get("close_spread" if kind == "spread" else "close_total")
            if close is None or r.line != r.line:
                continue
            line = float(r.line)
            if kind == "spread":
                v = (line - close) if r.side == "home" else (close - line)
            else:
                v = (close - line) if r.side == "over" else (line - close)
            vals.append(v)
        if vals:
            v = np.array(vals)
            out[kind] = dict(n=int(len(v)), avg=round(float(v.mean()), 2), beat=round(float((v > 0).mean()), 3))
    return out
