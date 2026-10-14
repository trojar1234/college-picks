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
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(reps, len(g)))
    rates = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return [round(float(np.quantile(rates, 0.025)), 3), round(float(np.quantile(rates, 0.975)), 3)]


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

    sa = _grade_spread(d, "cfbd_spread", 0)   # every game vs the closing line
    sa = sa[(sa.pm + sa.cfbd_spread) != 0]   # model exactly on the line = no pick
    ta = _grade_total(d, "cfbd_total", 0)
    ta = ta[ta.pt != ta.cfbd_total]
    so = _grade_spread(d, "cfbd_spread_open", C.SPREAD_EDGE)
    to = _grade_total(d, "cfbd_total_open", C.TOTAL_EDGE)
    # The same bets under the live rules: gaps of 10+ ("check news"), FCS games and the first
    # three weeks don't become bets. (QB changes and weather can't be replayed historically.)
    def live_rule(x, gap, cap):
        ok = (gap.abs() < cap) & (x.wk > 3)
        if "home_tier" in x and "away_tier" in x:
            ok &= (x.home_tier != "FCS") & (x.away_tier != "FCS")
        return x[ok]
    sol = live_rule(so, so.cfbd_spread_open + so.pm, C.DISAGREE_SPREAD)
    tol = live_rule(to, to.pt - to.cfbd_total_open, C.DISAGREE_TOTAL)
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
    for lab, fs_m, ft_m in (
            ("Gap 10+ points (blocked)", fs_.edge >= C.DISAGREE_SPREAD, ft_.edge >= C.DISAGREE_TOTAL),
            ("Weeks 1–3 (blocked)", fs_.wk <= 3, ft_.wk <= 3),
            ("FCS opponent (blocked)", fcs(fs_), fcs(ft_)),
            ("None of these: a bet under today's rules",
             (fs_.edge < C.DISAGREE_SPREAD) & (fs_.wk > 3) & ~fcs(fs_),
             (ft_.edge < C.DISAGREE_TOTAL) & (ft_.wk > 3) & ~fcs(ft_))):
        cautions.append(dict(label=lab, spread=_rec(fs_[fs_m].res), total=_rec(ft_[ft_m].res)))

    # Honest threshold test (no season helps pick its own rule), with and without the cautions
    def live_ok(x, cap):
        return x[(x.edge < cap) & (x.wk > 3) & ~fcs(x)]
    nested = dict(
        spread=_nested(gso, (5, 6, 7, 8, 9, 10)), total=_nested(gto, (5, 6, 7, 8, 9, 10)),
        spread_live=_nested(live_ok(gso, C.DISAGREE_SPREAD), (5, 6, 7, 8, 9)),
        total_live=_nested(live_ok(gto, C.DISAGREE_TOTAL), (5, 6, 7, 8, 9)),
    )

    # Uncertainty that allows for bets in the same week moving together
    boot = dict(ats_open=_block_boot(so), ou_open=_block_boot(to), ats_open_live=_block_boot(sol), ou_open_live=_block_boot(tol))

    # Win probability: accuracy (Brier score; lower is better) and calibration in equal-size groups
    wp = d.assign(p=_phi(d.pm / d.sd_m))
    wp = wp[wp.am != 0]
    wp["y"] = (wp.am > 0).astype(float)
    mk = wp.dropna(subset=["cfbd_spread"])
    win_prob = dict(brier=_brier(wp.p, wp.y), n=int(len(wp)),
                    brier_matched=_brier(mk.p, mk.y),
                    brier_market=_brier(_phi(-mk.cfbd_spread / mk.sd_m), mk.y), n_matched=int(len(mk)), bins=[])
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

    return dict(
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


def season_of(start):
    """Season a game belongs to: January bowl games count toward the previous season."""
    t = pd.Timestamp(start)
    return t.year if t.month >= 7 else t.year - 1


def live(store, season, margin_sd=None):
    """This season's prospectively recorded results (graded at kickoff from frozen predictions)."""
    rows = sorted([p for p in store.values() if p.get("result") and p.get("start") and season_of(p["start"]) == season],
                  key=lambda p: p["start"])
    if not rows:
        return dict(graded=0)
    r = pd.DataFrame([dict(**p["result"], wk=p["wk"], start=p["start"], p_home_win=p.get("p_home_win"),
                           p_side_cover=p.get("p_side_cover"), p_side_total=p.get("p_side_total"),
                           sflag=bool(p.get("spread_flag")), tflag=bool(p.get("total_flag")),
                           fs_ts=(p.get("first_spread_flag") or {}).get("ts"),
                           ft_ts=(p.get("first_total_flag") or {}).get("ts")) for p in rows])
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
                out["brier_market"] = _brier(_phi(-w.close_spread[m].astype(float) / margin_sd), y[m])
    # Published cover chances vs what happened (every game, the model's side vs the last line)
    for k, pc, res_c in (("spread", "p_side_cover", "ats_all"), ("total", "p_side_total", "ou_all")):
        if res_c in r:
            x = r[r[pc].notna() & r[res_c].isin(["win", "loss"])]
            if len(x):
                out[f"published_{k}"] = dict(avg_p=round(float(x[pc].mean()), 3), actual=round(float((x[res_c] == "win").mean()), 3),
                                             n=int(len(x)), brier=_brier(x[pc], (x[res_c] == "win").astype(float)))
    return out
