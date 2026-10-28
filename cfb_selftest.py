"""Checks run after every update and before anything is published.

    python cfb_selftest.py

Exits with an error (so the workflow publishes nothing and GitHub emails you) if:
  - the grading, line-handling or record-keeping rules give a wrong answer on known cases,
  - the published data files are missing or unreadable,
  - any page's JavaScript has a syntax error,
  - the Matchup page's formula drifts away from the weekly projections it should reproduce.
The checks only read; they never change the model, the bet rules or any saved data.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import cfb_config as C
import cfb_lines as lines
import cfb_predict as predict
import cfb_report as report

FAIL = []


def check(ok, msg):
    print(("ok    " if ok else "FAIL  ") + msg)
    if not ok:
        FAIL.append(msg)


def test_grading():
    # Spread lines are from the home team's view: home -3 means home favored by 3.
    check(predict._ats(7, "home", -3) == "win", "home -3 wins by 7 -> home covers")
    check(predict._ats(3, "home", -3) == "push", "home -3 wins by 3 -> push")
    check(predict._ats(2, "away", -3) == "win", "away +3, home wins by 2 -> away covers")
    check(predict._ats(-1, "home", 0) == "loss", "pick'em, home loses -> home bet loses")
    check(predict._ou(55, "over", 52.5) == "win" and predict._ou(52, "under", 52.5) == "win", "totals graded both ways")
    check(predict._ou(52, "over", 52) == "push", "total on the number -> push")


def test_best_line():
    books = [dict(book="A", spread=-3.0, total=52.5, home_price=-110, away_price=-110, over_price=-110, under_price=-110),
             dict(book="B", spread=-2.5, total=53.0, home_price=-115, away_price=-105, over_price=-105, under_price=-115),
             dict(book="C", spread=-3.0, total=52.5, home_price=-105, away_price=-115, over_price=-120, under_price=100)]
    check(lines.best_line(books, "spread", "home")["book"] == "B", "best home line is the smallest favorite number (-2.5)")
    check(lines.best_line(books, "spread", "away")["book"] == "A", "best away line: +3, tie broken by price (-110 beats -115)")
    check(lines.best_line(books, "total", "over")["line"] == 52.5, "best over is the lowest total")
    b = lines.best_line(books, "total", "under")
    check(b["line"] == 53.0, "best under is the highest total")
    check(lines.best_line([], "spread", "home") is None, "no books -> no best line")


def test_prices_and_eligibility():
    # unusual prices: even money, plus money, missing price (treated as -110)
    check(abs(lines.ev(0.5, 0, 100) - 0.0) < 1e-9, "a coin flip at even money is worth nothing")
    check(abs(lines.ev(0.5, 0, None) - (0.5 * 100 / 110 - 0.5)) < 1e-9, "a missing price is treated as -110")
    check(lines.ev(0.55, 0, -110) > 0 > lines.ev(0.5, 0, -110), "55% at -110 is positive, 50% is negative")
    books = [dict(book="A", total=39.5, over_price=-110, under_price=-110),
             dict(book="B", total=40.5, over_price=-105, under_price=-115)]
    check(lines.best_line(books, "total", "over", model=47.7, sd=16.6, shrink=0.18)["book"] == "A",
          "a full point on the total is worth more than five cents of price")
    books = [dict(book="A", spread=4.5, home_price=-125, away_price=105), dict(book="B", spread=4.5, home_price=-110, away_price=-110)]
    check(lines.best_line(books, "spread", "home", model=-3, sd=16, shrink=0.25)["book"] == "B", "same line: the better price wins")
    # a bet needs a fresh sportsbook quote
    now = pd.Timestamp("2026-10-20 12:00", tz="UTC")
    fresh = dict(market=dict(spread=-3.0, spread_src="odds", books=[dict(book="A")], last_ts="2026-10-20T08:00:00+00:00"))
    backup = dict(market=dict(spread=-3.0, spread_src="cfbd", books=[], last_ts=None))
    stale = dict(market=dict(spread=-3.0, spread_src="odds", books=[dict(book="A")], last_ts="2026-10-17T08:00:00+00:00"))
    check(predict.bet_eligible(fresh, "spread", now), "a fresh sportsbook quote can become a bet")
    check(not predict.bet_eligible(backup, "spread", now), "a backup CollegeFootballData line can't become a bet")
    check(not predict.bet_eligible(stale, "spread", now), "a quote more than 36 hours old can't become a bet")
    # adverse cases: one market's quote must never make the other look current
    mixed = dict(market=dict(spread=-3.0, total=52.5, spread_src="odds", total_src="odds", books=[dict(book="A")],
                             last_ts="2026-10-20T08:00:00+00:00", spread_ts="2026-10-17T08:00:00+00:00",
                             total_ts="2026-10-20T08:00:00+00:00", spread_books=1, total_books=1))
    check(not predict.bet_eligible(mixed, "spread", now), "an expired spread can't ride on a fresh total's timestamp")
    check(predict.bet_eligible(mixed, "total", now), "the fresh total is still eligible")
    pulled = dict(market=dict(mixed["market"], spread_ts="2026-10-20T08:00:00+00:00", spread_books=0))
    check(not predict.bet_eligible(pulled, "spread", now), "a spread no book offers in the newest snapshot can't become a bet")
    games = pd.DataFrame([dict(game_id=9, start=pd.Timestamp("2026-10-21 16:00", tz="UTC"))])
    hist = pd.DataFrame([dict(ts=pd.Timestamp("2026-10-17 08:00", tz="UTC"), game_id=9, book="A", home_spread=-3.0, total=52.5),
                         dict(ts=pd.Timestamp("2026-10-20 08:00", tz="UTC"), game_id=9, book="A", home_spread=np.nan, total=53.5)])
    mv = lines.market_view(hist, games)[9]
    check(mv["spread_ts"].startswith("2026-10-17") and mv["total_ts"].startswith("2026-10-20"),
          "each market carries the time of its own last quote")
    check(mv["spread_books"] == 0 and mv["total_books"] == 1, "each market counts its own books in the newest snapshot")
    check(not predict.bet_eligible(dict(market=dict(mv, spread_src="odds", total_src="odds")), "spread", now)
          and predict.bet_eligible(dict(market=dict(mv, spread_src="odds", total_src="odds")), "total", now),
          "end to end: pulled spread ineligible, fresh total eligible")
    # the key-number margin distribution and the probabilities built on it
    key = [0.0, 0.94, 0.78, 2.61, 0.98, 0.68, 0.91, 2.26, 0.82, 0.41, 1.36,  # learned weights, margins 0-21
           0.75, 0.47, 0.62, 1.4, 0.58, 0.54, 1.33, 0.94, 0.67, 0.68, 1.57]
    j, pm = lines.margin_pmf(-2.0, 15.7, key)
    check(abs(pm.sum() - 1) < 1e-9 and pm[j == 0][0] == 0.0, "margin distribution sums to 1 and has no ties")
    check(pm[j == 3][0] > 2 * pm[j == 4][0] and pm[j == 7][0] > 2 * pm[j == 8][0], "3 and 7 are much likelier than 4 and 8")
    w3, u3 = lines._cover_prob("spread", "home", -3.0, -2.0, 15.7, 1.0, key=key)
    w25, u25 = lines._cover_prob("spread", "home", -2.5, -2.0, 15.7, 1.0, ref=-3.0, key=key)
    w35, _ = lines._cover_prob("spread", "home", -3.5, -2.0, 15.7, 1.0, ref=-3.0, key=key)
    check(u3 > 0.045 and u25 == 0, "a whole-number 3 carries a real push chance; a half point none")
    check(abs(w25 - (w3 + u3)) < 1e-9 and abs(w35 - w3) < 1e-9, "the hook turns pushes into wins, nothing else")
    for sh in (1.0, 0.24):  # no edge -> win and loss equally likely, so no positive value at -110
        pw, pp = lines._cover_prob("spread", "home", -7.0, -7.0, 15.7, sh, key=key)
        check(abs(pw - (1 - pw - pp)) < 1e-9 and lines.ev(pw, pp, -110) < 0,
              f"no edge on a key number is never a positive bet (calibration {sh})")
    pa, _ = lines._cover_prob("spread", "away", 7.0, -14.0, 15.7, 0.24, key=key)
    ph, _ = lines._cover_prob("spread", "home", -7.0, -14.0, 15.7, 0.24, key=key)
    check(ph > 0.5 > pa, "a 7-point edge favors the model's side, calibrated or not")
    # The probabilities fit together at every line a book could hang, on both sides, calibrated or not:
    # win + push + loss = 1, more points never lowers the chance of winning, half a point off a whole
    # number turns exactly that number's pushes into wins (3, 7, 10, 14, ...), and one side's win is the
    # other side's loss.
    bad = []
    for ms in (-24.0, -10.5, -3.0, 0.0, 2.5, 7.0, 17.0):           # model spread (home view)
        for R in (ms - 7.5, ms - 3.0, ms, ms + 3.5, ms + 9.0):      # consensus line (home view)
            for sh in (1.0, 0.25):
                for side, ref in (("home", R), ("away", -R)):
                    Ls = [ref + 0.5 * i for i in range(-14, 15)]
                    pr = {L: lines._cover_prob("spread", side, L, ms, 15.7, sh, ref=ref, key=key) for L in Ls}
                    for L in Ls:
                        w, u = pr[L]
                        if not (0 <= w and 0 <= u and w + u <= 1 + 1e-12):
                            bad.append(("range", ms, R, sh, side, L))
                        if L + 0.5 in pr and (pr[L + 0.5][0] < w - 1e-12 or sum(pr[L + 0.5]) < w + u - 1e-12):
                            bad.append(("monotone", ms, R, sh, side, L))
                        if float(L).is_integer() and L + 0.5 in pr and L - 0.5 in pr:
                            if abs(pr[L + 0.5][0] - (w + u)) > 1e-9 or abs(pr[L - 0.5][0] - w) > 1e-9 or pr[L + 0.5][1] or pr[L - 0.5][1]:
                                bad.append(("hook", ms, R, sh, side, L))
                    if side == "home":
                        for L in Ls:
                            wa, ua = lines._cover_prob("spread", "away", -L, ms, 15.7, sh, ref=-R, key=key)
                            if abs(pr[L][0] + pr[L][1] + wa - 1) > 1e-9 or abs(ua - pr[L][1]) > 1e-12:
                                bad.append(("sides", ms, R, sh, L))
    for mt in (41.0, 55.5, 68.0):  # totals: same checks (over gets more room as the line rises)
        for R in (mt - 8.0, mt, mt + 6.5):
            for sh in (1.0, 0.18):
                Ls = [R + 0.5 * i for i in range(-14, 15)]
                po = {L: lines._cover_prob("total", "over", L, mt, 16.0, sh, ref=R) for L in Ls}
                for L in Ls:
                    w, u = po[L]
                    wu, uu = lines._cover_prob("total", "under", L, mt, 16.0, sh, ref=R)
                    if not (0 <= w and w + u <= 1 + 1e-12) or abs(w + u + wu - 1) > 1e-9 or abs(uu - u) > 1e-12:
                        bad.append(("total", mt, R, sh, L))
                    if L + 0.5 in po and po[L + 0.5][0] > w + 1e-12:
                        bad.append(("total monotone", mt, R, sh, L))
    check(not bad, f"probabilities fit together at every line (sum to 1, monotone, hooks, both sides){': ' + str(bad[:3]) if bad else ''}")
    j, pm = lines.margin_pmf(0.0, 15.7, key)
    pu = {k: pm[j == k][0] + pm[j == -k][0] for k in range(1, 16)}
    check(pu[3] > pu[2] and pu[3] > pu[4] and pu[7] > pu[6] and pu[7] > pu[8] and pu[10] > pu[9] and pu[14] > pu[13] and pu[14] > pu[15],
          "games end on 3, 7, 10 and 14 more often than the margins next to them")
    # the 7-point edge is inclusive and exact on half-point lines
    check(abs(-6.5 - (-13.5)) >= C.SPREAD_EDGE and abs(44.5 - 51.5) >= C.TOTAL_EDGE, "an exact 7.0 gap qualifies")
    check(not (abs(44.5 - 51.4) >= C.TOTAL_EDGE), "a 6.9 gap doesn't")


def test_cfbd_spellings():
    """CFBD has used camelCase (v2) and snake_case (v1) names; both must give the same lines."""
    import cfb_data as Dt
    camel = [dict(id=1, lines=[dict(provider="consensus", spread=-3.5, spreadOpen=-2.5, overUnder=51.5, overUnderOpen=50.0)])]
    snake = [dict(id=1, lines=[dict(provider="consensus", spread=-3.5, spread_open=-2.5, over_under=51.5, over_under_open=50.0)])]
    a, b = Dt.lines_table(camel).iloc[0], Dt.lines_table(snake).iloc[0]
    check(a.cfbd_spread_open == b.cfbd_spread_open == -2.5, "opening spread read from either spelling")
    check(a.cfbd_total == b.cfbd_total == 51.5 and a.cfbd_total_open == b.cfbd_total_open == 50.0, "totals read from either spelling")


def test_odds_matching():
    games = pd.DataFrame([dict(game_id=1, home="Texas", away="Florida", start=pd.Timestamp("2026-10-17 16:00", tz="UTC"),
                               completed=False, neutral=False),
                          dict(game_id=2, home="Georgia", away="Florida", start=pd.Timestamp("2026-11-01 19:30", tz="UTC"),
                               completed=False, neutral=True)])
    ev = [dict(home_team="Texas Longhorns", away_team="Florida Gators", commence_time="2026-10-17T16:00:00Z",
               bookmakers=[dict(title="Book", markets=[
                   dict(key="spreads", outcomes=[dict(name="Texas Longhorns", point=-10.5, price=-112),
                                                 dict(name="Florida Gators", point=10.5, price=-108)]),
                   dict(key="totals", outcomes=[dict(name="Over", point=52.5, price=-110), dict(name="Under", point=52.5, price=-110)])])]),
          # neutral site listed the other way round by the odds feed
          dict(home_team="Florida Gators", away_team="Georgia Bulldogs", commence_time="2026-11-01T19:30:00Z",
               bookmakers=[dict(title="Book", markets=[
                   dict(key="spreads", outcomes=[dict(name="Florida Gators", point=13.5, price=-110),
                                                 dict(name="Georgia Bulldogs", point=-13.5, price=-110)])])])]
    rows, unmatched = lines.snapshot_rows(ev, games, pd.Timestamp("2026-10-14", tz="UTC"))
    r1 = rows[rows.game_id == 1].iloc[0]
    r2 = rows[rows.game_id == 2].iloc[0]
    check(not unmatched, "both odds events matched to a game")
    check(r1.home_spread == -10.5 and r1.home_price == -112 and r1.away_price == -108, "spread and prices kept from the home side")
    check(r1.total == 52.5 and r1.over_price == -110, "total and over price kept")
    check(r2.home_spread == -13.5, "neutral site listed in reverse: spread still from the schedule's home team")


def test_record_keeping():
    b = report._book(["win", "loss", "loss", "push", "win"])
    check(abs(b["units"] - (2 * 100 / 110 - 2)) < 0.01, "units at -110")
    check(b["losing_streak"] == 2 and abs(b["max_drawdown"] - 2.0) < 1e-9, "losing streak and drawdown")
    x = pd.DataFrame(dict(season=[2023] * 40 + [2024] * 10, edge=[8] * 50, res=["win"] * 30 + ["loss"] * 10 + ["win"] * 6 + ["loss"] * 4))
    n = report._nested(x, (5, 7, 9))
    check([s["season"] for s in n["seasons"]] == [2024], "threshold test never scores the season it chose the rule on")
    w = report._wilson(55, 45)
    check(w[0] < 0.55 < w[1], "95% range contains the observed rate")


def test_keep_rule():
    """The feature keep rule: a real improvement passes, noise doesn't, and bets are graded the right way."""
    import cfb_model as model
    rng = np.random.default_rng(1)
    n = 4 * 15 * 50  # 4 seasons, 15 weeks, 50 games
    act = rng.normal(0, 16, n)
    base = pd.DataFrame(dict(season=np.repeat([2021, 2022, 2023, 2024], n // 4), wk=np.tile(np.repeat(np.arange(1, 16), 50), 4),
                             game_id=np.arange(n), home_pts=30 + act / 2, away_pts=30 - act / 2,
                             home_tier="P4", away_tier="P4"))
    ref = base.assign(pred_h=30 + (act + rng.normal(0, 14, n)) / 2, pred_a=30 - (act + rng.normal(0, 14, n)) / 2)
    ref["pred_a"] = 60 - ref.pred_h  # totals identical in every version, so only the margin differs
    better = base.assign(pred_h=ref.pred_h + (act - (ref.pred_h - ref.pred_a)) * 0.1 / 2)
    better["pred_a"] = 60 - better.pred_h
    noise = base.assign(pred_h=ref.pred_h + rng.normal(0, 0.3, n))
    noise["pred_a"] = 60 - noise.pred_h
    seasons = [2021, 2022, 2023, 2024]
    check(model._better(better, ref, seasons)[0], "keep rule: a real accuracy gain is kept")
    check(not model._better(noise, ref, seasons)[0], "keep rule: random noise is not kept")
    g = pd.DataFrame(dict(season=[2024, 2024], wk=[5, 5], game_id=[1, 2], home_pts=[30, 20], away_pts=[20, 30],
                          pred_h=[35, 35], pred_a=[20, 20], home_tier="P4", away_tier="G5"))
    ln = pd.DataFrame(dict(game_id=[1, 2], cfbd_spread_open=[-5.0, -5.0], cfbd_total_open=[45.0, 45.0]))
    u = model.open_units(g, ln, [2024])
    # game 1: model home by 15 vs line home -5 -> home bet, wins by 10 -> covers; total 55 vs 45 -> over, 50 -> wins
    # game 2: same bets, home loses by 10 -> spread loss; total 50 -> over wins
    check(u["w"] == 3 and u["l"] == 1, "bet check grades spreads and totals against opening lines correctly")


def test_no_leakage():
    """The backtest never sees the future: changing one week's results must leave every earlier
    projection exactly as it was (the weekly points-conversion refit is where a leak could hide)."""
    import cfb_model as model
    rng = np.random.default_rng(3)
    n = 3 * 6 * 60  # 3 seasons, 6 weeks, 60 games
    P = rng.normal(70, 6, (n, 2)); Q = rng.normal(0, 0.1, (n, 2)); S_ = rng.normal(0.42, 0.04, (n, 2))
    bt = pd.DataFrame(dict(season=np.repeat([2022, 2023, 2024], n // 3), wk=np.tile(np.repeat(np.arange(1, 7), 60), 3),
                           game_id=np.arange(n), home=[f"H{i}" for i in range(n)], away=[f"A{i}" for i in range(n)],
                           home_tier="P4", away_tier="P4", h=1.0,
                           p_h=P[:, 0], p_a=P[:, 1], q_h=Q[:, 0], q_a=Q[:, 1], s_h=S_[:, 0], s_a=S_[:, 1]))
    bt["home_pts"] = (28 + 0.25 * (bt.p_h - 70) + 60 * bt.q_h + rng.normal(0, 10, n)).round()
    bt["away_pts"] = (25 + 0.25 * (bt.p_a - 70) + 60 * bt.q_a + rng.normal(0, 10, n)).round()
    a = model.walk_forward(bt, 2023)
    bt2 = bt.copy()
    hit = (bt2.season == 2024) & (bt2.wk == 4)
    bt2.loc[hit, "home_pts"] = bt2.loc[hit, "home_pts"] + 21  # rewrite week 4's results
    b = model.walk_forward(bt2, 2023)
    key = ["game_id"]
    m = a.merge(b, on=key, suffixes=("", "_2"))
    early = (m.season < 2024) | (m.wk <= 4)
    check(len(m) == len(a) and np.allclose(m.loc[early, "pred_h"], m.loc[early, "pred_h_2"])
          and np.allclose(m.loc[early, "pred_a"], m.loc[early, "pred_a_2"]),
          "backtest: changing week 4's results leaves every projection up to week 4 unchanged")
    check(not np.allclose(m.loc[~early, "pred_h"], m.loc[~early, "pred_h_2"]),
          "backtest: later weeks do learn from week 4 (the test can see a change)")


def test_storage():
    """The bet store and line history survive a save and reload exactly, and finished weeks aren't rewritten."""
    import cfb_lines as L
    tmp = Path(tempfile.mkdtemp())
    old = (predict.STORE, predict.STORE_DIR, L.HIST, L.HIST_DIR)
    try:
        predict.STORE, predict.STORE_DIR = tmp / "predictions.json", tmp / "predictions"
        L.HIST, L.HIST_DIR = tmp / "line_history.csv", tmp / "lines"
        store = {"1": dict(start="2025-09-06T16:00:00+00:00", wk=2, home="A", away="B", first_spread_flag=dict(line=-3.5)),
                 "2": dict(start="2026-01-01T20:00:00+00:00", wk=21, home="C", away="D"),
                 "3": dict(start="2026-10-31T16:00:00+00:00", wk=10, home="E", away="F")}
        predict.save_store(store)
        check(predict.load_store() == store, "bet store: save and reload give back exactly the same records")
        check((predict.STORE_DIR / "2025" / "week-21.json").exists(), "bet store: a January bowl is filed under the season it belongs to")
        f = predict.STORE_DIR / "2025" / "week-02.json"
        before = f.stat().st_mtime_ns
        store["3"]["model_spread"] = -7.0
        predict.save_store(store)
        check(f.stat().st_mtime_ns == before, "bet store: a finished week's file isn't rewritten when another week changes")
        rows = pd.DataFrame(dict(ts=[pd.Timestamp("2026-10-27T14:00:00Z")] * 2, game_id=[1, 2], book=["X", "Y"],
                                 home_spread=[-3.5, 7.0], total=[50.5, 44.0]))
        for c in L.COLS:
            if c not in rows:
                rows[c] = np.nan
        h = L.append(rows)
        check(len(h) == 2 and set(h.game_id) == {1, 2}, "line history: a snapshot is saved and read back")
    finally:
        predict.STORE, predict.STORE_DIR, L.HIST, L.HIST_DIR = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_published():
    site = C.SITE
    def load(name, var):
        p = site / name
        if not p.exists():
            check(False, f"{name} exists")
            return None
        t = p.read_text()
        try:
            return json.loads(t[len(f"window.{var} = "):-1])
        except Exception as e:
            check(False, f"{name} is valid ({e})")
            return None
    D, M, H = load("data.js", "DATA"), load("model.js", "MODEL"), load("history.js", "HIST")
    if D is None:
        return
    check(isinstance(D.get("games"), list) and isinstance(D.get("teams"), list), "data.js has games and teams")
    check(H is None or isinstance(H, list), "history.js is a list of graded games")
    live = (D.get("report") or {}).get("live") or {}
    if H is not None and live.get("graded"):
        n = sum(1 for p in H if (p.get("result") or {}).get("ats")) + sum(1 for p in H if (p.get("result") or {}).get("ou"))
        s = sum((live.get(k) or {}).get(x, 0) for k in ("spread", "total") for x in ("w", "l", "p"))
        check(n == s, f"every graded bet in the game list adds up to the season totals ({n} vs {s})")
    ids = [g["game_id"] for g in D["games"]]
    check(len(ids) == len(set(ids)), "every game appears once on the projections page")
    if H is not None:
        hid = [p["game_id"] for p in H]
        check(len(hid) == len(set(hid)), "every graded game appears once in the history")
    dec = site / "decisions.csv"
    if dec.exists():
        try:
            d = pd.read_csv(dec)
            check({"decision_id", "ts", "game_id", "market", "status", "policy", "supersedes"} <= set(d.columns),
                  "decision log has its columns (id, policy, supersedes)")
            check(not d.duplicated(["game_id", "market", "status", "policy"]).any(), "decision log has no repeated entries")
            check(d.decision_id.notna().all() and not d.decision_id.duplicated().any(), "every decision has its own id")
            sup = d.supersedes.dropna().astype(str)
            check(sup[sup != ""].isin(set(d.decision_id.astype(str))).all(), "every superseded decision is in the log")
        except Exception as e:
            check(False, f"decision log readable ({e})")
    for g in D["games"]:
        if g.get("model_spread") is not None:
            ok = np.isfinite(g["model_spread"]) and 10 <= g["model_total"] <= 130
            if not ok:
                check(False, f"sensible projection for {g['away']} @ {g['home']}")
    node = shutil.which("node")
    if not node:
        print("skip  JavaScript checks (node not installed)")
        return
    for page in ("index", "matchup", "ratings", "report", "method"):
        p = site / f"{page}.html"
        if not p.exists():
            check(False, f"{page}.html exists")
            continue
        scripts = re.findall(r"<script>(.*?)</script>", p.read_text(), re.S)
        for i, js in enumerate(scripts):
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
                f.write(js)
            r = subprocess.run([node, "--check", f.name], capture_output=True, text=True)
            check(r.returncode == 0, f"{page}.html script {i + 1} has valid JavaScript" + ("" if r.returncode == 0 else f": {r.stderr.strip()[:200]}"))
    # The Matchup page's formula (run in node) must reproduce the weekly projections it shares ratings with.
    if M and M.get("conv"):
        js = re.findall(r"<script>(.*?)</script>", (site / "matchup.html").read_text(), re.S)[-1]
        fn = js[:js.index("\nconst MD=window.MODEL")]  # the shared model functions at the top of the page script
        games = [dict(home=g["home"], away=g["away"], neutral=g["neutral"], s=g["model_spread"], t=g["model_total"])
                 for g in D["games"] if g.get("model_spread") is not None and not g.get("completed")]
        prog = ("global.window={MODEL:" + json.dumps(M) + "};" + fn +
                ";const G=" + json.dumps(games) + ";let dm=0,dt=0,mx=0;for(const g of G){const p=project(g.home,g.away,g.neutral);"
                "const e=Math.abs(-g.s-(p.ph-p.pa));dm+=e;dt+=Math.abs(g.t-(p.ph+p.pa));mx=Math.max(mx,e);}"
                "console.log(JSON.stringify({n:G.length,dm:G.length?dm/G.length:0,dt:G.length?dt/G.length:0,mx}));")
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(prog)
        r = subprocess.run([node, f.name], capture_output=True, text=True)
        if r.returncode:
            check(False, f"matchup formula runs ({r.stderr.strip()[:200]})")
            return
        out = json.loads(r.stdout)
        if out["n"]:
            # Weekly projections can add manual adjustments, so allow a small average gap.
            check(out["dm"] < 1.0 and out["dt"] < 1.5,
                  f"matchup formula matches weekly projections (avg gap {out['dm']:.2f} pts margin, {out['dt']:.2f} total, {out['n']} games)")


if __name__ == "__main__":
    test_grading()
    test_best_line()
    test_prices_and_eligibility()
    test_cfbd_spellings()
    test_odds_matching()
    test_record_keeping()
    test_keep_rule()
    test_no_leakage()
    test_storage()
    test_published()
    if FAIL:
        print(f"\n{len(FAIL)} check(s) failed. Nothing should be published until they pass.")
        sys.exit(1)
    print("\nAll checks passed.")
