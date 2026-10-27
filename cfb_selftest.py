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
    # the 7-point edge is inclusive and exact on half-point lines
    check(abs(-6.5 - (-13.5)) >= C.SPREAD_EDGE and abs(44.5 - 51.5) >= C.TOTAL_EDGE, "an exact 7.0 gap qualifies")
    check(not (abs(44.5 - 51.4) >= C.TOTAL_EDGE), "a 6.9 gap doesn't")


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
            check({"ts", "game_id", "market", "status"} <= set(d.columns), "decision log has its columns")
            check(not d.duplicated(["game_id", "market", "status"]).any(), "decision log has no repeated entries")
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
    test_odds_matching()
    test_record_keeping()
    test_keep_rule()
    test_storage()
    test_published()
    if FAIL:
        print(f"\n{len(FAIL)} check(s) failed. Nothing should be published until they pass.")
        sys.exit(1)
    print("\nAll checks passed.")
