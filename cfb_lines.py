"""Line tracking. Odds API snapshots are matched to CFBD games and appended to
data/line_history.csv. Lines are only ever used for display and grading,
never as model inputs."""
import re
import unicodedata

import numpy as np
import pandas as pd

import cfb_config as C

HIST = C.DATA / "line_history.csv"
# Prices are American odds for each side (e.g. -110), kept so results can later be graded at real prices.
COLS = ["ts", "game_id", "book", "home_spread", "total", "home_price", "away_price", "over_price", "under_price"]

# Odds API full names that don't start with the CFBD school name.
ALIASES = {
    "southern mississippi": "southern miss",
    "louisiana monroe": "ul monroe",
    "ul lafayette": "louisiana",
    "louisiana lafayette": "louisiana",
    "umass": "massachusetts",
    "massachusetts": "umass",
    "connecticut": "uconn",
    "appalachian state": "app state",
    "sam houston state": "sam houston",
    "san jose st": "san jose state",
    "north carolina state": "nc state",
    "fiu": "florida international",
}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", "").replace("'", "").replace(".", "")
    s = re.sub(r"[()\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def matcher(team_names):
    schools = sorted({norm(t): t for t in team_names}.items(), key=lambda x: -len(x[0]))

    def find(full):
        n0 = norm(full)
        tries = [n0]
        for a, b in ALIASES.items():
            if n0.startswith(a + " ") or n0 == a:
                tries.insert(0, b + n0[len(a):])
                break
        for n in tries:
            for key, orig in schools:
                if n == key or n.startswith(key + " "):
                    return orig
        return None

    return find


def snapshot_rows(odds, games, ts):
    """Turn one Odds API response into rows keyed by CFBD game_id."""
    if not odds or games.empty:
        return pd.DataFrame(columns=COLS), []
    find = matcher(set(games.home) | set(games.away))
    up = games[~games.completed]
    rows, unmatched = [], []
    for ev in odds:
        h, a = find(ev.get("home_team")), find(ev.get("away_team"))
        t0 = pd.to_datetime(ev.get("commence_time"), utc=True)
        cand = up[((up.home == h) & (up.away == a)) | ((up.home == a) & (up.away == h))]
        cand = cand[(cand.start - t0).abs() < pd.Timedelta(days=2)]
        if cand.empty:
            unmatched.append(f"{ev.get('away_team')} @ {ev.get('home_team')}")
            continue
        g = cand.iloc[0]
        flip = g.home != h  # Odds API's "home" can differ at neutral sites
        for bk in ev.get("bookmakers", []):
            sp = tot = hp = ap = op = up_ = np.nan
            for mk in bk.get("markets", []):
                if mk.get("key") == "spreads":
                    for o in mk.get("outcomes", []):
                        if find(o.get("name")) == g.home:
                            sp = o.get("point", np.nan)
                            hp = o.get("price", np.nan)
                        elif find(o.get("name")) == g.away:
                            ap = o.get("price", np.nan)
                    if sp != sp:
                        for o in mk.get("outcomes", []):
                            if find(o.get("name")) == g.away and o.get("point") is not None:
                                sp = -o["point"]
                elif mk.get("key") == "totals":
                    for o in mk.get("outcomes", []):
                        if o.get("name") == "Over":
                            tot = o.get("point", np.nan)
                            op = o.get("price", np.nan)
                        elif o.get("name") == "Under":
                            up_ = o.get("price", np.nan)
            if sp == sp or tot == tot:
                rows.append(dict(ts=ts, game_id=int(g.game_id), book=bk.get("title", bk.get("key")),
                                 home_spread=sp, total=tot, home_price=hp, away_price=ap, over_price=op, under_price=up_))
    return pd.DataFrame(rows, columns=COLS), unmatched


def load_history():
    if HIST.exists():
        h = pd.read_csv(HIST)
        h["ts"] = pd.to_datetime(h.ts, utc=True, format="ISO8601")
        for c in COLS:  # files written before prices were tracked
            if c not in h:
                h[c] = np.nan
        return h
    return pd.DataFrame(columns=COLS)


def append(rows):
    h = load_history()
    if len(rows):
        h = pd.concat([h, rows], ignore_index=True)
        h["ts"] = pd.to_datetime(h.ts, utc=True, format="ISO8601")
        h.to_csv(HIST, index=False)
    return h


def market_view(hist, games):
    """Per game: opening and current consensus (median across books), books' latest lines,
    and the closing line (last snapshot before kickoff)."""
    out = {}
    if hist.empty:
        return out
    starts = games.set_index("game_id").start.to_dict()
    for gid, grp in hist.groupby("game_id"):
        st = starts.get(gid)
        pre = grp[grp.ts < st] if st is not None and st == st else grp
        if pre.empty:
            continue  # only in-game quotes: never treat those as the opening or closing line
        snaps = pre.groupby("ts").agg(spread=("home_spread", "median"), total=("total", "median")).sort_index()
        # Consensus = median of the books, rounded to the half point books actually use.
        snaps = (np.floor(snaps * 2 + 0.5) / 2)
        latest = pre[pre.ts == pre.ts.max()]
        out[int(gid)] = dict(
            open_spread=_f(snaps.spread.dropna().iloc[0]) if snaps.spread.notna().any() else None,
            open_total=_f(snaps.total.dropna().iloc[0]) if snaps.total.notna().any() else None,
            spread=_f(snaps.spread.dropna().iloc[-1]) if snaps.spread.notna().any() else None,
            total=_f(snaps.total.dropna().iloc[-1]) if snaps.total.notna().any() else None,
            books=[dict(book=r.book, spread=_f(r.home_spread), total=_f(r.total),
                        home_price=_f(getattr(r, "home_price", None)), away_price=_f(getattr(r, "away_price", None)),
                        over_price=_f(getattr(r, "over_price", None)), under_price=_f(getattr(r, "under_price", None)))
                   for r in latest.itertuples()],
            first_ts=snaps.index[0].isoformat(), last_ts=snaps.index[-1].isoformat(),
            history=[dict(ts=t.isoformat(), spread=_f(r.spread), total=_f(r.total)) for t, r in snaps.iterrows()],
        )
    return out


def _payout(price):
    """Profit per 1 unit risked at American odds (missing price: assume -110)."""
    p = price if price is not None and price == price else -110
    return p / 100 if p > 0 else 100 / -p


def best_line(books, kind, side):
    """The single book offering the most for one side: most points on a spread, the lowest total
    for an over, the highest for an under; ties go to the better price."""
    best = None
    for b in books or []:
        if kind == "spread":
            hs = b.get("spread")
            if hs is None:
                continue
            line = hs if side == "home" else -hs
            price = b.get("home_price" if side == "home" else "away_price")
            val = line
        else:
            line = b.get("total")
            if line is None:
                continue
            price = b.get("over_price" if side == "over" else "under_price")
            val = -line if side == "over" else line
        key = (val, _payout(price))
        if best is None or key > best[0]:
            best = (key, dict(book=b.get("book"), line=line, price=None if price is None or price != price else int(price)))
    return best[1] if best else None


def _f(x):
    try:
        x = float(x)
        return None if x != x else x
    except (TypeError, ValueError):
        return None
