"""Line tracking. Odds API snapshots are matched to CFBD games and appended to
data/line_history.csv. Lines are only ever used for display and grading,
never as model inputs."""
import math
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
    rows, unmatched, odd = [], [], []
    for ev in odds:
        h, a = find(ev.get("home_team")), find(ev.get("away_team"))
        t0 = pd.to_datetime(ev.get("commence_time"), utc=True)
        cand = up[((up.home == h) & (up.away == a)) | ((up.home == a) & (up.away == h))]
        cand = cand[(cand.start - t0).abs() < pd.Timedelta(days=2)]
        if cand.empty:
            unmatched.append(f"{ev.get('away_team')} @ {ev.get('home_team')}")
            continue
        if len(cand) > 1:  # same teams twice within two days: take the closest kickoff, and say so
            cand = cand.iloc[(cand.start - t0).abs().argsort()]
            odd.append(f"{ev.get('away_team')} @ {ev.get('home_team')}: two possible games, used the closest kickoff")
        g = cand.iloc[0]
        flip = g.home != h  # Odds API's "home" can differ at neutral sites
        if abs((g.start - t0).total_seconds()) > 3 * 3600:
            odd.append(f"{g.away} @ {g.home}: kickoff differs by {abs((g.start - t0).total_seconds()) / 3600:.0f} h between sources"
                       " (usually a kickoff time not announced yet; same teams within 2 days, so the match is safe)")
        if flip and not g.neutral:
            odd.append(f"{g.away} @ {g.home}: odds feed lists the teams the other way round at a home game")
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
    out = pd.DataFrame(rows, columns=COLS)
    out.attrs["oddities"] = odd  # shown under Data health; the lines are still used
    return out, unmatched


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


def ev(p_win, p_push=0.0, price=None):
    """Expected profit per 1 unit risked: win pays at the American price (missing: -110), push refunds."""
    p_push = p_push or 0.0
    return p_win * _payout(price) - (1 - p_win - p_push)


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _edge(kind, side, line, model):
    """Points the side gets beyond the model's number. Spread lines are from the side's view;
    model is the home spread (negative = home favored)."""
    if kind == "spread":
        return (line - model) if side == "home" else (line + model)
    return (model - line) if side == "over" else (line - model)


def _cover_prob(kind, side, line, model, sd, shrink, ref=None):
    """Chance the side wins at this line, and the chance of a push on a whole number.
    The level is calibrated (shrunk toward 50%, as the model's edges are) at the reference line;
    moving between lines uses the full outcome distribution, because the value of an extra half
    point depends on how often games land there, not on the model's edge."""
    ref = line if ref is None else ref
    zr, z = _edge(kind, side, ref, model) / sd, _edge(kind, side, line, model) / sd
    push = (math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * sd)) if float(line).is_integer() else 0.0
    p = 0.5 + shrink * (_phi(zr) - 0.5) + (_phi(z) - _phi(zr)) - push / 2
    return min(max(p, 0.0), 1.0 - push), push


def best_line(books, kind, side, model=None, sd=16.0, shrink=1.0):
    """The single book that pays best for one side. With the model's number, books are ranked by
    expected value (line and price together, so half a point at -125 doesn't beat -110 automatically);
    without it, by the most points with price as the tiebreak."""
    best = None
    vals = [((b.get("spread") if side == "home" else -b.get("spread")) if kind == "spread" else b.get("total"))
            for b in books or [] if b.get("spread" if kind == "spread" else "total") is not None]
    ref = float(np.median(vals)) if vals else None
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
        price = None if price is None or price != price else int(price)
        if model is not None:
            pw, pp = _cover_prob(kind, side, line, model, sd, shrink, ref)
            key = (ev(pw, pp, price), val)
        else:
            pw = pp = None
            key = (val, _payout(price))
        if best is None or key > best[0]:
            best = (key, dict(book=b.get("book"), line=line, price=price,
                              **({} if pw is None else dict(p=round(pw, 3), ev=round(key[0], 3)))))
    return best[1] if best else None


def _f(x):
    try:
        x = float(x)
        return None if x != x else x
    except (TypeError, ValueError):
        return None
