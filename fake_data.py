"""Generates fake API responses in CFBD / Odds API format so the whole
pipeline can be tested offline. Usage: python tests/fake_data.py <raw_dir>"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
confs = ["SEC", "Big Ten", "Big 12", "ACC", "American", "Mountain West", "Sun Belt", "MAC", "Conference USA"]
FBS = [f"Team{i:03d}" for i in range(120)]
FCS = [f"Fcs{i:02d}" for i in range(20)]
conf = {t: confs[i % len(confs)] for i, t in enumerate(FBS)}
strength_off = {t: rng.normal(0.0 if conf[t] in confs[:4] else -0.08, 0.12) for t in FBS}
strength_def = {t: rng.normal(0.0 if conf[t] in confs[:4] else 0.08, 0.10) for t in FBS}
for t in FCS:
    strength_off[t], strength_def[t] = rng.normal(-0.25, 0.08), rng.normal(0.25, 0.08)
pace = {t: rng.normal(0, 5) for t in FBS + FCS}
now = datetime(2026, 10, 7, 20, tzinfo=timezone.utc)
gid = 100000
venues = []
for i, t in enumerate(FBS):
    venues.append(dict(id=i, name=f"{t} Stadium", latitude=30 + i % 15, longitude=-100 + i % 20, dome=(i % 17 == 0)))
(out / "venues.json").write_text(json.dumps(venues))


def score(e, d):
    p7 = max(e, 1) / d / 8.5
    n7 = rng.binomial(d, min(p7, 0.8))
    n3 = rng.binomial(d - n7, min(0.5 * p7 / (1 - p7), 0.9))
    return 7 * n7 + 3 * n3


odds = []
for season in range(2015, 2027):
    # Offseason: strengths regress, influenced by returning production and talent.
    ret = {t: float(rng.uniform(0.3, 0.9)) for t in FBS}
    talent = {t: 700 + 400 * (strength_off[t] - strength_def[t]) + rng.normal(0, 30) for t in FBS}
    if season > 2015:
        for t in FBS:
            strength_off[t] = 0.6 * strength_off[t] + 0.15 * (ret[t] - 0.6) + rng.normal(0, 0.06)
            strength_def[t] = 0.6 * strength_def[t] + rng.normal(0, 0.05)
    games, adv, lines = [], [], []
    start0 = datetime(season, 9, 1, 18, tzinfo=timezone.utc)
    for wk in range(1, 15):
        teams = FBS.copy()
        rng.shuffle(teams)
        pairs = [(teams[i], teams[i + 1]) for i in range(0, len(teams) - 1, 2)]
        if wk <= 3:
            pairs = pairs[:50] + [(teams[100 + k], FCS[k]) for k in range(min(20, len(teams) - 100))]
        for h, a in pairs:
            gid += 1
            st = start0 + timedelta(days=7 * (wk - 1), hours=int(rng.integers(0, 8)))
            neutral = rng.random() < 0.03
            hf = 0 if neutral else 0.03
            qh = 0.12 + strength_off[h] + strength_def[a] + hf
            qa = 0.12 + strength_off[a] + strength_def[h] - hf
            ph, pa = 66 + pace[h] + pace[a] / 2, 66 + pace[a] + pace[h] / 2
            eh, ea = ph * (0.42 + 0.9 * qh), pa * (0.42 + 0.9 * qa)
            done = st < now
            hp, ap = (score(eh, 13), score(ea, 13)) if done else (None, None)
            if done and hp == ap:
                hp += 3
            hcls = "fbs" if h in FBS else "fcs"
            acls = "fbs" if a in FBS else "fcs"
            games.append(dict(id=gid, season=season, week=wk, seasonType="regular", startDate=st.isoformat(),
                              neutralSite=bool(neutral), venueId=FBS.index(h) if h in FBS else None,
                              homeTeam=h, homeConference=conf.get(h, "FCS"), homeClassification=hcls, homePoints=hp,
                              awayTeam=a, awayConference=conf.get(a, "FCS"), awayClassification=acls, awayPoints=ap,
                              completed=bool(done)))
            if done:
                for t, o, q, p in ((h, a, qh, ph), (a, h, qa, pa)):
                    pl = int(p + rng.normal(0, 6))
                    adv.append(dict(gameId=gid, season=season, week=wk, team=t, opponent=o,
                                    offense=dict(ppa=q + rng.normal(0, 0.15), plays=pl, drives=int(pl / 5.5))))
            mk = round((-(eh - ea) + rng.normal(0, 2.5)) * 2) / 2
            mt = round((eh + ea + rng.normal(0, 3)) * 2) / 2
            lines.append(dict(id=gid, homeTeam=h, awayTeam=a, lines=[
                dict(provider="consensus", spread=mk, spreadOpen=mk + rng.choice([-1, -0.5, 0, 0.5, 1]),
                     overUnder=mt, overUnderOpen=mt + rng.choice([-1, 0, 1]))]))
            if not done and st < now + timedelta(days=8):
                books = []
                for b in ("DraftKings", "FanDuel", "BetMGM"):
                    s2 = mk + rng.choice([0, 0.5, -0.5])
                    books.append(dict(key=b.lower(), title=b, markets=[
                        dict(key="spreads", outcomes=[dict(name=f"{h} Tigers", point=s2, price=-110),
                                                      dict(name=f"{a} Tigers", point=-s2, price=-110)]),
                        dict(key="totals", outcomes=[dict(name="Over", point=mt, price=-110),
                                                     dict(name="Under", point=mt, price=-110)])]))
                odds.append(dict(id=str(gid), commence_time=st.isoformat(), home_team=f"{h} Tigers",
                                 away_team=f"{a} Tigers", bookmakers=books))
    (out / f"games_{season}.json").write_text(json.dumps(games))
    (out / f"adv_{season}.json").write_text(json.dumps(adv))
    (out / f"lines_{season}.json").write_text(json.dumps(lines))
    (out / f"talent_{season}.json").write_text(json.dumps([dict(year=season, team=t, talent=v) for t, v in talent.items()]))
    (out / f"returning_{season}.json").write_text(json.dumps([dict(season=season, team=t, percentPPA=v) for t, v in ret.items()]))
(out / "odds_latest.json").write_text(json.dumps(odds))
print("fake data written to", out)
