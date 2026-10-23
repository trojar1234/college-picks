"""Builds the static website in docs/ (served by GitHub Pages)."""
import json

import numpy as np
import pandas as pd

import cfb_config as C
from cfb_report import season_of


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if o != o else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


def team_table(final, state=None):
    """Every FBS team: the model's power rating (projected margin against an average FBS
    team on a neutral field, using the same ratings and formula as the game predictions),
    plus the per-play efficiency ratings behind it."""
    import cfb_model as M
    import cfb_ratings as Rt
    Rp, Rc = final["ppa"], final["pace"]
    teams = [t for t, tier in final["tiers"].items() if tier != "FCS" and t in Rp["off"]]
    rows = []
    for t in teams:
        rows.append(dict(team=t, tier=final["tiers"][t], off=Rp["off"][t], deff=Rp["deff"][t],
                         net=Rp["off"][t] - Rp["deff"][t], pace=Rc["mu"] + Rc["off"][t]))
    df = pd.DataFrame(rows)
    if state and state.get("conv") and len(df):
        feats = []
        for t in teams:
            f = dict(season=0, game_id=0, h=0)
            for key, m in M.FEAT_KEYS:
                R = final[m]
                # team's offense vs an average defense, and an average offense vs the team's defense
                f[f"{key}_h"] = R["mu"] + Rt._side(R, t, 0)
                f[f"{key}_a"] = R["mu"] + Rt._side(R, t, 1)
            feats.append(f)
        pr = M.predict_bt(pd.DataFrame(feats), state["conv"], state.get("groups") or M.BASE_GROUPS, state.get("total_model"))
        df["power"] = (pr.pred_h - pr.pred_a).to_numpy()
        df["pf"] = pr.pred_h.to_numpy()
        df["pa"] = pr.pred_a.to_numpy()
        df["power"] -= df["power"].mean()  # 0 = average FBS team
        ac = state.get("academy") or {}
        if ac.get("n"):  # same service-academy correction as the game projections
            m = df.team.isin(ac.get("teams") or [])
            df.loc[m, "power"] += ac["margin"]
            df.loc[m, "pf"] += (ac["margin"] + ac["total"]) / 2
            df.loc[m, "pa"] += (ac["total"] - ac["margin"]) / 2
        df = df.sort_values("power", ascending=False)
    else:
        df = df.sort_values("net", ascending=False)
    df["off_rank"] = df.off.rank(ascending=False).astype(int)
    df["def_rank"] = df.deff.rank(ascending=True).astype(int)
    return df.to_dict("records")


def model_export(final, state):
    """Everything the browser needs to project any matchup with the same formula as the
    weekly predictions: each team's offense/defense rating for every metric the points
    conversion uses, league averages, home field, and the conversion itself."""
    import cfb_model as M
    import cfb_ratings as Rt
    if not state or not state.get("conv"):
        return None
    teams = sorted(final["tiers"])
    metrics = {}
    for key, m in M.FEAT_KEYS:
        R = final.get(m)
        if not R:
            continue
        metrics[key] = dict(mu=R["mu"], hfa=R["hfa"],
                            off={t: round(Rt._side(R, t, 0), 5) for t in teams},
                            deff={t: round(Rt._side(R, t, 1), 5) for t in teams})
    return dict(conv=state["conv"], groups=state.get("groups") or M.BASE_GROUPS,
                margin_sd=state.get("margin_sd"), total_sd=state.get("total_sd"),
                total_model=state.get("total_model"), metrics=metrics, academy=state.get("academy") or {},
                rating_tau2=state.get("rating_tau2"), lam=state.get("lam"),
                tiers={t: final["tiers"][t] for t in teams})


def schedule_export(games):
    """This season's games (FBS involved), compact, for team pages and records."""
    if games is None or not len(games):
        return []
    out = []
    for r in games.sort_values("start").itertuples():
        out.append([int(r.wk), r.start.isoformat(), r.home, r.away, bool(r.neutral),
                    None if r.home_pts != r.home_pts else int(r.home_pts),
                    None if r.away_pts != r.away_pts else int(r.away_pts), bool(r.completed)])
    return out


def conf_map(games):
    out = {}
    if games is None or not len(games):
        return out
    for r in games.sort_values("start").itertuples():
        if getattr(r, "home_conf", ""):
            out[r.home] = r.home_conf
        if getattr(r, "away_conf", ""):
            out[r.away] = r.away_conf
    return out


def _with_movement(now_rows, prev_rows):
    """Add last week's rank and rating so the ratings page can show movement."""
    if prev_rows:
        prev = {r["team"]: (i + 1, r.get("power")) for i, r in enumerate(prev_rows)}
        for i, r in enumerate(now_rows):
            p = prev.get(r["team"])
            if p:
                r["prev_rank"], r["prev_power"] = p
    return now_rows


HIST_KEYS = ("game_id", "start", "wk", "home", "away", "neutral", "home_tier", "away_tier", "model_spread", "model_total",
             "home_med", "away_med", "p_home_win", "p_side_cover", "p_side_total", "spread_side", "total_side",
             "spread_flag", "total_flag", "first_spread_flag", "first_total_flag", "version", "result")


def build(season, wk, slate, store, rep, final, now, unmatched, state=None, prev_final=None, lines_ts=None, season_games=None,
          rating_hist=None, health=None):
    games = []
    for r in slate.sort_values("start").itertuples():
        p = store.get(str(r.game_id))
        g = dict(game_id=int(r.game_id), start=r.start.isoformat(), home=r.home, away=r.away,
                 home_conf=getattr(r, "home_conf", "") or "", away_conf=getattr(r, "away_conf", "") or "",
                 neutral=bool(r.neutral), completed=bool(r.completed),
                 home_pts=None if r.home_pts != r.home_pts else int(r.home_pts),
                 away_pts=None if r.away_pts != r.away_pts else int(r.away_pts))
        if p:
            g.update({k: v for k, v in p.items() if k not in g or g[k] is None})
            g["started"] = pd.Timestamp(p["start"]) <= now
            res = p.get("result") or {}
            if res.get("home_pts") is not None:  # graded = final, even if the schedule feed lags
                g.update(completed=True, home_pts=int(res["home_pts"]), away_pts=int(res["away_pts"]))
            # a bet is a flag recorded by the same model version that made the current prediction
            for kind in ("spread", "total"):
                f = p.get(f"first_{kind}_flag")
                ok = f and f.get("line") is not None and f.get("version") is not None and f.get("version") == p.get("version")
                g[f"{kind}_bet"] = f if ok else None
        games.append(g)
    # Every graded game this season (no cap), trimmed to what the Performance page uses
    history = sorted(
        [{k: p[k] for k in HIST_KEYS if k in p} for p in store.values()
         if p.get("result") and p.get("start") and season_of(p["start"]) == season],
        key=lambda p: p["start"], reverse=True,
    )
    confs = conf_map(season_games)
    teams = _with_movement(team_table(final, state), team_table(prev_final, state) if prev_final else None)
    for t in teams:
        t["conf"] = confs.get(t["team"], "")
    payload = _clean(dict(
        generated=now.isoformat(), lines_updated=lines_ts, season=season, week=wk, games=games, report=rep,
        teams=teams, unmatched=unmatched, schedule=schedule_export(season_games),
        rating_hist=rating_hist or {}, health=health or {},
        thresholds=dict(spread=C.SPREAD_EDGE, total=C.TOTAL_EDGE),
    ))
    C.SITE.mkdir(exist_ok=True)
    (C.SITE / "data.js").write_text("window.DATA = " + json.dumps(payload, separators=(",", ":")) + ";")
    # The full season of graded games is only needed on the Performance page, so it loads separately
    (C.SITE / "history.js").write_text("window.HIST = " + json.dumps(_clean(history), separators=(",", ":")) + ";")
    mx = model_export(final, state)
    (C.SITE / "model.js").write_text("window.MODEL = " + json.dumps(_clean(dict(mx or {}, confs=confs)), separators=(",", ":")) + ";")
    # a new version tag each run so browsers never pair a fresh page with stale cached data
    ver = now.strftime("%Y%m%d%H%M%S")
    (C.SITE / "index.html").write_text(_page(INDEX_BODY, INDEX_JS, "Projections", ver))
    (C.SITE / "report.html").write_text(_page(REPORT_BODY.replace("__VER__", ver), REPORT_JS, "Performance", ver))
    (C.SITE / "ratings.html").write_text(_page(RATINGS_BODY.replace("__VER__", ver), RATINGS_JS, "Team ratings", ver))
    (C.SITE / "matchup.html").write_text(_page(MATCHUP_BODY.replace("__VER__", ver), MATCHUP_JS, "Matchup", ver))
    dec = C.DATA / "decisions.csv"
    if dec.exists():  # the append-only decision log, downloadable from the Performance page
        (C.SITE / "decisions.csv").write_text(dec.read_text())
    (C.SITE / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    (C.SITE / ".nojekyll").write_text("")


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<meta http-equiv="Cache-Control" content="no-cache">
<title>__TITLE__ · CFB model</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{color-scheme:dark;--bg:#0e131a;--card:#151c25;--soft:#1c2531;--text:#e7ecf1;--muted:#9aa6b2;--faint:#8593a1;--line:#263240;
--acc:#2dd4bf;--accbg:#0f3330;--good:#5eead4;--goodbg:#0f3330;--warn:#fbbf24;--warnbg:#3a2c0b;--bad:#f87171;--badbg:#3b1719;
--series:#139c8c;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}
.skip{position:absolute;left:-999px;top:8px;background:var(--acc);color:#04201c;padding:8px 12px;border-radius:8px;z-index:20;font-weight:600}.skip:focus{left:16px}
.tagline{font-size:12px;color:var(--faint);font-weight:500;margin-left:10px;letter-spacing:0}@media (max-width:720px){.tagline{display:none}}
.banner{background:var(--warnbg);color:var(--warn);border-radius:10px;padding:10px 14px;margin:0 0 14px;font-size:14px}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important;animation:none!important}}
@media (pointer:coarse){.chip,header nav a,input[type=search]{min-height:40px}}
@media print{:root{--bg:#fff;--card:#fff;--soft:#f2f4f6;--text:#111;--muted:#444;--faint:#555;--line:#ccc;--acc:#0b7d70;--good:#0b7d70;--bad:#b42318;--warn:#8a5a00;--series:#0b7d70}
 body:before,header nav,.secnav,.bar,#swap,.venue,.skip,.banner,#filters,.legend,a.lnk{display:none!important}.tbl{overflow:visible}details.help{display:none}body{font-size:12px}}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;font-feature-settings:"tnum" 1,"cv11" 1;-webkit-font-smoothing:antialiased}
body:before{content:"";position:fixed;inset:0 0 auto 0;height:260px;background:radial-gradient(800px 220px at 20% -40px,rgba(45,212,191,.08),transparent 70%);pointer-events:none;z-index:-1}
.wrap{max-width:1100px;margin:0 auto;padding:20px 16px 60px}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid var(--line)}
h1{font-size:22px;margin:0;font-weight:700;letter-spacing:-.02em}
h2{font-size:19px;margin:36px 0 10px;font-weight:650;letter-spacing:-.01em}
header nav{display:flex;gap:2px;background:var(--card);border:1px solid var(--line);border-radius:999px;padding:3px}
header nav a{color:var(--muted);text-decoration:none;font-size:14px;padding:6px 14px;border-radius:999px;white-space:nowrap}
header nav a:hover{color:var(--text)}
header nav a.on{color:var(--text);font-weight:600;background:var(--soft);box-shadow:inset 0 0 0 1px var(--line)}
.navs{display:none}@media (max-width:720px){header nav{width:100%}header nav a{flex:1;text-align:center;padding:6px 4px;font-size:13px}.navl{display:none}.navs{display:inline}}
.sub{color:var(--muted);font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px}
.card{background:linear-gradient(180deg,var(--soft),var(--card));border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.card .l{font-size:12px;color:var(--muted)}.card{min-width:0}.card .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.card .v .push{font-size:14px;color:var(--muted);font-weight:500}.card .v .nw{white-space:nowrap}
.rbar{display:inline-block;position:relative;width:70px;height:6px;background:var(--soft);border-radius:3px;vertical-align:middle;margin-left:6px}.rbar span{position:absolute;top:0;height:6px;border-radius:3px}.rbar .pos{background:var(--series)}.rbar .neg{background:var(--muted);opacity:.55}
@media (max-width:720px){.rbar{width:40px}}
a.lnk{color:var(--acc);text-decoration:none;font-size:13px}a.lnk:hover{text-decoration:underline}
.ln{line-height:1.5;min-height:21px;white-space:nowrap}.c-pick .ln{white-space:normal}
.hot{color:var(--acc);font-weight:600}.mid{font-weight:600}b.weak{color:var(--muted);font-weight:500}
.score{font-size:13px;margin-top:3px;color:var(--text)}.big{font-size:15px;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}.sm-inline{display:none}
h1{color:var(--text)}header h1:before{content:"";display:inline-block;width:12px;height:12px;border-radius:3px;background:linear-gradient(135deg,var(--acc),#0ea5e9);margin-right:10px;vertical-align:0}
.games th:nth-child(1){width:25%}.games th:nth-child(2){width:15%}.games th:nth-child(3),.games th:nth-child(4){width:16%}.games th:nth-child(5){width:6%}
.games th .when{display:block;font-weight:400}th{vertical-align:top}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12px;color:var(--muted);margin:0 0 10px;align-items:center}
.summary{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin-bottom:14px}
.summary h3{font-size:13px;font-weight:600;margin:0 0 8px}.summary .grp+.grp{margin-top:12px}
.sitem{display:grid;grid-template-columns:auto 1fr auto;gap:4px 10px;align-items:baseline;padding:6px 0;border-top:1px solid var(--line);cursor:pointer;font-size:14px}
.sitem:first-of-type{border-top:0}.sitem .g2{color:var(--muted);font-size:12px}.sitem .r{font-size:12px;color:var(--faint);text-align:right}
.sitem:hover .t{text-decoration:underline}
tr.day td{background:var(--soft);font-size:12px;font-weight:600;color:var(--muted);padding:7px 12px;letter-spacing:.02em}
.pick .t.weak{color:var(--muted)}
.help{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin:0 0 14px;font-size:13px;color:var(--muted)}
.help summary{cursor:pointer;color:var(--text);font-weight:600}.help p,.help li{margin:8px 0;line-height:1.5}.help ul{padding-left:18px;margin:6px 0}
.mpick{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end;margin-bottom:6px}.mpick label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:4px}
.mpick input{font:inherit;font-size:15px;padding:8px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--text);min-width:220px}
.venue{display:flex;gap:6px;flex-wrap:wrap}
.mcard{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px}
.mhead{display:grid;grid-template-columns:1fr auto 1fr;align-items:end;gap:10px}.mteam{font-size:15px;font-weight:600}.mscore{font-size:40px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.1}
.mvs{text-align:center;color:var(--muted);font-size:13px}.mat{font-size:18px;color:var(--faint)}.mcards{margin:16px 0 0;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}.mnote{font-size:13px;color:var(--muted);margin-top:12px;line-height:1.6}.mh{margin:18px 0 6px;font-size:14px}.mtry{margin-top:12px}a.chip{text-decoration:none;color:var(--text)}a.chip:hover{border-color:var(--acc)}.mtbl td:first-child{color:var(--muted)}.vtbl tr.on td{background:var(--soft)}.vtbl tr.on td:first-child{color:var(--text);font-weight:600}
.presets{display:flex;flex-wrap:wrap;gap:6px}.scen{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:10px 0 0;font-size:13px;color:var(--muted)}.scen input[type=range]{accent-color:var(--acc);width:180px}
.eq{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;margin:4px 0 8px}.eq>span{display:flex;flex-direction:column}.eq .op{font-size:18px;color:var(--faint)}.eq .when{display:block}
.wbar{display:flex;gap:2px;height:6px;margin:8px 0 4px}.wbar span{display:block;height:100%;border-radius:3px;background:var(--line)}.wbar span.fav{background:var(--series)}
#swap{align-self:flex-end;font-size:16px;padding:6px 12px}
@media (max-width:720px){.mpick input{min-width:0;width:100%}.mpick label{flex:1 1 100%}.mscore{font-size:32px}.mcards{grid-template-columns:1fr 1fr}.mcards .card:nth-child(3){grid-column:1/-1}}
th .sb{font:inherit;color:inherit;background:none;border:0;padding:0;cursor:pointer;text-align:left}th .sb:hover{color:var(--text);text-decoration:underline}th.on{color:var(--text)}th.on .sb:after{content:' ▾';font-size:10px}th[data-s='deff'].on .sb:after{content:' ▴'}
.def{font-size:13px;color:var(--muted);margin:0 0 10px}.rt th{white-space:normal;vertical-align:bottom}.rt th .when{display:block;font-weight:400}.mtbl th,.mtbl td{white-space:normal}@media (max-width:720px){.mtbl td,.mtbl th{padding:7px 6px}}.movers{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 10px}.rh{width:100%;max-width:320px;height:auto;display:block}
.crng{position:relative;display:inline-block;width:120px;height:8px;background:var(--soft);border-radius:4px;vertical-align:middle}.crng span{position:absolute;top:0;height:8px;background:var(--series);opacity:.45;border-radius:4px}.crng i{position:absolute;top:-3px;width:2px;height:14px;background:var(--text)}
.btt th{white-space:normal;vertical-align:bottom}.btt td{white-space:nowrap}
.method{font-size:12px;font-weight:500;color:var(--faint);margin-left:8px;letter-spacing:0;white-space:nowrap}
.score2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.scard{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}.scard h3{margin:0 0 6px;font-size:14px}.scard .big2{margin:2px 0 6px}
.okbox{font-size:13px;color:var(--muted);margin:10px 0 0}.okbox.bad{color:var(--warn)}
.secnav{position:sticky;top:0;z-index:5;background:var(--bg);display:flex;gap:4px;overflow-x:auto;white-space:nowrap;padding:8px 0;margin:-6px 0 10px;border-bottom:1px solid var(--line)}
.secnav a{color:var(--muted);text-decoration:none;font-size:13px;padding:4px 10px;border-radius:999px}.secnav a:hover{background:var(--soft);color:var(--text)}
h2[id]{scroll-margin-top:52px}.summary[id]{scroll-margin-top:52px}
.ci{display:block;font-size:12px;font-weight:500}.ci.warnc{color:var(--warn)}.ci.up{color:var(--good)}.ci.down{color:var(--bad)}
.chartgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.chart{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}.chart h4{margin:0 0 8px;font-size:13px;font-weight:600}.chart svg{width:100%;height:auto;display:block}
.glance{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}.gblock .l{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}.big2{font-size:26px;font-weight:700;letter-spacing:-.02em;margin:4px 0}.gtxt{font-size:13px;color:var(--muted);line-height:1.5}
.summary p.gl{margin:6px 0;font-size:14px;line-height:1.5}
tr.g.flash td{background:var(--accbg)}.card .v .when{display:block;font-size:12px}
.pick{display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin:2px 0}.pick .t{font-size:13px}.pick .e{font-size:12px;color:var(--faint)}
.bet{font-weight:600}.notes{font-size:12px;color:var(--muted);margin-top:8px;line-height:1.4}
.bar{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
.chip{font:inherit;font-size:13px;padding:5px 11px;border-radius:999px;border:1px solid var(--line);background:var(--card);color:var(--muted);cursor:pointer}
.chip:is(select){padding-right:8px}
.chip.on{background:var(--accbg);color:var(--acc);border-color:transparent}
input[type=search]{font:inherit;font-size:14px;padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--text);min-width:180px;flex:1;max-width:260px}
.tbl{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{font-size:12px;font-weight:500;color:var(--muted);text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:10px;border-bottom:1px solid var(--line);font-size:14px;vertical-align:top}
tr.g{cursor:pointer}tr.g:hover td{background:var(--soft)}tr.g:focus-visible td,.sitem:focus-visible{outline:2px solid var(--acc);outline-offset:-2px}
tr:last-child td{border-bottom:0}
.home{font-weight:600}.when{font-size:12px;color:var(--faint)}
.pill{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;white-space:nowrap;margin:1px 0}
.p-good{background:var(--goodbg);color:var(--good)}.p-warn{background:var(--warnbg);color:var(--warn)}
.p-bad{background:var(--badbg);color:var(--bad)}.p-mv{background:var(--soft);color:var(--muted);border:1px solid var(--line)}.p-acc{background:var(--accbg);color:var(--acc)}
.muted{color:var(--faint)}.up{color:var(--good)}.down{color:var(--bad)}
.detail td{background:var(--soft);padding:14px 16px}
.dgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:18px}
.dgrid h4{margin:0 0 6px;font-size:12px;color:var(--muted);font-weight:500}
.kv{display:flex;justify-content:space-between;font-size:13px;padding:2px 0}
.hist{display:flex;align-items:flex-end;gap:2px;height:60px}
.hist div{flex:1;background:var(--acc);opacity:.75;border-radius:2px 2px 0 0;min-height:1px}
.hist div.z{opacity:.35}
.spark{margin-top:10px}.vsl{font-size:13px;padding:3px 0;line-height:1.45}.spark svg{width:100%;max-width:300px;height:auto;display:block}
.note{font-size:12px;color:var(--muted);margin-top:10px}
.flag{color:var(--bad);font-weight:600}
@media (max-width:720px){.hide-sm{display:none}td,th{padding:8px 6px;font-size:13px}.card .v{font-size:19px}}
.show-sm{display:none}@media (max-width:720px){.show-sm{display:block}}
.nw{white-space:nowrap}

@media (max-width:720px){table.games thead{display:none}table.games tr.g{display:grid;grid-template-columns:1fr auto;border-bottom:1px solid var(--line)}
table.games tr.g td{border:0;padding:10px 12px 2px}table.games td.c-proj{text-align:right}table.games td.c-pick{grid-column:1/-1;padding:4px 12px 12px}.sm-inline{display:inline}
#cards{grid-template-columns:repeat(3,1fr);gap:8px}#cards .card{padding:8px 10px}#cards .card .v{font-size:17px}#cards .card .l{font-size:12px}table.games tr.g td.hide-sm{display:none}table.games tr.detail td{display:block}table.games tr.day td{display:block}.sitem{grid-template-columns:auto 1fr}.sitem .r{grid-column:1/-1;text-align:left}}
</style></head><body><div class="wrap">
__BODY__
<p class="note" id="foot"></p>
</div>
<script src="data.js?v=__VER__"></script>
<script>
const D=window.DATA;
if(!D){document.querySelector('.wrap').insertAdjacentHTML('afterbegin','<div class="summary"><b>Data unavailable.</b> The latest update did not publish its data file. It should be back after the next scheduled run.</div>');throw new Error('data.js missing');}
const fmt=x=>x==null?'—':(Math.round(x*10)/10).toString();
const sgn=x=>{if(x==null)return '—';const r=Math.round(x*10)/10;return r===0?'0':(r>0?'+':'−')+Math.abs(r);};
const f2=x=>x==null?'—':x.toFixed(2);
const pct=x=>x==null?'—':Math.round(x*100)+'%';
const pct1=x=>x==null?'—':(x*100).toFixed(1)+'%';
const epa=x=>x==null?'—':(x>0?'+':x<0?'−':'')+Math.abs(x).toFixed(3);
const rec=r=>!r||(r.w+r.l+r.p)===0?'—':`<span class="nw">${r.w}–${r.l}</span>${r.p?`<wbr><span class="push nw">–${r.p}</span>`:''}`;
const sl=x=>x==null?'—':(x===0?'PK':(x>0?'+':'−')+Math.abs(x));
// The model's pick against a line, for every game. Spread lines are from the home team's view.
function spreadPick(g,lineVal,model){if(lineVal==null||model==null)return null;const e=lineVal-model;if(Math.abs(e)<0.05)return {none:true};
 const home=e>0;return {team:home?g.home:g.away,line:home?lineVal:-lineVal,side:home?'home':'away',edge:Math.abs(e)};}
function totalPick(lineVal,model){if(lineVal==null||model==null)return null;const e=model-lineVal;if(Math.abs(e)<0.05)return {none:true};
 return {side:e>0?'Over':'Under',line:lineVal,edge:Math.abs(e)};}
// Projected score that matches the model's spread and total exactly (away–home)
function projScore(g){const S=g.model_spread,T=g.model_total;if(S==null||T==null)return g.home_med==null?null:`${g.away_med}–${g.home_med}`;
 const h=(T-S)/2,a=(T+S)/2;let H=Math.round(h),A=Math.round(a);
 if(H===A&&Math.abs(S)>=0.05){if(S<0){H=Math.ceil(h);A=Math.floor(a);}else{H=Math.floor(h);A=Math.ceil(a);}if(H===A){if(S<0)H++;else A++;}}
 return `${A}–${H}`;}
const resPill=(v,label)=>v?`<span class="pill ${v==='win'?'p-good':v==='loss'?'p-bad':'p-acc'}">${label?label+' ':''}${v==='win'?'W':v==='loss'?'L':'Push'}</span>`:'';
const when=s=>new Date(s).toLocaleString('en-US',{timeZone:'America/Chicago',weekday:'short',hour:'numeric',minute:'2-digit'});
function line(home,away,s){if(s==null)return '—';if(s===0)return 'Pick';return s<0?`${home} −${Math.abs(s)}`:`${away} −${s}`;}
const wkName=w=>w==null?'—':(w>20?'Bowls':`Week ${w}`);
const gamesPlayed=t=>(D.schedule||[]).filter(r=>r[7]&&r[5]!=null&&(r[2]===t||r[3]===t)).length;
// 80% range of a team's rating: tau2 learned from how much bigger early-season misses are (see cfb_model.rating_uncertainty)
const ratingPM=t=>{const M=window.MODEL||{};if(!M.rating_tau2)return null;return 1.2816*Math.sqrt(M.rating_tau2/(gamesPlayed(t)+(M.lam||3)));};
const teamRecord=t=>{let w=0,l=0;for(const r of (D.schedule||[])){if(!r[7]||r[5]==null)continue;if(r[2]===t){r[5]>r[6]?w++:l++;}else if(r[3]===t){r[6]>r[5]?w++:l++;}}return `${w}–${l}`;};
// What the ratings reflect: preseason, or games through the latest completed week
const ratedThrough=()=>{const w=Math.max(0,...(D.schedule||[]).filter(r=>r[7]).map(r=>r[0]));return w?`ratings through ${wkName(w)} games`:'preseason ratings';};
const am=x=>x==null?'':(x>0?'+':'−')+Math.abs(Math.round(x));
// Freshness, shown on every page: when the model last ran and when the newest lines were fetched
const freshTxt=()=>{const f=t=>new Date(t).toLocaleString('en-US',{timeZone:'America/Chicago',weekday:'short',hour:'numeric',minute:'2-digit'});return `model ${f(D.generated)} CT`+(D.lines_updated?` · lines ${f(D.lines_updated)} CT`:'');};
const RM=window.matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('foot').innerHTML=`Model run ${new Date(D.generated).toLocaleString('en-US',{timeZone:'America/Chicago',dateStyle:'medium',timeStyle:'short'})} CT · Data: CollegeFootballData.com (games, stats), The Odds API (lines), Open-Meteo (weather). Sportsbook lines are only used to grade and display picks, never as model inputs. For information only, not betting advice.`+(D.unmatched&&D.unmatched.length?` · Unmatched odds games: ${D.unmatched.length}`:'')+((D.health||{}).model_version?` · Model v${D.health.model_version}`:'')+((D.health||{}).build?` · build ${esc(D.health.build)}`:'');
(()=>{const h=(Date.now()-new Date(D.generated))/36e5;if(h>36)document.querySelector('.wrap').insertAdjacentHTML('afterbegin',`<div class="banner" role="status">This page hasn't updated in ${Math.round(h)} hours. The automatic update may have failed; check the Actions tab on GitHub.</div>`);})();
</script>
<script>__SCRIPT__</script>
</body></html>"""

NAV = lambda on: f"""<a class="skip" href="#content">Skip to content</a><header><div><h1>College football model<span class="tagline">market-blind forecasts</span></h1><div class="sub" id="sub"></div></div>
<nav><a href="index.html" class="{'on' if on=='p' else ''}">Projections</a><a href="matchup.html" class="{'on' if on=='m' else ''}">Matchup</a><a href="ratings.html" class="{'on' if on=='t' else ''}"><span class="navl">Team ratings</span><span class="navs">Ratings</span></a><a href="report.html" class="{'on' if on=='r' else ''}"><span class="navl">Performance</span><span class="navs">Results</span></a></nav></header><div id="content" tabindex="-1"></div>"""

INDEX_BODY = NAV("p") + r"""
<div class="cards" id="cards"></div>
<details class="help"><summary>How to read this page</summary>
<ul>
<li><b>Prediction</b> is the model's projected score (away–home, rounded so it matches the model's spread and total) and the favorite's chance to win. Tap any game for the full breakdown: score distribution, likely range, inputs, every sportsbook's line, and weather.</li>
<li><b>Line</b> is the middle of the sportsbooks' lines; <b>Model</b> is the model's own number. The top number is the spread, the bottom the total. <b>Edge</b> is the gap between the two, in points.</li>
<li><b>Pick</b> is the model's side in every game. Most are close calls: picks with an edge under 2 points are shown in grey, 2+ in bold.</li>
<li><span class="pill p-good">BET</span> = the model and line are 7+ points apart and nothing makes the projection shaky. In past seasons these went <span id="h-rates">about 60% against the spread and 64% on totals</span> against opening lines. A bet keeps the line it was first flagged at.</li>
<li><span class="pill p-warn">FLAG</span> = 7+ points apart, but with a reason for caution: a QB change, a manual adjustment, wind or rain (the model doesn't use weather), an FCS team, early season (before the typical FBS team has played 3 games, about weeks 1–3), or a gap of 10+ points. A 10+ point gap usually means news the model can't see (an injury, a suspension, a QB change). In past seasons those gaps won about as often as other flags against opening lines but only about half the time against closing lines, so they're held for a news check instead of bet automatically (the Performance page tracks how each caution does). 8–10 point bets, and lines quoted by fewer than 3 books, get a reminder to check.</li>
<li><span class="pill p-mv">LINE MOVED</span> / <span class="pill p-mv">MODEL MOVED</span> = a recorded bet whose edge today is under 7 points, because the line moved or the model's number changed (QB news, a correction). It's still graded at the original line, but it isn't a bet at today's number.</li>
<li>A bet needs a live sportsbook quote from the last 36 hours. When a game only has the backup line from CollegeFootballData, a 7-point gap shows as a FLAG instead.</li>
<li><b>EV</b> is the expected profit per unit at −110 for the model's side, using probabilities calibrated on every past game. It's deliberately conservative, so small positive numbers are normal; the best-book suggestion is chosen by the same measure (line and price together).</li>
<li>The model's edge is against early-week lines (about 50% against closing lines), so bets are worth acting on early. Each bet shows the <b>best available book</b> for that side; results are still graded at the middle of the books, which is the conservative choice. Sportsbook lines are never model inputs.</li>
</ul></details>
<div id="summary"></div>
<div class="bar" id="filters"></div>
<div class="legend"><button class="chip" id="wcsv" style="order:9">Download this week (CSV)</button><span><span class="pill p-good">BET</span> clean 7+ pt edge</span><span><span class="pill p-warn">FLAG</span> edge with a caution</span><span><span class="pill p-mv">LINE / MODEL MOVED</span> edge now under 7</span><span><b>Bold</b> pick = 2+ pt edge · away @ home · tap a game for details</span></div>
<div class="tbl"><table class="games"><thead><tr>
<th title="Away team @ home team, kickoff in Central time">Game</th><th title="Projected score (away–home) and the favorite's chance to win">Prediction <span class="when">score · win chance</span></th><th class="hide-sm" title="Middle of the sportsbooks' current lines">Line <span class="when">spread · total</span></th><th class="hide-sm" title="The model's own spread and total, from expected points">Model <span class="when">spread · total</span></th><th class="hide-sm" title="Points between the model and the line; 7+ can be a bet">Edge</th><th title="The model's side in every game; BET = qualifies, FLAG = 7+ with a caution">Pick</th>
</tr></thead><tbody id="rows"></tbody></table></div>
""".replace("</div>\n<p", "</div>\n<p", 1) + "<script>window.PAGE='index'</script>"

REPORT_BODY = NAV("r") + r"""
<script src="history.js?v=__VER__"></script>
<nav class="secnav" aria-label="Sections"><a href="#glance">Summary</a><a href="#s-live">This season</a><a href="#s-backtest">Backtest</a><a href="#s-edges">Edges &amp; cautions</a><a href="#s-prob">Probabilities</a><a href="#s-tested">Model tests</a><a href="#s-recent">Every game</a><a href="#s-health">Data health</a></nav>
<div class="summary" id="glance"></div>
<div id="recon"></div>

<h2 id="s-live">This season <span class="method">Live · recorded before kickoff, graded after</span></h2>
<p class="sub">Every number here comes from predictions saved before each game started and never changed afterward. Bets are graded at the line where they were first flagged (the middle of the sportsbooks), with 1 unit risked per bet at −110.</p>
<div class="score2" id="scorecards"></div>
<div class="chartgrid" style="margin-top:12px"><div class="chart"><h4>Bets: profit in units <span class="when">1 unit per bet at −110</span></h4><div class="bar" id="ufilters" style="margin:0 0 6px"></div><div id="units"></div></div>
<div class="chart"><h4>Predicted vs actual margin <span class="when">every graded game, points</span></h4><div id="scatter"></div></div></div>
<div class="cards" id="live" style="margin-top:12px"></div>
<div id="livemore"></div>
<div class="tbl" style="margin-top:10px"><table><thead><tr><th>Week</th><th>Bets</th><th>Units</th><th>ATS, all games</th><th>O/U, all games</th><th class="hide-sm">Margin miss <span class="when">model / market, same games</span></th></tr></thead><tbody id="weekly"></tbody></table></div>

<h2 id="s-backtest">Backtest <span class="method">Historical · rebuilt from past results and lines</span></h2>
<p class="sub">Walk-forward: each week is predicted using only data available before it, then graded against that week's lines. It approximates the live rules but can't replay everything (weather alerts, or the exact lines available on the day; QB changes are replayed from past box scores), and these same seasons were used to choose the 7-point rule, so treat it as an upper bound. "Miss" = average points off (lower is better).</p>
<details class="help"><summary>What each column means</summary><ul>
<li><b>Bets ATS vs open</b>: games where the model and the opening spread were 7+ points apart, graded against that opening line (when the model's bets are meant to be placed).</li>
<li><b>Bets ATS vs close</b>: the same rule against the closing line, the sharpest line of the week.</li>
<li><b>Bets O/U</b>: the same for totals.</li>
<li><b>All games ATS vs close</b>: the model's side in every game against the closing line, whatever the gap.</li>
<li><b>Approx. live rules</b>: the opening-line bets that today's cautions would still allow (gap under 10, after week 3, no FCS teams).</li>
<li><b>Range</b>: the 95% range for the true win rate given the number of settled bets (pushes excluded). It reflects luck only, not the extra optimism from choosing the rules on these seasons. The <b>week-grouped</b> range allows for bets in the same week moving together.</li>
<li>Small grey numbers after a record are pushes. Opening lines exist mostly from 2023 on.</li>
</ul></details>
<div class="cards" id="btcards"></div>
<p class="sub">Ranges: <span class="up">green</span> = even the low end clears the 52.4% needed to profit at −110, <span style="color:var(--warn)">amber</span> = can't be told apart from break-even yet, <span class="down">red</span> = below break-even.</p>
<div class="tbl"><table class="btt"><thead><tr><th>Season</th><th>Games</th><th>Margin miss <span class="when">model / market</span></th><th class="hide-sm">Total miss</th><th>Bets ATS vs open</th><th>Bets ATS vs close</th><th>Bets O/U vs open</th><th class="hide-sm">Bets O/U vs close</th><th class="hide-sm">All games ATS vs close</th></tr></thead><tbody id="bt"></tbody></table></div>
<p class="sub" id="stability"></p>
<details class="help"><summary>Data coverage by season</summary><div id="coverage"></div></details>

<h2 id="s-edges">Edge size and cautions <span class="method">Historical</span></h2>
<p class="sub">Do bigger disagreements with the line really win more often? And do the cautions keep out bad bets? Records are against the opening line unless noted.</p>
<div class="chartgrid"><div class="chart"><h4>Win rate by gap size <span class="when">spreads vs open · bars = 95% range</span></h4><div id="ebs"></div></div>
<div class="chart"><h4>Win rate by gap size <span class="when">totals vs open · bars = 95% range</span></h4><div id="ebt"></div></div></div>
<div class="tbl" style="margin-top:10px"><table><thead><tr><th>Gap</th><th>Spreads vs open</th><th>Spreads vs close</th><th>Totals vs open</th><th>Totals vs close</th></tr></thead><tbody id="ebtbl"></tbody></table></div>
<h4 class="mh">The cautions</h4>
<p class="sub">Flagged bets (7+ vs open) that each caution blocks. A caution helps if its blocked bets did worse than the ones it lets through. QB changes are replayed from past box scores (the starter from each team's previous game vs its regular starter). Weather can't be replayed; the live season records it below as a shadow record. A game can fall under more than one caution.</p>
<div class="tbl"><table><thead><tr><th>Group</th><th>Spreads</th><th>Totals</th></tr></thead><tbody id="cautions"></tbody></table></div>
<p class="sub" id="cautionnote"></p>
<h4 class="mh">An honest test of the 7-point rule</h4>
<p class="sub">Choosing the threshold on the same seasons used to grade it flatters the result. Here each season is bet with whichever threshold (5–10 points) made the most units in the seasons <i>before</i> it, so no season helps pick its own rule.</p>
<div class="tbl"><table><thead><tr><th>Rule set</th><th>Seasons tested</th><th>Thresholds chosen</th><th>Record</th><th>Units</th></tr></thead><tbody id="nested"></tbody></table></div>

<h2 id="s-prob">Accuracy and calibration <span class="method">Historical + live</span></h2>
<p class="sub"><b>Avg miss</b> = average points off. <b>Big-miss measure</b> (RMSE) weighs large misses more. <b>Lean</b> = predicted minus actual on average (positive = projecting too high). <b>In 80% range</b> = how often the result landed inside the model's 80% range (about 80% is right).</p>
<div class="tbl"><table><thead><tr><th></th><th>Avg miss</th><th>Big-miss measure</th><th>Lean</th><th>In 80% range</th><th>Games</th></tr></thead><tbody id="acc"></tbody></table></div>
<p class="sub">A probability is well calibrated if things it calls 70% happen about 70% of the time. The <b>Brier score</b> measures probability accuracy: 0 is perfect, 0.25 is a coin flip, lower is better.</p>
<div class="chartgrid"><div class="chart"><h4>Win probability: predicted vs actual <span class="when">10 equal-size groups · dot size = games</span></h4><div id="wpchart"></div><div class="sub" id="wpnote" style="margin-top:6px"></div></div>
<div class="chart"><h4>Cover probability vs the closing line <span class="when">dot size = games · diagonal = perfect</span></h4><div id="calchart"></div></div></div>
<div class="tbl" style="margin-top:10px"><table><thead><tr><th>Model said (home covers)</th><th>Average</th><th>Actually covered</th><th>Games</th></tr></thead><tbody id="cal"></tbody></table></div>
<p class="sub" id="shrink"></p>
<details class="help"><summary>Win-probability accuracy by season</summary><div id="wpseason"></div></details>
<details class="help"><summary>Where the model is systematically off (by week, matchup type and favorite size)</summary>
<p class="sub">Bias = average of (predicted − actual). Red rows are biases unlikely to be chance (100+ games, more than 2.5 standard errors).</p>
<div id="seg"></div><div id="conflean"></div></details>

<h2 id="s-tested">Model tests and changes</h2>
<p class="sub" id="expsub"></p>
<div class="tbl"><table><thead><tr><th>Feature</th><th>Margin error</th><th>Total error</th><th>Result</th></tr></thead><tbody id="exp"></tbody></table></div>
<p class="sub" id="exphold"></p>
<details class="help"><summary>Ideas tested earlier and dropped</summary><p>None of these improved accuracy on seasons the model hadn't seen: weather at kickoff, travel and rest, recency weighting, run/pass matchups, fitting the margin directly, a diminishing-returns margin rating, per-team home field, coaching changes, turnover-based preseason weighting, star-rating-based recruiting/portal data, the expected starting QB as a model input (it stays an alert), separate prior strengths per stat, blending ratings with different prior strengths, a median-based (instead of average-based) points fit, a matchup-type totals correction, explosiveness, standard/passing-down efficiency, line yards and stuff rate, a QB adjustment only for new starters, a prior strength that changes through the season, a Power 4 vs Group of 5 margin adjustment, and a machine-learning (gradient-boosted) correction on top of the model. A separate totals model made totals more accurate but its flagged over/under bets did worse against opening lines, so it's off. Retested or added in Oct 2026 under the stricter rule, and still no: rest days, byes and short weeks; travel distance; time zones; altitude; West Coast teams in early Eastern kickoffs; weighting recent games more within a season (worse at every setting); CollegeFootballData's Elo rating as an input (passed 2017–2024, then cut 2025's bet units in half); squared and interaction terms for totals (totals error down for real, but fewer bet wins on the newest season); and different recency weights for the points fit.</p></details>
<details class="help"><summary>Model version and changelog</summary><div id="changelog"></div></details>

<h2 id="s-recent">Every graded game <span class="method">Live</span></h2>
<p class="sub">Scores are away–home. Picks are against the last line tracked before kickoff; <span class="pill p-good">BET</span> marks flagged bets, graded at the line when they were first flagged. "Units" is the running total for bets (1 unit each at −110). CLV = points gained versus that last line (positive is good).</p>
<div class="bar" id="hfilters"></div>
<p class="sub">Every bet and blocked flag is also written, once and never edited, to a <a class="lnk" href="decisions.csv">decision log (CSV)</a> with the time, line, best book, price, probability and reasons.</p>
<div class="tbl"><table><thead><tr><th>Game</th><th>Final</th><th class="hide-sm">Projected</th><th>Spread pick</th><th>Total pick</th><th class="hide-sm">CLV</th><th class="hide-sm">Units</th></tr></thead><tbody id="hist"></tbody></table></div>

<h2 id="s-health">Data health and settings</h2>
<div id="health"></div>
<div class="sub" id="settings" style="margin-top:10px"></div>
"""


INDEX_JS = r"""
const G=D.games;
const ctTime=t=>new Date(t).toLocaleString('en-US',{timeZone:'America/Chicago',weekday:'short',hour:'numeric',minute:'2-digit'});
const linesAge=D.lines_updated?(new Date(D.generated)-new Date(D.lines_updated))/36e5:null;
document.getElementById('sub').innerHTML=`${D.season} season · ${D.week==null?'offseason':wkName(D.week)} · model run ${ctTime(D.generated)} CT`+(D.lines_updated?` · lines ${ctTime(D.lines_updated)} CT`:'')+(linesAge!=null&&linesAge>12?` <span class="pill p-warn">lines may be stale</span>`:'');
const L=D.report.live||{};
// A flag can only become a bet from a fresh sportsbook quote (not the backup line, not a stale snapshot)
const noQuote=(g,k)=>{const m=g.market||{};if(m[k+'_src']==='cfbd'||!(m.books||[]).length||!m.last_ts)return true;
 return (new Date(D.generated)-new Date(m.last_ts))/36e5>36;};
const quoteWhy=(g,k)=>noQuote(g,k)?[(g.market||{})[k+'_src']==='cfbd'?'backup line only: no live sportsbook quote':'no fresh sportsbook quote']:[];
const isLowTotal=g=>(g.low_conf||[]).length||(g.total_caution||[]).length||noQuote(g,'total');
const isLowSpread=g=>(g.low_conf||[]).length||(g.spread_caution||[]).length||noQuote(g,'spread');
// Edges are shown cut (not rounded) to one decimal, so a 6.96 never shows as a qualifying 7.0
const e1=x=>(Math.floor(x*10+1e-9)/10).toFixed(1);
const SOFT=(g,k)=>(g.soft_notes||{})[k];
// A recorded bet whose current line no longer clears the bar: still graded at the flagged line,
// but not something to act on now.
const TH=D.thresholds||{spread:7,total:7};
const movedS=g=>{const b=g.spread_bet,m=g.market||{};if(!b||g.started||m.spread==null||g.model_spread==null)return false;
 const e=m.spread-g.model_spread;return b.side==='home'?e<TH.spread:-e<TH.spread;};
const movedT=g=>{const b=g.total_bet,m=g.market||{};if(!b||g.started||m.total==null||g.model_total==null)return false;
 const e=g.model_total-m.total;return b.side==='over'?e<TH.total:-e<TH.total;};
const bets=G.filter(g=>!g.started&&(g.spread_bet||g.total_bet));
// Which changed since a bet was flagged: the line, or the model's number (QB news, corrections)?
// points the line moved since a bet was recorded: positive = in the bet's favor (it beat today's number)
const gainTxt=v=>v==null||Math.abs(v)<0.05?'':` (${v>0?`${v.toFixed(1)} pts in your favor`:`${Math.abs(v).toFixed(1)} pts against you`})`;
const gainS=(g,b)=>{const m=g.market||{};return m.spread==null?null:(b.side==='home'?b.line-m.spread:m.spread-b.line);};
const gainT=(g,b)=>{const m=g.market||{};return m.total==null?null:(b.side==='over'?m.total-b.line:b.line-m.total);};
const mvWhat=(g,k)=>{const b=g[k+'_bet'],m=g.market||{};if(!b)return 'LINE MOVED';return m[k]!=null&&m[k]!==b.line?'LINE MOVED':'MODEL MOVED';};
const nW=G.filter(g=>!g.started).reduce((n,g)=>n+(g.spread_bet?1:0)+(g.total_bet?1:0),0);
const nMv=G.filter(g=>!g.started).reduce((n,g)=>n+(movedS(g)?1:0)+(movedT(g)?1:0),0);
const nF=G.filter(g=>!g.started).reduce((n,g)=>n+(!g.spread_bet&&g.spread_flag&&isLowSpread(g)?1:0)+(!g.total_bet&&g.total_flag&&isLowTotal(g)?1:0),0);
const withPct=r=>!r||(r.w+r.l+r.p)===0?'—':`${rec(r)}<span class="when">${pct(r.pct)}${(r.w+r.l)<50?` · only ${r.w+r.l} bets`:''}</span>`;
document.getElementById('cards').innerHTML=[
['Bets this week',`${nW}${!nW?'<span class="when">none yet</span>':nMv?`<span class="when">${nW-nMv} still 7+ today · ${nMv} line moved (still graded)</span>`:'<span class="when">all still 7+ today</span>'}`],['Flags',`${nF}<span class="when">check the news</span>`],['Season bets ATS',withPct(L.ats)],['Season bets O/U',withPct(L.ou)],
['All picks ATS',withPct(L.ats_all)],['All picks O/U',withPct(L.ou_all)]].map(([l,v])=>`<div class="card"><div class="l">${l}</div><div class="v">${v}</div></div>`).join('');
const day=s=>new Date(s).toLocaleDateString('en-US',{timeZone:'America/Chicago',weekday:'long',month:'short',day:'numeric'});
const F=[['all','All games'],['soon','Next 24 hours'],['bets','Bets'],['edges','All flags'],['P4','Power 4'],['G5','Group of 5']];
let cur='all',hideLow=false,q='';
const fb=document.getElementById('filters');
fb.innerHTML=F.map(([k,l])=>`<button class="chip${k==='all'?' on':''}" data-k="${k}">${l}</button>`).join('')+
`<button class="chip" id="low">Hide low confidence</button><select id="conf" class="chip"><option value="">All conferences</option>${[...new Set(G.flatMap(g=>[g.home_conf,g.away_conf]).filter(Boolean))].sort().map(c=>`<option>${esc(c)}</option>`).join('')}</select><select id="mkt" class="chip" aria-label="Market"><option value="">Spreads and totals</option><option value="s">Spreads only</option><option value="t">Totals only</option></select><select id="sort" class="chip" aria-label="Sort"><option value="time">Sort: kickoff</option><option value="edge">Sort: biggest edge now</option><option value="flag">Sort: biggest edge when flagged</option><option value="move">Sort: model moved most</option><option value="ev">Sort: expected value</option></select><select id="day" class="chip" aria-label="Day"><option value="">Every day</option>${[...new Set(G.filter(g=>!g.completed).map(g=>day(g.start)))].map(d=>`<option>${esc(d)}</option>`).join('')}</select><input type="search" id="q" placeholder="Find a team">`;
let sortBy='time',conf='',mkt='',dayF='';document.getElementById('day').onchange=e=>{dayF=e.target.value;draw();};document.getElementById('sort').onchange=e=>{sortBy=e.target.value;draw();};document.getElementById('mkt').onchange=e=>{mkt=e.target.value;draw();};document.getElementById('conf').onchange=e=>{conf=e.target.value;draw();};
fb.querySelectorAll('[data-k]').forEach(b=>b.onclick=()=>{cur=b.dataset.k;fb.querySelectorAll('[data-k]').forEach(x=>x.classList.toggle('on',x===b));draw();});
document.getElementById('low').onclick=e=>{hideLow=!hideLow;e.target.classList.toggle('on',hideLow);draw();};
document.getElementById('q').oninput=e=>{q=e.target.value.toLowerCase();draw();};
const isBet=g=>!!(g.spread_bet||g.total_bet);
function keep(g){
 if(q&&!(g.home+' '+g.away).toLowerCase().includes(q))return false;
 if(conf&&g.home_conf!==conf&&g.away_conf!==conf)return false;
 if(hideLow&&((g.low_conf||[]).length||(g.spread_caution||[]).length||(g.total_caution||[]).length))return false;
 const r=g.result||{},S=mkt!=='t',T=mkt!=='s';
 if(dayF&&day(g.start)!==dayF)return false;
 if(cur==='soon'){const t=new Date(g.start)-Date.now();return !g.completed&&t<24*36e5;}
 if(cur==='bets')return g.completed?!!((S&&r.ats)||(T&&r.ou)):!!((S&&g.spread_bet)||(T&&g.total_bet));
 if(cur==='edges')return g.completed?!!((S&&r.ats)||(T&&r.ou)):!!((S&&(g.spread_flag||g.spread_bet))||(T&&(g.total_flag||g.total_bet)));
 if(cur==='P4')return g.home_tier==='P4'||g.away_tier==='P4';
 if(cur==='G5')return g.home_tier==='G5'||g.away_tier==='G5';
 return true;}
// Plain-English reasons for a caution
const RS=(g,x)=>{const r=REASON(x);if(r!=='QB change: see details')return r;const t=qbTeams(g);return t.length?`QB change: ${t.join(', ')}`:r;};
const qbTeams=g=>[...new Set((g.notes||[]).filter(n=>n.includes('started last game')||n.includes('expected starter')).map(n=>n.split(':')[0]))];
const REASON=x=>x.startsWith('big disagreement')?'Model and line 10+ pts apart: check injury/QB news':x.startsWith('8+ pts')?'8+ pts apart: worth a quick news check':x.startsWith('wind/rain')?"Wind/rain forecast: the model doesn't use weather":x==='QB change'?'QB change: see details':x==='manual adjustment'?'Manual adjustment':x==='FCS opponent'?'FCS opponent':x==='early season'?'Early season':x;
// Everything the row needs about the two picks (spread, total) in one place
function pickData(g){
 const m=g.market||{};
 if(g.model_spread==null)return null;
 if(g.completed&&g.result){
  const r=g.result,sBet=!!r.ats,tBet=!!r.ou;
  const sp=spreadPick(g,r.close_spread,g.model_spread),tp=totalPick(r.close_total,g.model_total);
  // A recorded bet is shown exactly as flagged, whatever the closing line did
  let spx=sp,tpx=tp;
  if(sBet&&g.first_spread_flag){const f=g.first_spread_flag,home=f.side==='home';spx={team:home?g.home:g.away,line:home?f.line:-f.line,edge:Math.abs(f.line-f.model)};}
  if(tBet&&g.first_total_flag){const f=g.first_total_flag;tpx={side:f.side==='over'?'Over':'Under',line:f.line,edge:Math.abs(f.model-f.line)};}
  return {sp:spx,tp:tpx,sSt:sBet?'bet':(g.spread_flag&&((g.low_conf||[]).length||(g.spread_caution||[]).length)?'caution':''),tSt:tBet?'bet':(g.total_flag&&((g.low_conf||[]).length||(g.total_caution||[]).length)?'caution':''),sRes:resPill(sBet?r.ats:r.ats_all),tRes:resPill(tBet?r.ou:r.ou_all),why:[],
          lineS:r.close_spread,lineT:r.close_total,closing:true};}
 let sp=spreadPick(g,m.spread,g.model_spread),tp=totalPick(m.total,g.model_total);
 let sSt=g.spread_flag&&isLowSpread(g)?'caution':'',tSt=g.total_flag&&isLowTotal(g)?'caution':'';
 // A bet stays at the line where it was first flagged, even if the line has moved since
 const sb=g.spread_bet,tb=g.total_bet;
 if(sb){const home=sb.side==='home';sp={team:home?g.home:g.away,line:home?sb.line:-sb.line,edge:Math.abs(sb.line-g.model_spread)};
  if(m.spread!=null&&m.spread!==sb.line){sp.now=sl(home?m.spread:-m.spread);sp.edgeNow=Math.abs(m.spread-g.model_spread);}sSt=movedS(g)?'moved':'bet';}
 if(tb){tp={side:tb.side==='over'?'Over':'Under',line:tb.line,edge:Math.abs(g.model_total-tb.line)};
  if(m.total!=null&&m.total!==tb.line){tp.now=m.total;tp.edgeNow=Math.abs(g.model_total-m.total);}tSt=movedT(g)?'moved':'bet';}
 const why=[...((sSt||tSt)?(g.low_conf||[]):[]),...(sSt?(g.spread_caution||[]):[]),...(tSt?(g.total_caution||[]):[]),...(sSt==='caution'?quoteWhy(g,'spread'):[]),...(tSt==='caution'?quoteWhy(g,'total'):[]),...(sSt&&SOFT(g,'spread')?[SOFT(g,'spread')]:[]),...(tSt&&SOFT(g,'total')?[SOFT(g,'total')]:[])];
 return {sp,tp,sSt,tSt,sRes:'',tRes:'',why,lineS:m.spread,lineT:m.total,closing:false};}
function pickLine(kind,p,status,result,g){
 if(!p)return `<div class="ln muted">${kind==='Spread'?'No spread line':'No total line'}</div>`;
 if(p.none)return `<div class="ln muted">${kind==='Spread'?'Spread':'Total'}: no pick</div>`;
 const txt=kind==='Spread'?`${esc(p.team)} ${sl(p.line)}`:`${p.side} ${p.line}`;
 const tag=status==='bet'?'<span class="pill p-good">BET</span> ':status==='moved'?`<span class="pill p-mv" title="Recorded at this line; today's edge is under 7 points">${g?mvWhat(g,kind==='Spread'?'spread':'total'):'LINE MOVED'}</span> `:status==='caution'?'<span class="pill p-warn">FLAG</span> ':'';
 const weak=!status&&p.edge<2;
 return `<div class="ln">${tag}<b class="${weak?'weak':''}">${txt}</b> ${result||''}<span class="sm-inline e"> edge ${e1(p.edge)}</span>${p.now?`<span class="e"> · now ${p.now}</span>`:''}</div>`;}
const edgeCell=p=>{if(!p||p.none)return '<div class="ln muted">—</div>';const e=p.edgeNow!=null?p.edgeNow:p.edge;
 return `<div class="ln ${e>=7?'hot':e>=4?'mid':e<2?'muted':''}"${p.edgeNow!=null?` title="Edge against today's line. The bet's own edge at ${p.line} is ${e1(p.edge)}."`:''}>${e1(e)}</div>`;};
function moveCell(g){const m=g.market||{};if(m.open_spread==null||m.spread==null)return '<span class="muted">—</span>';
 const d=m.spread-m.open_spread;if(Math.abs(d)<0.25)return `<span class="muted nw">No move</span>`;
 let toward='';if(g.model_spread!=null){toward=Math.abs(g.model_spread-m.spread)<Math.abs(g.model_spread-m.open_spread)?'up':'down';}
 const o=m.open_spread,c=m.spread;let txt;
 if(o<=0&&c<=0)txt=`${esc(g.home)} ${o===0?'PK':'−'+Math.abs(o)} → ${c===0?'PK':'−'+Math.abs(c)}`;
 else if(o>=0&&c>=0)txt=`${esc(g.away)} ${o===0?'PK':'−'+o} → ${c===0?'PK':'−'+c}`;
 else txt=`${line(g.home,g.away,o)} → ${line(g.home,g.away,c)}`;
 return `<span class="${toward}">${txt}</span><div class="when">${toward==='up'?'toward model':'away from model'}</div>`;}
function icons(g){const w=g.weather||{};let s='';
 if(!g.completed&&!w.dome&&(w.wind>=15||w.precip>=0.1))s+=` <span class="pill p-warn" title="Wind/rain forecast">${w.wind>=15?Math.round(w.wind)+' mph':''}${w.wind>=15&&w.precip>=0.1?' · ':''}${w.precip>=0.1?'rain':''}</span>`;
 const n=g.notes||[],qn=n.filter(x=>x.includes('expected starter')||x.includes('started last game'));
 if(qn.length)s+=` <span class="pill p-bad" title="${esc(qn.join('; '))}">QB change</span>`;
 if(n.length>qn.length)s+=' <span class="pill p-bad" title="Manual adjustment">manual adj</span>';
 if([...(g.low_conf||[]),...(g.spread_caution||[]),...(g.total_caution||[])].some(x=>x.startsWith('big disagreement')))s+=' <span class="pill p-warn" title="Model and market 10+ points apart: check injury/QB news">check news</span>';return s;}
// 10th to 90th percentile of the simulated home margin, from the 3-point histogram bins (-45..+45)
function marginRange(g){const side0=m=>Math.abs(m)<0.5?'tie':(m>0?`${esc(g.home)} by ${Math.round(m)}`:`${esc(g.away)} by ${Math.round(-m)}`);
 if(g.range80)return `${side0(g.range80[0])} to ${side0(g.range80[1])}`;
 const h=g.hist||[];const n=h.reduce((a,b)=>a+b,0);if(!n)return null;
 const q=t=>{let c=0;for(let i=0;i<h.length;i++){if(c+h[i]>=t*n){const lo=-45+3*i;return lo+3*((t*n-c)/Math.max(1,h[i]));}c+=h[i];}return 45;};
 const side=m=>Math.abs(m)<0.5?'tie':(m>0?`${esc(g.home)} by ${Math.round(m)}`:`${esc(g.away)} by ${Math.round(-m)}`);
 return `${side(q(0.1))} to ${side(q(0.9))}`;}
// The single book offering the most for a side (most points; lowest total for an over, highest for an under)
function bestFor(g,kind,side){const bs=((g.market||{}).books||[]);let best=null;
 const pay=p=>{p=p==null?-110:p;return p>0?p/100:100/-p;};
 for(const b of bs){let line,price,val;
  if(kind==='spread'){if(b.spread==null)continue;line=side==='home'?b.spread:-b.spread;price=side==='home'?b.home_price:b.away_price;val=line;}
  else{if(b.total==null)continue;line=b.total;price=side==='over'?b.over_price:b.under_price;val=side==='over'?-line:line;}
  if(!best||val>best.val||(val===best.val&&pay(price)>pay(best.price)))best={book:b.book,line,price,val};}
 return best;}
const bestTxt=(g,kind,side)=>{const sv=g['best_'+kind];const b=sv&&g[kind+'_side']===side?sv:bestFor(g,kind,side);if(!b)return '';
 return kind==='spread'?`best ${esc(side==='home'?g.home:g.away)} ${sl(b.line)}${b.price!=null?` (${am(b.price)})`:''} at ${esc(b.book)}`:`best ${side==='over'?'Over':'Under'} ${b.line}${b.price!=null?` (${am(b.price)})`:''} at ${esc(b.book)}`;};
// Tiny line-history chart: the consensus line at each snapshot, with the model's number dashed
function spark(vals,model,label,fmtv){const v=vals.filter(x=>x.v!=null);if(v.length<2)return '';
 const W=260,H=58,lo=Math.min(...v.map(x=>x.v),model??Infinity)-0.5,hi=Math.max(...v.map(x=>x.v),model??-Infinity)+0.5;
 const X=i=>8+(W-16)*i/(v.length-1),Y=y=>6+(H-18)*(hi-y)/(hi-lo||1);
 const path=v.map((x,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(x.v).toFixed(1)}`).join('');
 const dots=v.map((x,i)=>`<circle cx="${X(i).toFixed(1)}" cy="${Y(x.v).toFixed(1)}" r="2.5" fill="var(--series)"><title>${ctTime(x.t)} CT: ${fmtv(x.v)}</title></circle>`).join('');
 const mdl=model==null?'':`<line x1="8" x2="${W-8}" y1="${Y(model).toFixed(1)}" y2="${Y(model).toFixed(1)}" stroke="var(--faint)" stroke-dasharray="4 3"/><text x="${W-8}" y="${(Y(model)-3).toFixed(1)}" text-anchor="end" font-size="10" fill="var(--muted)">model ${fmtv(model)}</text>`;
 return `<div class="spark"><div class="when">${label}: ${fmtv(v[0].v)} → ${fmtv(v[v.length-1].v)} over ${v.length} snapshots</div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${label} history">${mdl}<path d="${path}" fill="none" stroke="var(--series)" stroke-width="2"/>${dots}<text x="8" y="${H-2}" font-size="10" fill="var(--faint)">${ctTime(v[0].t)}</text><text x="${W-8}" y="${H-2}" text-anchor="end" font-size="10" fill="var(--faint)">${ctTime(v[v.length-1].t)}</text></svg></div>`;}
function detail(g){
 if(g.home_med==null)return '<div class="sub">No projection stored for this game.</div>';
 const h=g.hist||[],mx=Math.max(1,...h);const m=g.market||{};
 const bars=h.map((v,i)=>`<div class="${(i*3-45)<0?'z':''}" style="height:${Math.round(v/mx*100)}%" title="${i*3-45} to ${i*3-43}"></div>`).join('');
 const likely=(g.likely||[]).map(([a,b,p])=>`<div class="kv"><span>${esc(g.away)} ${b}, ${esc(g.home)} ${a}</span><span>${pct1(p)}</span></div>`).join('');
 // Books sorted from best to worst for the model's side, with prices
 const sSide=g.spread_side||(m.spread!=null&&g.model_spread!=null?(m.spread-g.model_spread>0?'home':'away'):'home');
 const sval=b=>b.spread==null?-1e9:(sSide==='home'?b.spread:-b.spread);
 const bS=g.best_spread&&g.spread_side===sSide?g.best_spread:bestFor(g,'spread',sSide);
 // the expected-value pick first, then better line, then better price
 const spr=b=>{const v=sSide==='home'?b.home_price:b.away_price;return v==null?-1e9:v;};
 const bk=(m.books||[]).slice().sort((x,y)=>((bS&&y.book===bS.book)-(bS&&x.book===bS.book))||(sval(y)-sval(x))||(spr(y)-spr(x)));
 const sTeam=esc(sSide==='home'?g.home:g.away);
 const books=bk.map(b=>{const pr=sSide==='home'?b.home_price:b.away_price;return `<div class="kv"><span>${esc(b.book)}${bS&&b.book===bS.book?' <span class="pill p-acc">best</span>':''}</span><span>${b.spread==null?'—':`${sTeam} ${sl(sSide==='home'?b.spread:-b.spread)}`}${pr!=null?` <span class="when" style="display:inline">${am(pr)}</span>`:''} · ${fmt(b.total)}</span></div>`;}).join('')||'<div class="sub">No sportsbook snapshot yet (line from CollegeFootballData).</div>';
 const R=g.ratings||{};const wx=g.weather||{};
 const wtxt=wx.dome?'Dome':wx.wind!=null?`${Math.round(wx.temp)}°F · ${Math.round(wx.wind)} mph wind · ${wx.precip>=0.01?wx.precip.toFixed(2)+' in rain':'no rain'}`:null;
 const fs=g.spread_bet,ft=g.total_bet;
 // The model vs the line, in words
 const fav=x=>x<0?`${esc(g.home)} by ${Math.abs(x).toFixed(1)}`:x>0?`${esc(g.away)} by ${x.toFixed(1)}`:'even';
 const sp=spreadPick(g,m.spread,g.model_spread),tp=totalPick(m.total,g.model_total);
 const evT=x=>x==null?'':` · EV ${x>=0?'+':'−'}${Math.abs(x*100).toFixed(1)}%`;
 const vs=[sp&&!sp.none?`<div class="vsl"><span class="when" style="display:inline">Spread</span> Model ${fav(g.model_spread)}, line ${line(g.home,g.away,m.spread)}: <b>${e1(sp.edge)} pts toward ${esc(sp.team)}</b><span class="when" style="display:inline">${evT(g.ev_spread)}</span></div>`:'',
  tp&&!tp.none?`<div class="vsl"><span class="when" style="display:inline">Total</span> Model ${fmt(g.model_total)}, line ${m.total}: <b>${e1(tp.edge)} pts to the ${tp.side==='Over'?'over':'under'}</b><span class="when" style="display:inline">${evT(g.ev_total)}</span></div>`:''].join('');
 const fm=g.first_model;const moved=fm&&(Math.abs(fm.spread-g.model_spread)>=0.1||Math.abs(fm.total-g.model_total)>=0.1)?
  `<div class="kv"><span>Model since ${ctTime(fm.ts)}</span><span>${line(g.home,g.away,fm.spread)} → ${line(g.home,g.away,g.model_spread)} · ${fmt(fm.total)} → ${fmt(g.model_total)}</span></div>`:'';
 const hs=(m.history||[]);
 const sparks=spark(hs.map(x=>({t:x.ts,v:x.spread})),g.model_spread,`${esc(g.home)} spread`,v=>v===0?'PK':(v>0?'+':'−')+Math.abs(v))+
  spark(hs.map(x=>({t:x.ts,v:x.total})),g.model_total,'Total',v=>String(Math.round(v*10)/10));
 const notes=[m.last_ts?`Odds as of ${ctTime(m.last_ts)} CT${(new Date(D.generated)-new Date(m.last_ts))/36e5>=1?` (${Math.round((new Date(D.generated)-new Date(m.last_ts))/36e5)} h before this update)`:''}`:(m.spread!=null?'Backup line from CollegeFootballData (no sportsbook snapshot): shown, never bet':'No line yet'),
  wtxt?null:'No weather forecast yet'].filter(Boolean);
 const flagRow=(f,kind)=>{if(!f)return '';const home=f.side==='home';const txt=kind==='spread'?`${esc(home?g.home:g.away)} ${sl(home?f.line:-f.line)}`:`${f.side==='over'?'Over':'Under'} ${fmt(f.line)}`;
  const b=f.best?` · best then ${kind==='spread'?sl(f.best.line):f.best.line}${f.best.price!=null?` (${am(f.best.price)})`:''} at ${esc(f.best.book)}`:'';
  return `<div class="kv"><span>${kind==='spread'?'Spread':'Total'} bet flagged ${f.ts?ctTime(f.ts):''}</span><span>${txt} (edge ${Math.abs(f.line-f.model).toFixed(1)})${b}</span></div>`;};
 const pushS=g.p_push_spread>=0.005?`<div class="kv"><span>Push on the spread</span><span>${pct(g.p_push_spread)}</span></div>`:'';
 const pushT=g.p_push_total>=0.005?`<div class="kv"><span>Push on the total</span><span>${pct(g.p_push_total)}</span></div>`:'';
 return `<div class="dgrid">
 <div><h4>Model vs the line</h4>${vs||'<div class="sub">No line to compare yet.</div>'}${moved}
  <h4 style="margin-top:12px">Probabilities</h4>
  <div class="kv"><span>${esc(g.away)} wins</span><span>${winPct(1-g.p_home_win)}</span></div>
  <div class="kv"><span>${esc(g.home)} wins</span><span>${winPct(g.p_home_win)}</span></div>
  ${g.p_side_cover!=null?`<div class="kv"><span>${esc(g.spread_side==='home'?g.home:g.away)} covers</span><span>${pct(g.p_side_cover)}</span></div>`:''}${pushS}
  ${g.p_side_total!=null?`<div class="kv"><span>${g.total_side==='over'?'Over':'Under'} hits</span><span>${pct(g.p_side_total)}</span></div>`:''}${pushT}
  <h4 style="margin-top:12px">Most likely final scores</h4>${likely}</div>
 <div><h4>Simulated margins (${esc(g.home)} − ${esc(g.away)})</h4><div class="hist">${bars}</div>
  <div class="kv sub"><span>−45</span><span>0</span><span>+45</span></div>
  ${(()=>{const r=marginRange(g);return r?`<div class="kv"><span>80% range of outcomes</span><span>${r}</span></div>`:''})()}
  <h4 style="margin-top:12px">Inputs</h4>
  <div class="kv"><span>Expected points (away · home)</span><span>${fmt(g.exp_away)} · ${fmt(g.exp_home)}</span></div>
  <div class="kv"><span>Off EPA/play (away · home)</span><span>${epa(R.away_off)} · ${epa(R.home_off)}</span></div>
  <div class="kv"><span>Def EPA/play allowed (away · home)</span><span>${epa(R.away_def)} · ${epa(R.home_def)}</span></div>
  <div class="kv"><span>Expected plays (away · home)</span><span>${fmt(R.away_pace)} · ${fmt(R.home_pace)}</span></div>
  ${wtxt?`<div class="kv"><span>Weather</span><span>${wtxt}</span></div>`:''}
  ${(g.notes||[]).length?`<div class="notes"><b>Notes:</b> ${esc(g.notes.join('; '))}</div>`:''}
  ${(g.auto_adj||[]).length?`<div class="notes"><b>Automatic correction:</b> ${esc(g.auto_adj.join('; '))}. The model underrates the service academies' option offenses; this is learned from past out-of-sample misses and doesn't block bets.</div>`:''}
  ${notes.length?`<div class="notes">${notes.map(esc).join(' · ')}</div>`:''}</div>
 <div><h4>Lines by book <span class="when" style="display:inline">${sTeam}'s line and price, best first · total</span></h4>${books}
  <div class="kv" style="margin-top:8px"><span>First tracked</span><span>${line(g.home,g.away,m.open_spread)} · ${fmt(m.open_total)}</span></div>
  <div class="kv"><span>Line move (spread)</span><span>${moveCell(g)}</span></div>
  <div class="kv"><span>Now (middle of books)</span><span>${line(g.home,g.away,m.spread)} · ${fmt(m.total)}</span></div>
  ${flagRow(fs,'spread')}${flagRow(ft,'total')}${sparks}</div>
 </div>
 <div style="margin-top:12px"><a class="lnk" href="matchup.html#${encodeURIComponent(g.away)}|${encodeURIComponent(g.home)}|${g.neutral?0:2}">Compare these teams on the Matchup page →</a></div>`;}
const winPct=p=>p>0.995?'>99%':p<0.005?'<1%':pct(p);
function rowHtml(g){
  const at=g.neutral?'vs':'@';
  const fav=g.p_home_win==null?'':(g.p_home_win>=0.5?`${winPct(g.p_home_win)} ${esc(g.home)}`:`${winPct(1-g.p_home_win)} ${esc(g.away)}`);
  const ps=projScore(g);
  const pred=g.completed?`<td class="c-proj"><div class="big">${g.away_pts}–${g.home_pts}</div><div class="when">final${ps?` · proj ${ps}`:''}</div></td>`:
    `<td class="c-proj">${ps?`<div class="big">${ps}</div><div class="when">${fav}</div>`:'<span class="muted">—</span>'}</td>`;
  const d=pickData(g);
  const first=`<td class="c-game">${esc(g.away)} ${at} ${esc(g.home)}${icons(g)}<div class="when">${when(g.start)}${g.started&&!g.completed?' · in progress':''}${g.neutral?' · neutral site':''}</div></td>${pred}`;
  if(!d)return `<tr class="g" id="g${g.game_id}" data-id="${g.game_id}">${first}<td class="hide-sm"></td><td class="hide-sm"></td><td class="hide-sm"></td><td class="c-pick muted">No projection</td></tr>
  <tr class="detail" id="d${g.game_id}" hidden><td colspan="6">${detail(g)}</td></tr>`;
  const lt=d.lineT==null?'—':`O/U ${d.lineT}`;
  return `<tr class="g" id="g${g.game_id}" data-id="${g.game_id}">${first}
  <td class="hide-sm"><div class="ln">${line(g.home,g.away,d.lineS)}${!d.closing&&(g.market||{}).spread_src==='cfbd'?' <span class="when" style="display:inline" title="Backup line from CollegeFootballData: shown, never bet">backup</span>':''}</div><div class="ln">${lt}</div>${d.closing?'<div class="when">last line before kickoff</div>':''}</td>
  <td class="hide-sm"><div class="ln">${line(g.home,g.away,g.model_spread)}</div><div class="ln">${fmt(g.model_total)}</div></td>
  <td class="hide-sm">${edgeCell(d.sp)}${edgeCell(d.tp)}</td>
  <td class="c-pick">${pickLine('Spread',d.sp,d.sSt,d.sRes,g)}${pickLine('Total',d.tp,d.tSt,d.tRes,g)}<div class="show-sm when">Line ${line(g.home,g.away,d.lineS)} · ${lt} · Model ${line(g.home,g.away,g.model_spread)} · ${fmt(g.model_total)}</div>${d.why.length?`<div class="when">${esc([...new Set(d.why.map(x=>RS(g,x)))].join(' · '))}</div>`:''}</td></tr>
  <tr class="detail" id="d${g.game_id}" hidden><td colspan="6">${detail(g)}</td></tr>`;}
// Largest gap between the model and TODAY's line (spread or total, per the market filter)
function maxEdge(g){const m=g.market||{};const e=[];
 if(mkt!=='t'&&m.spread!=null&&g.model_spread!=null)e.push(Math.abs(m.spread-g.model_spread));
 if(mkt!=='s'&&m.total!=null&&g.model_total!=null)e.push(Math.abs(g.model_total-m.total));
 return e.length?Math.max(...e):-1;}
// Largest edge at the moment a bet was first flagged (games without a recorded bet sort last)
function flagEdge(g){const e=[];const s=g.spread_bet,t=g.total_bet;
 if(mkt!=='t'&&s)e.push(Math.abs(s.line-s.model));if(mkt!=='s'&&t)e.push(Math.abs(t.model-t.line));return e.length?Math.max(...e):-1;}
// How far the projection has moved since it was first made
// Expected value of the model's side today (spread or total, per the market filter)
function evOf(g){const e=[];if(mkt!=='t'&&g.ev_spread!=null)e.push(g.ev_spread);if(mkt!=='s'&&g.ev_total!=null)e.push(g.ev_total);return e.length?Math.max(...e):-9;}
function modelMove(g){const f=g.first_model;if(!f)return -1;return Math.max(mkt==='t'?0:Math.abs(g.model_spread-f.spread),mkt==='s'?0:Math.abs(g.model_total-f.total));}
function draw(){
 const rows=G.filter(keep);const tb=document.getElementById('rows');
 if(!rows.length){tb.innerHTML=`<tr><td colspan="6" class="muted">${G.length?'No games match these filters.':'No games on the slate right now. Projections appear once the next week\'s schedule is out.'}</td></tr>`;return;}
 // Upcoming games grouped by day, finished games at the bottom
 const up=rows.filter(g=>!g.completed),done=rows.filter(g=>g.completed);
 const groups=[];
 if(sortBy==='edge')groups.push([`Biggest edge against today's line${mkt==='s'?' (spreads)':mkt==='t'?' (totals)':''}`,[...up].sort((a,b)=>maxEdge(b)-maxEdge(a))]);
 else if(sortBy==='flag')groups.push(['Biggest edge when the bet was flagged',[...up].sort((a,b)=>flagEdge(b)-flagEdge(a))]);
 else if(sortBy==='ev')groups.push(['Highest expected value today (conservative, calibrated probabilities)',[...up].sort((a,b)=>evOf(b)-evOf(a))]);
 else if(sortBy==='move')groups.push(['Projection moved most since it was first made',[...up].sort((a,b)=>modelMove(b)-modelMove(a))]);
 else up.forEach(g=>{const k=day(g.start);if(!groups.length||groups[groups.length-1][0]!==k)groups.push([k,[]]);groups[groups.length-1][1].push(g);});
 if(done.length)groups.push(['Final',done]);
 const n=k=>`${k.length} game${k.length===1?'':'s'}`;
 tb.innerHTML=groups.map(([k,gs])=>`<tr class="day"><td colspan="6">${k} · ${n(gs)}</td></tr>`+gs.map(rowHtml).join('')).join('');
 tb.querySelectorAll('tr.g').forEach(tr=>{tr.tabIndex=0;tr.setAttribute('aria-expanded','false');
  const t=()=>{const d=document.getElementById('d'+tr.dataset.id);d.hidden=!d.hidden;tr.setAttribute('aria-expanded',String(!d.hidden));
   try{history.replaceState(null,'',d.hidden?location.pathname:'#g'+tr.dataset.id);}catch(e){}};
  tr.onclick=t;tr.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();t();}};});}
// This week's bets and flags, above the table
function summary(){
 const open=G.filter(g=>!g.started);const items=[];
 open.forEach(g=>{const m=g.market||{};const gm=`${esc(g.away)} ${g.neutral?'vs':'@'} ${esc(g.home)} · ${when(g.start)}`;
  const low=(g.low_conf||[]);
  if(g.spread_bet){const b=g.spread_bet,home=b.side==='home';items.push({k:movedS(g)?'moved':'bet',kind:'spread',ev:g.ev_spread,g,txt:`${esc(home?g.home:g.away)} ${sl(home?b.line:-b.line)}`,gm,ts:b.ts,best:movedS(g)?'':bestTxt(g,'spread',b.side),
   edge:movedS(g)&&b.model!=null?Math.abs(b.line-b.model):Math.abs(b.line-g.model_spread),note:[m.spread!=null&&m.spread!==b.line?`line now ${sl(home?m.spread:-m.spread)}${gainTxt(gainS(g,b))}${movedS(g)?`, edge now ${Math.abs(m.spread-g.model_spread).toFixed(1)}`:''}`:'',...[...low,...(g.spread_caution||[]),SOFT(g,'spread')].filter(Boolean).map(x=>RS(g,x))].filter(Boolean).join(' · ')});}
  else if(g.spread_flag&&isLowSpread(g)){const p=spreadPick(g,m.spread,g.model_spread);if(p&&!p.none)items.push({k:'flag',g,txt:`${esc(p.team)} ${sl(p.line)}`,gm,edge:p.edge,best:bestTxt(g,'spread',p.side),note:[...new Set([...low,...(g.spread_caution||[]),...quoteWhy(g,'spread')].map(x=>RS(g,x)))].join(' · ')});}
  if(g.total_bet){const b=g.total_bet;items.push({k:movedT(g)?'moved':'bet',kind:'total',ev:g.ev_total,g,txt:`${b.side==='over'?'Over':'Under'} ${b.line}`,gm,edge:movedT(g)&&b.model!=null?Math.abs(b.model-b.line):Math.abs(g.model_total-b.line),ts:b.ts,best:movedT(g)?'':bestTxt(g,'total',b.side),
   note:[m.total!=null&&m.total!==b.line?`line now ${m.total}${gainTxt(gainT(g,b))}${movedT(g)?`, edge now ${Math.abs(g.model_total-m.total).toFixed(1)}`:''}`:'',...[...low,...(g.total_caution||[]),SOFT(g,'total')].filter(Boolean).map(x=>RS(g,x))].filter(Boolean).join(' · ')});}
  else if(g.total_flag&&isLowTotal(g)){const p=totalPick(m.total,g.model_total);if(p&&!p.none)items.push({k:'flag',g,txt:`${p.side} ${p.line}`,gm,edge:p.edge,best:bestTxt(g,'total',p.side.toLowerCase()),note:[...new Set([...low,...(g.total_caution||[]),...quoteWhy(g,'total')].map(x=>RS(g,x)))].join(' · ')});}
 });
 const el=document.getElementById('summary');
 const row=i=>`<div class="sitem" data-id="${i.g.game_id}"><span class="pill ${i.k==='bet'?'p-good':i.k==='moved'?'p-mv':'p-warn'}">${i.k==='bet'?'BET':i.k==='moved'?mvWhat(i.g,i.kind):'FLAG'}</span><span><span class="t ${i.k==='bet'?'bet':''}">${i.txt}</span> <span class="g2">${i.gm}</span></span><span class="r">edge ${e1(i.edge)}${i.k==='moved'?' when flagged':''}${i.ts?` · flagged ${ctTime(i.ts)}`:''}${i.k==='bet'&&i.ev!=null?` · <span title="Expected profit per unit at −110, from conservative calibrated probabilities">EV ${i.ev>=0?'+':'−'}${Math.abs(i.ev*100).toFixed(1)}%</span>`:''}${i.note?` · ${esc(i.note)}`:''}${i.best?`<br>${i.best}`:''}</span></div>`;
 const b=items.filter(i=>i.k==='bet'||i.k==='moved').sort((x,y)=>(x.k==='moved')-(y.k==='moved')),f=items.filter(i=>i.k==='flag');
 el.className='summary';
 el.innerHTML=`<div class="grp"><h3>This week's bets <span class="when" style="display:inline;font-weight:400">${(()=>{const o=G.filter(g=>!g.started),wl=o.filter(g=>(g.market||{}).spread!=null||(g.market||{}).total!=null).length;return `${o.length} games to play · ${wl} with lines · ${b.filter(i=>i.k==='bet').length} active bets · ${f.length} flags`;})()}</span></h3>${b.length?b.map(row).join(''):`<div class="sub">${open.length?'No bets right now: no game has a clean 7-point edge.':'No upcoming games on the slate.'}</div>`}</div>`+
  (f.length?`<div class="grp"><h3>Flags: check before betting</h3>${f.map(row).join('')}</div>`:'');
 el.querySelectorAll('.sitem').forEach(x=>{x.tabIndex=0;x.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();x.click();}};});
 el.querySelectorAll('.sitem').forEach(x=>x.onclick=()=>{cur='all';q='';hideLow=false;conf='';document.getElementById('q').value='';document.getElementById('conf').value='';document.getElementById('low').classList.remove('on');
  fb.querySelectorAll('[data-k]').forEach(c=>c.classList.toggle('on',c.dataset.k==='all'));draw();
  const tr=document.getElementById('g'+x.dataset.id);if(!tr)return;document.getElementById('d'+x.dataset.id).hidden=false;tr.setAttribute('aria-expanded','true');tr.scrollIntoView({behavior:RM?'auto':'smooth',block:'center'});
  tr.classList.add('flash');setTimeout(()=>tr.classList.remove('flash'),1500);});}
summary();
draw();
document.getElementById('wcsv').onclick=()=>{const q=v=>`"${String(v??'').replace(/"/g,'""')}"`;
 const head=['game_id','kickoff_utc','away','home','neutral','away_conf','home_conf','proj_away','proj_home','home_win_prob','line_spread_home','line_total','line_source','odds_time','model_spread_home','model_total','spread_side','spread_edge','p_spread_side','ev_spread','spread_status','total_side','total_edge','p_total_side','ev_total','total_status','best_spread','best_total','cautions','notes'];
 const st=(g,k)=>g[k+'_bet']?((k==='spread'?movedS(g):movedT(g))?'moved':'bet'):(g[k+'_flag']?'flag':'');
 const bb=b=>b?`${b.book} ${b.line}${b.price!=null?' ('+b.price+')':''}`:'';
 const rows=[head.join(',')].concat(G.map(g=>{const m=g.market||{},ps=projScore(g)||'',sc=ps.split('–');
  return [g.game_id,g.start,g.away,g.home,g.neutral?'yes':'',g.away_conf,g.home_conf,sc[0]||'',sc[1]||'',g.p_home_win!=null?g.p_home_win.toFixed(3):'',m.spread,m.total,m.spread_src||'',m.last_ts||'',g.model_spread,g.model_total,
   g.spread_side||'',g.spread_edge??'',g.p_side_cover??'',g.ev_spread??'',st(g,'spread'),g.total_side||'',g.total_edge??'',g.p_side_total??'',g.ev_total??'',st(g,'total'),bb(g.best_spread),bb(g.best_total),
   [...(g.low_conf||[]),...(g.spread_caution||[]),...(g.total_caution||[])].join('; '),[...(g.notes||[]),...(g.auto_adj||[])].join('; ')].map(q).join(',');}));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\n')],{type:'text/csv'}));a.download=`cfb-${D.season}-${wkName(D.week).replace(' ','-').toLowerCase()}.csv`;document.body.appendChild(a);a.click();a.remove();};
// Open a game from a link (index.html#g<id>), e.g. from the Matchup page
function openFromHash(){const m=location.hash.match(/^#g(\d+)$/);if(!m)return;const tr=document.getElementById('g'+m[1]),d=document.getElementById('d'+m[1]);if(!tr||!d)return;
 d.hidden=false;tr.setAttribute('aria-expanded','true');tr.scrollIntoView({block:'center'});tr.classList.add('flash');setTimeout(()=>tr.classList.remove('flash'),1500);}
openFromHash();window.addEventListener('hashchange',openFromHash);
(()=>{const o=(D.report.backtest||{}).overall||{},e=document.getElementById('h-rates');if(e&&o.ats_open&&o.ou_open&&o.ats_open.pct!=null&&o.ou_open.pct!=null){const N=(D.report.backtest||{}).nested||{},a=N.spread&&N.spread.total,c=N.total&&N.total.total;e.textContent=`${pct(o.ats_open.pct)} against the spread (${o.ats_open.w}–${o.ats_open.l}) and ${pct(o.ou_open.pct)} on totals (${o.ou_open.w}–${o.ou_open.l})`+(a&&c&&a.pct!=null&&c.pct!=null?`. Those seasons also helped choose the 7-point rule; tested honestly (each season's rule picked only from earlier seasons) it's closer to ${pct(a.pct)} and ${pct(c.pct)}, a more realistic expectation`:'');}})();
"""

REPORT_JS = r"""
document.getElementById('sub').textContent=`${D.season} season · ${freshTxt()}`;
const f1=x=>x==null?'—':x.toFixed(1);const s1=x=>x==null?'—':(Math.abs(x)<0.05?'0.0':(x>0?'+':'−')+Math.abs(x).toFixed(1));
const R=D.report,L=R.live||{},B=R.backtest||{};
const HS=window.HIST||[];  // every graded game this season (history.js)
const rp=r=>!r||(r.w+r.l+r.p)===0?'<span class="muted">—</span>':`<span class="nw">${r.w}–${r.l}${r.p?`<span class="when">–${r.p}</span>`:''}</span> <span class="when nw">${pct(r.pct)}</span>`;
// 95% Wilson range for a win rate (pushes excluded), colored against the 52.4% break-even at -110
const wil=r=>{const n=r.w+r.l;if(!n)return null;const z=1.96,p=r.w/n,d=1+z*z/n,c=p+z*z/(2*n),h=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n));return [(c-h)/d,(c+h)/d];};
const ci=r=>{const w=r&&wil(r);if(!w)return '';const cls=w[0]>0.524?'up':w[1]<0.524?'down':'warnc';return `<span class="ci ${cls}">range ${Math.round(w[0]*100)}–${Math.round(w[1]*100)}%</span>`;};
const rpc=r=>!r||(r.w+r.l+r.p)===0?'<span class="muted">—</span>':`<span class="nw">${r.w}–${r.l}</span><span class="when">${pct(r.pct)}${r.p?` · ${r.p.toLocaleString()} push${r.p>1?'es':''}`:''}</span>`+((r.w+r.l)?ci(r):'');
const rtxt=r=>`${r.w}–${r.l}${r.p?'–'+r.p:''}`;
const card=(l,v)=>`<div class="card"><div class="l">${l}</div><div class="v">${v}</div></div>`;
document.getElementById('live').innerHTML=L.graded?[
 card('Games graded',L.graded),card('Avg miss, team score',f1(L.team_mae)),
 card('Margin miss <span class="when" style="display:inline">model / market</span>',`${f1(L.matched_margin_mae??L.margin_mae)} <span class="sub">/ ${f1(L.matched_mkt_margin_mae??L.mkt_margin_mae)}</span>${L.matched_n?`<span class="when">${L.matched_n} games with a line</span>`:''}`),
 card('Total miss <span class="when" style="display:inline">model / market</span>',`${f1(L.matched_total_mae??L.total_mae)} <span class="sub">/ ${f1(L.matched_mkt_total_mae)}</span>`),
 card('All picks ATS',rpc(L.ats_all)),card('All picks O/U',rpc(L.ou_all)),
 card('Win probability <span class="when" style="display:inline">Brier</span>',L.brier==null?'—':`${L.brier.toFixed(3)}${L.brier_market!=null?`<span class="when">line-implied ${L.brier_market.toFixed(3)} · ${L.brier_n} games</span>`:`<span class="when">${L.brier_n} games · lower is better</span>`}`)].join('')
 :card('Games graded','0')+'<p class="sub" style="grid-column:1/-1">Results appear here once games the model projected before kickoff are final.</p>';
// Spread and total scorecards: record, profit, risk and closing-line value, next to what the backtest led us to expect
(()=>{const o=B.overall||{};
 const one=(k,lab,bt)=>{const x=L[k]||{};const n=(x.w||0)+(x.l||0);const w=n?wil(x):null;
  const u=x.units,roi=x.roi;const kv=(a,b)=>`<div class="kv"><span>${a}</span><span>${b}</span></div>`;
  return `<div class="scard"><h3>${lab}</h3><div class="big2">${n||x.p?`${x.w}–${x.l}${x.p?`<span class="when" style="display:inline">–${x.p}</span>`:''}`:'No bets yet'} ${n?`<span class="when" style="display:inline">${pct(x.pct)}</span>`:''}</div>
   ${n?`<div class="sub" style="margin-bottom:6px">${w?`95% range ${Math.round(w[0]*100)}–${Math.round(w[1]*100)}% · `:''}52.4% breaks even${bt&&bt.pct!=null?` · backtest ${pct(bt.pct)}`:''}</div>`:''}
   ${kv('Profit',u==null?'—':`<span class="${u>0?'up':u<0?'down':''}">${u>=0?'+':'−'}${Math.abs(u).toFixed(2)} units</span>`)}
   ${kv('Plausible profit range',x.units_range?`${x.units_range[0]>=0?'+':'−'}${Math.abs(x.units_range[0]).toFixed(1)} to ${x.units_range[1]>=0?'+':'−'}${Math.abs(x.units_range[1]).toFixed(1)} units`:'<span class="muted">needs 10+ bets</span>')}
   ${kv('Return on risk (ROI)',roi==null?'—':`${roi>=0?'+':'−'}${Math.abs(roi*100).toFixed(1)}%`)}
   ${kv('Worst drop from a high',x.max_drawdown==null?'—':`${x.max_drawdown.toFixed(2)} units`)}
   ${kv('Longest losing streak',x.losing_streak==null?'—':x.losing_streak)}
   ${kv('Closing-line value',x.clv_n?`avg ${s1(x.clv_avg)} · median ${s1(x.clv_median)} pts`:'—')}
   ${kv('Beat the last line',x.clv_n?`${pct(x.clv_beat)} of ${x.clv_n}${x.clv_push?` · ${pct(x.clv_push)} same`:''}`:'—')}</div>`;};
 document.getElementById('scorecards').innerHTML=one('spread','Spread bets',o.ats_open_live||o.ats_open)+one('total','Total bets',o.ou_open_live||o.ou_open);})();
// More live checks: published probabilities, flags the cautions blocked, and when value was captured
(()=>{const parts=[];const kv=(a,b)=>`<div class="kv"><span>${a}</span><span>${b}</span></div>`;
 const pub=(x,lab)=>x?kv(lab,`said ${pct(x.avg_p)}, happened ${pct(x.actual)} <span class="when" style="display:inline">${x.n} games · Brier ${x.brier.toFixed(3)}</span>`):'';
 const P=pub(L.published_spread,'Model side covers (every game)')+pub(L.published_total,'Model side on the total (every game)');
 if(P)parts.push(`<div class="chart"><h4>Published probabilities vs results</h4>${P}<div class="when" style="margin-top:6px">Close to 50% is expected against sharp last lines; the check is that "said" and "happened" stay close.</div></div>`);
 const bl=L.blocked||{};const bn=r=>r&&(r.w+r.l+r.p)?`${rec(r)} <span class="when" style="display:inline">${pct(r.pct)}</span>`:'—';
 if(bl.spread||bl.total)parts.push(`<div class="chart"><h4>Shadow record: flags the cautions blocked</h4>${kv('Spreads',bn(bl.spread))}${kv('Totals',bn(bl.total))}<div class="when" style="margin-top:6px">7+ point gaps that weren't bets because of a QB change, weather, a 10+ gap, an FCS team or early season. Graded at the last line the model saw. Not counted as bets; it shows whether the cautions are earning their keep.</div></div>`);
 const fc=L.flag_clv||{};if(fc.spread||fc.total){const t=x=>x?`avg ${s1(x.avg)} pts · beat the line ${pct(x.beat)} <span class="when" style="display:inline">${x.n}</span>`:'—';
  parts.push(`<div class="chart"><h4>Closing-line value: bets vs blocked flags</h4>${kv('Spread bets',t(L.spread&&L.spread.clv_n?{avg:L.spread.clv_avg,beat:L.spread.clv_beat,n:L.spread.clv_n}:null))}${kv('Spread flags',t(fc.spread))}${kv('Total bets',t(L.total&&L.total.clv_n?{avg:L.total.clv_avg,beat:L.total.clv_beat,n:L.total.clv_n}:null))}${kv('Total flags',t(fc.total))}<div class="when" style="margin-top:6px">Flags are measured from the line when they were first flagged (from the decision log).</div></div>`);}
 const cd=L.clv_by_day||[];
 if(cd.length)parts.push(`<div class="chart"><h4>Closing-line value by day the bet was flagged</h4>${cd.map(x=>kv(x.day,`avg ${s1(x.avg)} pts · beat the line ${pct(x.beat)} <span class="when" style="display:inline">${x.n} bets</span>`)).join('')}<div class="when" style="margin-top:6px">Positive = the line moved toward the bet after it was flagged, so there was value in betting it early.</div></div>`);
 const sh=L.shadow_thresholds||[];
 if(sh.length){const cell=(t,k)=>{const x=sh.find(y=>y.threshold===t&&y.kind===k);return x&&(x.w+x.l)?`${x.w}–${x.l} <span class="when" style="display:inline">${pct(x.pct)}</span>`:'—';};
  parts.push(`<div class="chart"><h4>Shadow record: other gap sizes this season</h4><div class="tbl"><table><thead><tr><th>Gap</th><th>Spreads</th><th>Totals</th></tr></thead><tbody>${[5,6,7,8,9].map(t=>`<tr${t===7?' style="font-weight:600"':''}><td>${t}+ pts${t===7?' (the rule)':''}</td><td>${cell(t,'spread')}</td><td>${cell(t,'total')}</td></tr>`).join('')}</tbody></table></div><div class="when" style="margin-top:6px">Every game this season graded against the last line before kickoff, with no cautions applied. A frozen comparison: it shows whether a different gap would be doing better, without changing the rule mid-season.</div></div>`);}
 const sg=L.segments||[];if(sg.length)parts.push(`<div class="chart"><h4>Bets by game type</h4>${sg.map(x=>kv(x.label,(x.w+x.l+x.p)?`${x.w}–${x.l}${x.p?'–'+x.p:''} · ${x.units>=0?'+':'−'}${Math.abs(x.units).toFixed(2)}u`:'—')).join('')}${L.same_game?`<div class="when" style="margin-top:6px">${L.same_game} game(s) had both a spread and a total bet; their results tend to move together.</div>`:''}</div>`);
 const vd=L.value_decay||{},bp=L.best_price;
 if(vd.n||bp)parts.push(`<div class="chart"><h4>Getting the number</h4>${vd.n?kv('Bets whose number got worse after flagging',`${vd.moved} of ${vd.n}`)+kv('Typical time until it did',vd.median_hours!=null?`${vd.median_hours} hours`:'—')+kv('Moved within a day',`${vd.within_24h}`):''}${bp?kv('Profit at the best book’s line and price',`${bp.units>=0?'+':'−'}${Math.abs(bp.units).toFixed(2)} units (${bp.n} bets)`):''}<div class="when" style="margin-top:6px">"Got worse" = the line moved half a point or more toward the model's side, so anyone betting later got a worse number (good for a bet already placed). If that usually happens within a day, bet soon after the flag.</div></div>`);
 document.getElementById('livemore').innerHTML=parts.length?`<div class="chartgrid" style="margin-top:12px">${parts.join('')}</div>`:'';})();
const mm=(a,b)=>`${f1(a)} <span class="when">/ ${f1(b)}</span>`;
const bt=(B.seasons||[]).map(s=>`<tr><td class="nw">${s.season}${s.holdout?' <span class="pill p-acc" title="Never used for tuning">holdout</span>':''}</td><td>${s.games}</td><td class="nw">${mm(s.margin_mae,s.mkt_margin_mae)}</td><td class="hide-sm">${f1(s.total_mae)}</td><td>${rp(s.ats_open)}</td><td>${rp(s.ats)}</td><td>${rp(s.ou_open)}</td><td class="hide-sm">${rp(s.ou)}</td><td class="hide-sm">${rp(s.ats_all)}</td></tr>`);
const o=B.overall;if(o)bt.push(`<tr><td><b>All</b></td><td>${o.games}</td><td class="nw">${mm(o.margin_mae,o.mkt_margin_mae)}</td><td class="hide-sm">${f1(o.total_mae)}</td><td><b>${rp(o.ats_open)}</b></td><td><b>${rp(o.ats)}</b></td><td><b>${rp(o.ou_open)}</b></td><td class="hide-sm"><b>${rp(o.ou)}</b></td><td class="hide-sm"><b>${rp(o.ats_all)}</b></td></tr>`);
document.getElementById('bt').innerHTML=bt.join('')||'<tr><td colspan="9" class="muted">Backtest not available yet.</td></tr>';
const PR=B.profit||{};
const bootT=r=>r?`<span class="ci" style="color:var(--muted)">week-grouped ${Math.round(r[0]*100)}–${Math.round(r[1]*100)}%</span>`:'';
const profT=k=>{const x=PR[k];return x&&x.range?`<span class="ci" style="color:var(--muted)">${x.units>=0?'+':'−'}${Math.abs(x.units).toFixed(1)} units (plausibly ${x.range[0]>=0?'+':'−'}${Math.abs(x.range[0]).toFixed(0)} to ${x.range[1]>=0?'+':'−'}${Math.abs(x.range[1]).toFixed(0)})</span>`:'';};
const BT=B.boot||{};
if(o){const mv=B.line_move||{};document.getElementById('btcards').innerHTML=[
 card('Bets ATS vs open',rpc(o.ats_open)+bootT(BT.ats_open)+profT('ats_open')+(o.ats_open_live?`<span class="when">approx. live rules: ${rtxt(o.ats_open_live)}${o.ats_open_live.pct!=null?' · '+pct(o.ats_open_live.pct):''}</span>${ci(o.ats_open_live)}${bootT(BT.ats_open_live)}`:'')),card('Bets O/U vs open',rpc(o.ou_open)+bootT(BT.ou_open)+profT('ou_open')+(o.ou_open_live?`<span class="when">approx. live rules: ${rtxt(o.ou_open_live)}${o.ou_open_live.pct!=null?' · '+pct(o.ou_open_live.pct):''}</span>${ci(o.ou_open_live)}${bootT(BT.ou_open_live)}`:'')),
 card('All games ATS vs close',rpc(o.ats_all)),card('All games O/U vs close',rpc(o.ou_all)),
 card('Lines moving toward model',mv.pct==null?'—':pct(mv.pct)+` <span class="sub">of ${mv.n}</span>`),
 card('Avg margin miss, same games (model / market)',`${f1(o.matched_margin_mae??o.margin_mae)} <span class="sub">/ ${f1(o.mkt_margin_mae)}</span>${o.matched_n?`<span class="when">${o.matched_n.toLocaleString()} games with a closing line</span>`:''}`)].join('');}
(()=>{const cb=B.calibration||[];const el=document.getElementById('calchart');if(!cb.length){el.innerHTML='<div class="sub">Not enough games yet.</div>';return;}
 const S=260,P=34,X=v=>P+(S-P-18)*v,Y=v=>S-P-(S-P-8)*v,mx=Math.max(...cb.map(c=>c.n));
 const ticks=[0,.25,.5,.75,1].map(v=>`<line x1="${X(v)}" x2="${X(v)}" y1="8" y2="${S-P}" stroke="var(--line)"/><line x1="${P}" x2="${X(1)}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${X(v)}" y="${S-P+14}" text-anchor="middle" font-size="10" fill="var(--faint)">${v*100}%</text><text x="${P-5}" y="${Y(v)+3}" text-anchor="end" font-size="10" fill="var(--faint)">${v*100}%</text>`).join('');
 const dots=cb.map(c=>`<circle cx="${X(c.predicted)}" cy="${Y(c.actual)}" r="${(3+9*Math.sqrt(c.n/mx)).toFixed(1)}" fill="var(--series)" fill-opacity=".75" stroke="var(--card)" stroke-width="2"><title>Model said ${c.bin} (avg ${pct(c.predicted)}): covered ${pct(c.actual)} of ${c.n} games</title></circle>`).join('');
 el.innerHTML=`<svg viewBox="0 0 ${S} ${S}" style="max-width:320px;margin:auto" role="img" aria-label="Calibration: predicted versus actual cover rate">${ticks}<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="var(--faint)" stroke-dasharray="4 4"/>${dots}<text x="${(S+P)/2}" y="${S-4}" text-anchor="middle" font-size="11" fill="var(--muted)">model's raw cover chance</text></svg>`;})();
document.getElementById('cal').innerHTML=(B.calibration||[]).map(c=>`<tr><td>${c.bin}</td><td>${pct(c.predicted)}</td><td>${pct(c.actual)}</td><td>${c.n}</td></tr>`).join('')||'<tr><td colspan="4" class="muted">—</td></tr>';
document.getElementById('shrink').textContent=B.shrink_spread!=null?`Learned from this table: cover probabilities shown on the projections page keep ${pct(B.shrink_spread)} of the model's raw confidence for spreads and ${pct(B.shrink_total)} for totals.`:'';
// Season-to-season stability of the bet rule
(()=>{const S=(B.seasons||[]).filter(x=>x.ats_open&&(x.ats_open.w+x.ats_open.l)>=10);const T=(B.seasons||[]).filter(x=>x.ou_open&&(x.ou_open.w+x.ou_open.l)>=10);
 const up=a=>a.filter(x=>x.pct>0.524).length;
 if(S.length||T.length)document.getElementById('stability').textContent=`Season by season (10+ bets): spread bets beat 52.4% against the opening line in ${up(S.map(x=>x.ats_open))} of ${S.length} seasons, total bets in ${up(T.map(x=>x.ou_open))} of ${T.length}.`;
 const cv=(B.seasons||[]).filter(x=>x.coverage);
 document.getElementById('coverage').innerHTML=cv.length?`<div class="tbl"><table><thead><tr><th>Season</th><th>Games modeled</th><th>Opening spread</th><th>Closing spread</th><th>Opening total</th><th>Closing total</th></tr></thead><tbody>${cv.map(x=>`<tr><td>${x.season}</td><td>${x.games}</td><td>${x.coverage.open_spread}</td><td>${x.coverage.close_spread}</td><td>${x.coverage.open_total}</td><td>${x.coverage.close_total}</td></tr>`).join('')}</tbody></table></div><p class="sub">Games without a line can't be graded, so seasons with few opening lines carry little weight in the bet records.</p>`:'<div class="sub">Not available.</div>';})();
// Win rate by gap size, with 95% ranges, against the 52.4% break-even line
(()=>{const E=B.edge_buckets||{};
 const chart=(rows,id)=>{const el=document.getElementById(id);const r=(rows||[]).filter(x=>x.w+x.l>0);if(!r.length){el.innerHTML='<div class="sub">Not enough games.</div>';return;}
  const W=520,H=210,P=40,lo=0.3,hi=0.8,X=i=>P+(W-P-10)*(i+0.5)/r.length,Y=v=>8+(H-36)*(hi-Math.min(hi,Math.max(lo,v)))/(hi-lo);
  const grid=[0.3,0.4,0.5,0.6,0.7,0.8].map(v=>`<line x1="${P}" x2="${W-10}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${P-5}" y="${Y(v)+3}" text-anchor="end" font-size="10" fill="var(--faint)">${Math.round(v*100)}%</text>`).join('');
  const be=`<line x1="${P}" x2="${W-10}" y1="${Y(0.524)}" y2="${Y(0.524)}" stroke="var(--warn)" stroke-dasharray="4 3"/><text x="${P+4}" y="${Y(0.524)-4}" font-size="10" fill="var(--warn)">52.4% break-even</text>`;
  const pts=r.map((x,i)=>{const w=wil(x);return `${w?`<line x1="${X(i)}" x2="${X(i)}" y1="${Y(w[0])}" y2="${Y(w[1])}" stroke="var(--series)" stroke-width="2" opacity=".45"/>`:''}<circle cx="${X(i)}" cy="${Y(x.pct)}" r="4.5" fill="var(--series)"><title>${x.bucket} pts: ${x.w}–${x.l} (${pct(x.pct)})</title></circle><text x="${X(i)}" y="${H-16}" text-anchor="middle" font-size="10" fill="var(--muted)">${x.bucket}</text><text x="${X(i)}" y="${H-4}" text-anchor="middle" font-size="9" fill="var(--faint)">n=${x.w+x.l}</text>`;}).join('');
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Win rate by gap size">${grid}${be}${pts}</svg>`;};
 chart(E.spread_open,'ebs');chart(E.total_open,'ebt');
 const labs=(E.spread_open||E.spread_close||[]).map(x=>x.bucket);
 const cell=(rows,k)=>{const x=(rows||[]).find(y=>y.bucket===k);return x&&(x.w+x.l)?`${x.w}–${x.l} <span class="when" style="display:inline">${pct(x.pct)}</span>`:'<span class="muted">—</span>';};
 document.getElementById('ebtbl').innerHTML=labs.map(k=>`<tr><td>${k} pts</td><td>${cell(E.spread_open,k)}</td><td>${cell(E.spread_close,k)}</td><td>${cell(E.total_open,k)}</td><td>${cell(E.total_close,k)}</td></tr>`).join('')||'<tr><td colspan="5" class="muted">Not available.</td></tr>';
 const C2=B.cautions||[];const rc=r=>r&&(r.w+r.l)?`${r.w}–${r.l}${r.p?`<span class="when" style="display:inline">–${r.p}</span>`:''} <span class="when" style="display:inline">${pct(r.pct)}</span>${ci(r)}`:'<span class="muted">—</span>';
 document.getElementById('cautions').innerHTML=C2.map(c=>`<tr><td>${esc(c.label)}</td><td>${rc(c.spread)}</td><td>${rc(c.total)}</td></tr>`).join('')||'<tr><td colspan="3" class="muted">Not available.</td></tr>';
 if(C2.length)document.getElementById('cautionnote').textContent='Groups overlap (an FCS game is usually also early season). These are the same seasons that shaped the rules, so small differences mean little; the shadow record in "This season" is the cleaner test going forward.';
 const N=B.nested||{};const nrow=(lab,x)=>!x||!(x.seasons||[]).length?'':`<tr><td>${lab}</td><td>${x.seasons.map(y=>y.season).join(', ')}</td><td>${x.seasons.map(y=>`${y.season}: ${y.threshold}+`).join(' · ')}</td><td>${rc(x.total)}</td><td class="${x.units>0?'up':x.units<0?'down':''}">${x.units>=0?'+':'−'}${Math.abs(x.units).toFixed(1)}</td></tr>`;
 document.getElementById('nested').innerHTML=[nrow('Spreads, all flags',N.spread),nrow('Spreads, with today\'s cautions',N.spread_live),nrow('Totals, all flags',N.total),nrow('Totals, with today\'s cautions',N.total_live)].join('')||'<tr><td colspan="5" class="muted">Not enough seasons with opening lines yet (each season needs 30+ earlier bets to choose from).</td></tr>';})();
// Win probability calibration (equal-size groups) and accuracy against the line's implied probability
(()=>{const W2=B.win_prob||{},bins=W2.bins||[],el=document.getElementById('wpchart');
 if(!bins.length){el.innerHTML='<div class="sub">Not enough games yet.</div>';return;}
 const S=260,P=44,X=v=>P+(S-P-18)*v,Y=v=>S-P-(S-P-8)*v,mx=Math.max(...bins.map(c=>c.n));
 const ticks=[0,.25,.5,.75,1].map(v=>`<line x1="${X(v)}" x2="${X(v)}" y1="8" y2="${S-P}" stroke="var(--line)"/><line x1="${P}" x2="${X(1)}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${X(v)}" y="${S-P+14}" text-anchor="middle" font-size="10" fill="var(--faint)">${v*100}%</text><text x="${P-5}" y="${Y(v)+3}" text-anchor="end" font-size="10" fill="var(--faint)">${v*100}%</text>`).join('');
 const dots=bins.map(c=>`<circle cx="${X(c.predicted)}" cy="${Y(c.actual)}" r="${(3+6*Math.sqrt(c.n/mx)).toFixed(1)}" fill="var(--series)" fill-opacity=".75" stroke="var(--card)" stroke-width="2"><title>Model said ${pct(c.predicted)}: home team won ${pct(c.actual)} of ${c.n} games</title></circle>`).join('');
 el.innerHTML=`<svg viewBox="0 0 ${S} ${S}" style="max-width:320px;margin:auto;display:block" role="img" aria-label="Win probability calibration">${ticks}<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="var(--faint)" stroke-dasharray="4 4"/>${dots}<text x="${(S+P)/2}" y="${S-4}" text-anchor="middle" font-size="11" fill="var(--muted)">model's home win chance</text><text transform="translate(9,${(S-P)/2}) rotate(-90)" text-anchor="middle" font-size="10" fill="var(--muted)">home team actually won</text></svg>`;
 document.getElementById('wpnote').innerHTML=`Backtest Brier score <b>${W2.brier!=null?W2.brier.toFixed(3):'—'}</b> over ${(W2.n||0).toLocaleString()} games.`+(W2.brier_market!=null?` On the ${(W2.n_matched||0).toLocaleString()} games with a closing line: model ${W2.brier_matched.toFixed(3)} vs the line ${W2.brier_market.toFixed(3)} (the line's win chances use its own error size, ${f1(W2.market_sd)} points, not the model's)${W2.brier_diff_range?`; the gap is ${W2.brier_diff>=0?'+':'−'}${Math.abs(W2.brier_diff).toFixed(4)}, plausibly ${W2.brier_diff_range[0].toFixed(4)} to ${W2.brier_diff_range[1].toFixed(4)}`:''}. A fixed home-team guess scores ${W2.brier_naive!=null?W2.brier_naive.toFixed(3):'—'}.`:'')+(L.brier!=null?` This season, the win chances actually published before kickoff: ${L.brier.toFixed(3)} over ${L.brier_n} games${L.brier_market!=null?` (the last line's: ${L.brier_market.toFixed(3)})`:''}.`:'');})();
// Where the model leans against the market, by conference
(()=>{const c=B.conf_lean||[];if(!c.length)return;document.getElementById('conflean').innerHTML=`<h4 class="sub" style="margin:14px 0 6px">Model vs the closing line, by conference</h4><div class="tbl"><table><thead><tr><th>Conference</th><th>Team-games</th><th>Model minus line</th><th>Actual minus line</th></tr></thead><tbody>${c.map(r=>`<tr><td>${esc(r.group)}</td><td>${r.n}</td><td>${s1(r.model_vs_line)}</td><td>${s1(r.actual_vs_line)}</td></tr>`).join('')}</tbody></table></div><p class="sub">Positive = rates the conference's teams higher than the market (model) or beat the market (actual). A conference where the model leans one way and results go the other is a blind spot; the service academies were one, now corrected.</p>`;})();
// Accuracy table: backtest and this season, model and market, margin and total
(()=>{const A=B.accuracy||{},LA=L.accuracy||{},cv=B.coverage||{},lc=L.coverage||{};
 const r=(lab,x,c)=>x&&x.n?`<tr><td>${lab}</td><td>${f1(x.mae)}</td><td>${f1(x.rmse)}</td><td>${s1(x.bias)}</td><td>${c!=null?pct(c):'<span class="muted">—</span>'}</td><td>${x.n.toLocaleString()}</td></tr>`:'';
 document.getElementById('acc').innerHTML=[r('Margin, model (backtest)',A.margin,cv.margin80),r('Margin, closing line (backtest)',A.market_margin),r('Total, model (backtest)',A.total,cv.total80),r('Total, closing line (backtest)',A.market_total),
  r('Margin, model (this season)',LA.margin,lc.margin80),r('Margin, last line (this season)',LA.market_margin),r('Total, model (this season)',LA.total,lc.total80),r('Total, last line (this season)',LA.market_total)].join('')||'<tr><td colspan="6" class="muted">Not available.</td></tr>';
 const W2=B.win_prob||{};const ws=W2.by_season||[];
 document.getElementById('wpseason').innerHTML=ws.length?`<div class="tbl"><table><thead><tr><th>Season</th><th>Brier</th><th>Log loss</th><th>Games</th></tr></thead><tbody>${ws.map(x=>`<tr><td>${x.season}</td><td>${x.brier.toFixed(3)}</td><td>${x.logloss.toFixed(3)}</td><td>${x.n}</td></tr>`).join('')}</tbody></table></div><p class="sub">Log loss punishes confident misses harder than Brier. Lower is better for both.</p>`+((W2.by_matchup||[]).length?`<div class="tbl" style="margin-top:8px"><table><thead><tr><th>Matchup</th><th>Brier</th><th>Games</th></tr></thead><tbody>${W2.by_matchup.map(x=>`<tr><td>${esc(x.group)}</td><td>${x.brier.toFixed(3)}</td><td>${x.n}</td></tr>`).join('')}</tbody></table></div>`:''):'<div class="sub">Not available.</div>';})();
const E=R.experiments||{};
if(E.tests){
 const ch=(v,base,rg)=>v==null?'—':`<span class="${v>0?'up':v<0?'down':''}">${v>0?'−':v<0?'+':''}${Math.abs(v).toFixed(3)}</span>${base!=null?` <span class="when" style="display:inline">${base.toFixed(2)} → ${(base-v).toFixed(2)}</span>`:''}${rg&&rg.length===2?`<span class="when" style="display:block">95% range ${-rg[1]>=0?'+':'−'}${Math.abs(rg[1]).toFixed(3)} to ${-rg[0]>=0?'+':'−'}${Math.abs(rg[0]).toFixed(3)}${rg[0]>0?' (real)':' (could be chance)'}</span>`:''}`;
 document.getElementById('expsub').textContent=`Each feature is tested on ${E.tune_seasons?E.tune_seasons.join('–'):'the tuning seasons'} against the core model (margin MAE ${f2(E.baseline&&E.baseline.margin)}, total MAE ${f2(E.baseline&&E.baseline.total)}). It's kept only if the gain is real (at least 0.01 points, with the 95% range of the gain, resampling whole weeks, above zero), the other measure is no worse, and it doesn't cost bets (units on 7+ point gaps against opening lines can't drop by more than chance would explain). Then it must also not make the newest complete season, which played no part in that test, clearly worse. Green means less error. Last tested ${new Date(E.run).toLocaleDateString()}.`;
 document.getElementById('exp').innerHTML=E.tests.map(t=>`<tr><td>${esc(t.feature)}</td><td>${t.skipped?'—':ch(t.gain_margin,E.baseline&&E.baseline.margin,t.range_margin)}</td><td>${t.skipped?'—':ch(t.gain_total,E.baseline&&E.baseline.total,t.range_total)}</td><td>${t.skipped?`<span class="muted">${/^switched off/i.test(t.skipped)?esc(t.skipped[0].toUpperCase()+t.skipped.slice(1)):'Not tested: '+esc(t.skipped)}</span>`:t.kept?'<span class="pill p-good">Kept</span>':t.ok?'<span class="muted">Passed, but replaced by a better version that includes it</span>':'<span class="muted">Not kept</span>'}${!t.skipped&&t.bets&&t.bets_ref?`<span class="when" style="display:block;margin-top:4px">7+ bets vs open: ${t.bets.w}–${t.bets.l} (${t.bets.units>=0?'+':'−'}${Math.abs(t.bets.units).toFixed(1)}u) vs ${t.bets_ref.w}–${t.bets_ref.l} (${t.bets_ref.units>=0?'+':'−'}${Math.abs(t.bets_ref.units).toFixed(1)}u) without</span>`:''}${!t.skipped&&t.confirm?`<span class="when" style="display:block">${t.confirm.season} check${t.confirm_ok?'':' (failed)'}: margin error ${t.confirm.gain_margin>=0?'−':'+'}${Math.abs(t.confirm.gain_margin).toFixed(3)}, total ${t.confirm.gain_total>=0?'−':'+'}${Math.abs(t.confirm.gain_total).toFixed(3)}${t.confirm.bets_change!=null?`, bets ${t.confirm.bets_change>=0?'+':'−'}${Math.abs(t.confirm.bets_change).toFixed(1)}u`:''}</span>`:''}</td></tr>`).join('');
 const H=E.holdout||{};if(H.baseline&&H.selected)document.getElementById('exphold').textContent=`Confirmation season ${H.season} (never used to tune or choose features, only to veto one that makes it clearly worse): core model margin MAE ${f2(H.baseline.margin)} vs with kept features ${f2(H.selected.margin)}; total MAE ${f2(H.baseline.total)} vs ${f2(H.selected.total)}.`;
}
const S=B.segments||{};const names={margin_by_week:'Margin by week',margin_by_matchup:'Margin by matchup',margin_by_fav_size:'Margin by projected favorite size',total_by_week:'Total by week',total_by_matchup:'Total by matchup'};
document.getElementById('seg').innerHTML=Object.entries(names).filter(([k])=>S[k]).map(([k,n])=>`<h4 class="sub" style="margin:14px 0 6px">${n}</h4><div class="tbl"><table><thead><tr><th>Group</th><th>Games</th><th>Bias</th><th>MAE</th></tr></thead><tbody>${S[k].map(r=>`<tr><td>${r.group}</td><td>${r.n}</td><td class="${r.flag?'flag':''}">${s1(r.bias)}</td><td>${f1(r.mae)}</td></tr>`).join('')}</tbody></table></div>`).join('');
const pill=v=>v?`<span class="pill ${v==='win'?'p-good':v==='loss'?'p-bad':'p-acc'}">${v}</span>`:'<span class="muted">—</span>';
// Every graded game: bets only by default, with filters, running units and CSV download
(()=>{const H=HS.slice().sort((x,y)=>x.start<y.start?-1:1);
 const u=v=>v==='win'?100/110:v==='loss'?-1:0;let run=0;
 const rows=H.map(p=>{const r=p.result,fs=p.first_spread_flag,ft=p.first_total_flag;
  const sp=spreadPick(p,r.close_spread,p.model_spread),tp=totalPick(r.close_total,p.model_total);
  const sTxt=r.ats&&fs?`${fs.side==='home'?p.home:p.away} ${sl(fs.side==='home'?fs.line:-fs.line)}`:(sp&&!sp.none?`${sp.team} ${sl(sp.line)}`:'');
  const tTxt=r.ou&&ft?`${ft.side==='over'?'Over':'Under'} ${fmt(ft.line)}`:(tp&&!tp.none?`${tp.side} ${tp.line}`:'');
  const sRes=r.ats||r.ats_all,tRes=r.ou||r.ou_all,isBet=!!(r.ats||r.ou);
  if(r.ats)run+=u(r.ats);if(r.ou)run+=u(r.ou);
  return {p,r,sTxt,tTxt,sRes,tRes,sBet:!!r.ats,tBet:!!r.ou,isBet,units:isBet?run:null,clv:[r.clv_spread,r.clv_total].filter(x=>x!=null)};}).reverse();
 let onlyBets=rows.some(x=>x.isBet),res='',hq='',hm='',hw='';
 const hf=document.getElementById('hfilters');
 hf.innerHTML=`<button class="chip${onlyBets?' on':''}" data-v="bets">Bets only</button><button class="chip${onlyBets?'':' on'}" data-v="all">All games</button>
  <select id="hres" class="chip" aria-label="Result"><option value="">Any result</option><option value="win">Wins</option><option value="loss">Losses</option></select>
  <select id="hmkt" class="chip" aria-label="Market"><option value="">Spreads and totals</option><option value="s">Spreads</option><option value="t">Totals</option></select>
  <select id="hwk" class="chip" aria-label="Week"><option value="">Every week</option>${[...new Set(rows.map(x=>x.p.wk))].sort((a,b)=>b-a).map(w=>`<option value="${w}">${wkName(w)}</option>`).join('')}</select>
  <input type="search" id="hq" placeholder="Find a team"><button class="chip" id="csv">Download CSV</button>`;
 const resCell=(txt,v,bet)=>txt?`${bet?'<span class="pill p-good">BET</span> <span class="bet">':''}${esc(txt)}${bet?'</span>':''} ${resPill(v)}`:'<span class="muted">—</span>';
 const keepRow=x=>(!hw||String(x.p.wk)===hw)&&(!hm||(onlyBets?(hm==='s'?x.sBet:x.tBet):true))&&(!onlyBets||x.isBet)&&(!hq||(x.p.home+' '+x.p.away).toLowerCase().includes(hq))&&
   (!res||(onlyBets?((x.sBet&&x.sRes===res)||(x.tBet&&x.tRes===res)):(x.sRes===res||x.tRes===res)));
 function drawH(){const R2=rows.filter(keepRow);
  document.getElementById('hist').innerHTML=R2.map(x=>`<tr><td>${esc(x.p.away)} ${x.p.neutral?'vs':'@'} <span class="home">${esc(x.p.home)}</span><div class="when">${wkName(x.p.wk)}</div></td><td>${x.r.away_pts}–${x.r.home_pts}</td><td class="hide-sm">${projScore(x.p)||'—'}</td>
   <td>${onlyBets&&!x.sBet?'<span class="muted">—</span>':resCell(x.sTxt,x.sRes,x.sBet)}</td><td>${onlyBets&&!x.tBet?'<span class="muted">—</span>':resCell(x.tTxt,x.tRes,x.tBet)}</td>
   <td class="hide-sm">${x.clv.length?x.clv.map(sgn).join(', '):'—'}</td><td class="hide-sm ${x.units==null?'':x.units>=0?'up':'down'}">${x.units==null?'—':(x.units>=0?'+':'−')+Math.abs(x.units).toFixed(2)}</td></tr>`).join('')
   ||`<tr><td colspan="7" class="muted">${onlyBets?'No graded bets yet.':'No graded games yet.'}</td></tr>`;}
 hf.querySelectorAll('[data-v]').forEach(bn=>bn.onclick=()=>{onlyBets=bn.dataset.v==='bets';hf.querySelectorAll('[data-v]').forEach(c=>c.classList.toggle('on',c===bn));drawH();});
 document.getElementById('hres').onchange=e=>{res=e.target.value;drawH();};
 document.getElementById('hmkt').onchange=e=>{hm=e.target.value;drawH();};document.getElementById('hwk').onchange=e=>{hw=e.target.value;drawH();};
 document.getElementById('hq').oninput=e=>{hq=e.target.value.toLowerCase();drawH();};
 document.getElementById('csv').onclick=()=>{const q=v=>`"${String(v??'').replace(/"/g,'""')}"`;
  // One row per game with everything needed to re-check a result: what was predicted, when, at which line
  const lines=[['game_id','kickoff_utc','week','away','home','away_pts','home_pts','model_version','model_spread_home','model_total','projected_score',
   'spread_pick','spread_result','spread_bet','spread_flagged_at','spread_flag_line_home','spread_flag_model','spread_best_book','spread_best_line','spread_best_price',
   'total_pick','total_result','total_bet','total_flagged_at','total_flag_line','total_flag_model','total_best_book','total_best_line','total_best_price',
   'last_line_spread_home','last_line_total','last_line_time','clv_spread','clv_total','units_running'].join(',')];
  rows.filter(keepRow).forEach(x=>{const fs=x.sBet?x.p.first_spread_flag||{}:{},ft=x.tBet?x.p.first_total_flag||{}:{},bs=fs.best||{},bt=ft.best||{};
   lines.push([x.p.game_id,x.p.start,wkName(x.p.wk),x.p.away,x.p.home,x.r.away_pts,x.r.home_pts,(x.p.version||'').split('|')[0],x.p.model_spread,x.p.model_total,projScore(x.p)||'',
    x.sTxt,x.sRes||'',x.sBet?'yes':'',fs.ts||'',fs.line??'',fs.model??'',bs.book||'',bs.line??'',bs.price??'',
    x.tTxt,x.tRes||'',x.tBet?'yes':'',ft.ts||'',ft.line??'',ft.model??'',bt.book||'',bt.line??'',bt.price??'',
    x.r.close_spread??'',x.r.close_total??'',x.r.close_ts||'',x.r.clv_spread??'',x.r.clv_total??'',x.units==null?'':x.units.toFixed(2)].map(q).join(','));});
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv'}));a.download=`cfb-results-${D.season}.csv`;document.body.appendChild(a);a.click();a.remove();};
 drawH();})();
const st=R.state||{};// Charts and weekly table from the graded games
(()=>{const Hs=HS.filter(p=>p.result).sort((a,b)=>a.start<b.start?-1:1);
 const u=v=>v==='win'?100/110:v==='loss'?-1:0;
 const allBets=[];Hs.forEach(p=>{const r=p.result;if(r.ats)allBets.push({p,v:r.ats,k:'spread'});if(r.ou)allBets.push({p,v:r.ou,k:'total'});});
 const W=560,Hh=210,pad=40;let ukind='';
 const uf=document.getElementById('ufilters');
 uf.innerHTML=[['','Combined'],['spread','Spreads'],['total','Totals']].map(([k,l])=>`<button class="chip${k===''?' on':''}" data-u="${k}">${l}</button>`).join('');
 function drawU(){const bets=allBets.filter(b=>!ukind||b.k===ukind);
  if(!bets.length){document.getElementById('units').innerHTML='<div class="sub">No bets graded yet.</div>';return;}
  let c=0;const pts=[[0,0],...bets.map((b,i)=>[i+1,(c+=u(b.v))])];
  const ys=pts.map(p=>p[1]),lo=Math.min(-1,...ys),hi=Math.max(1,...ys),X=i=>pad+(W-pad-8)*i/Math.max(1,bets.length),Y=v=>8+(Hh-pad-8)*(hi-v)/(hi-lo);
  const ticks=[lo,0,hi].map(v=>`<line x1="${pad}" x2="${W-8}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)" ${v===0?'':'stroke-dasharray="3 3"'}/><text x="${pad-6}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--faint)">${v>0?'+':''}${v.toFixed(1)}</text>`).join('');
  const path=pts.map((p,i)=>`${i?'L':'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
  let pk=0;const peak=pts.map((p,i)=>{pk=Math.max(pk,p[1]);return `${i?'L':'M'}${X(p[0]).toFixed(1)},${Y(pk).toFixed(1)}`;}).join('');
  const dd=Math.max(0,...pts.map((p,i)=>Math.max(...pts.slice(0,i+1).map(q=>q[1]))-p[1]));
  const dots=bets.map((b,i)=>`<circle cx="${X(i+1)}" cy="${Y(pts[i+1][1])}" r="3.5" fill="${b.v==='win'?'var(--good)':b.v==='loss'?'var(--bad)':'var(--faint)'}"><title>${esc(b.p.away)} @ ${esc(b.p.home)} · ${wkName(b.p.wk)} · ${b.k} · ${b.v} · running ${pts[i+1][1]>=0?'+':'−'}${Math.abs(pts[i+1][1]).toFixed(2)}</title></circle>`).join('');
  document.getElementById('units').innerHTML=`<svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Profit in units after each bet">${ticks}<path d="${peak}" fill="none" stroke="var(--faint)" stroke-dasharray="2 3"/><path d="${path}" fill="none" stroke="var(--series)" stroke-width="2"/>${dots}<text transform="translate(11,${(Hh-pad)/2}) rotate(-90)" text-anchor="middle" font-size="11" fill="var(--muted)">units</text><text x="${(W+pad)/2}" y="${Hh-6}" text-anchor="middle" font-size="11" fill="var(--muted)">bets in order graded (${bets.length}) · now ${c>=0?'+':'−'}${Math.abs(c).toFixed(2)} units · worst drop ${dd.toFixed(2)} (dotted = previous high)</text></svg>`;}
 uf.querySelectorAll('[data-u]').forEach(bn=>bn.onclick=()=>{ukind=bn.dataset.u;uf.querySelectorAll('[data-u]').forEach(x=>x.classList.toggle('on',x===bn));drawU();});
 drawU();
 const G2=Hs.filter(p=>p.model_spread!=null);
 if(G2.length){const pm=G2.map(p=>-p.model_spread),am=G2.map(p=>p.result.home_pts-p.result.away_pts);const m=Math.max(14,...pm.map(Math.abs),...am.map(Math.abs));const R=Math.ceil(m/14)*14;
  const S=260,X=v=>34+(S-42)*(v+R)/(2*R),Y=v=>8+(S-38)*(R-v)/(2*R);
  const grid=[-R,0,R].map(v=>`<line x1="${X(v)}" x2="${X(v)}" y1="8" y2="${S-30}" stroke="var(--line)"/><line x1="34" x2="${S-8}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${X(v)}" y="${S-16}" text-anchor="middle" font-size="10" fill="var(--faint)">${v}</text><text x="30" y="${Y(v)+3}" text-anchor="end" font-size="10" fill="var(--faint)">${v}</text>`).join('');
  const dots=G2.map((p,i)=>`<circle cx="${X(pm[i])}" cy="${Y(am[i])}" r="3" fill="var(--series)" fill-opacity=".7"><title>${esc(p.away)} @ ${esc(p.home)}: predicted ${pm[i]>0?'+':''}${pm[i].toFixed(1)}, actual ${am[i]>0?'+':''}${am[i]}</title></circle>`).join('');
  document.getElementById('scatter').innerHTML=`<svg viewBox="0 0 ${S} ${S}" style="max-width:320px;margin:auto">${grid}<line x1="${X(-R)}" y1="${Y(-R)}" x2="${X(R)}" y2="${Y(R)}" stroke="var(--faint)" stroke-dasharray="4 4"/>${dots}<text x="${(S+22)/2}" y="${S-2}" text-anchor="middle" font-size="10" fill="var(--muted)">predicted home margin</text><text transform="translate(9,${(S-30)/2}) rotate(-90)" text-anchor="middle" font-size="10" fill="var(--muted)">actual</text></svg><div class="when" style="text-align:center">Dots on the dashed line were predicted exactly; up and down shows how far off the actual margin was.</div>`;}
 else document.getElementById('scatter').innerHTML='<div class="sub">No graded games yet.</div>';
 const wk={};Hs.forEach(p=>{const k=p.wk;(wk[k]=wk[k]||[]).push(p);});
 const r2=(arr)=>{const w=arr.filter(v=>v==='win').length,l=arr.filter(v=>v==='loss').length,pu=arr.filter(v=>v==='push').length;return w+l+pu?`${w}–${l}${pu?'–'+pu:''}`:'—';};
 document.getElementById('weekly').innerHTML=Object.keys(wk).sort((a,b)=>b-a).map(k=>{const ps=wk[k];const bv=ps.flatMap(p=>[p.result.ats,p.result.ou]).filter(Boolean);
  const un=bv.reduce((a,v)=>a+u(v),0);const mm=ps.map(p=>Math.abs(p.result.err_margin)),mk=ps.filter(p=>p.result.mkt_err_margin!=null).map(p=>Math.abs(p.result.mkt_err_margin));
  const avg=a=>a.length?(a.reduce((x,y)=>x+y,0)/a.length).toFixed(1):'—';
  return `<tr><td>${wkName(+k)}</td><td>${r2(bv)}</td><td class="${un>0?'up':un<0?'down':''}">${bv.length?(un>=0?'+':'−')+Math.abs(un).toFixed(2):'—'}</td><td>${r2(ps.map(p=>p.result.ats_all).filter(Boolean))}</td><td>${r2(ps.map(p=>p.result.ou_all).filter(Boolean))}</td><td class="hide-sm">${avg(mm)} / ${avg(mk)}</td></tr>`;}).join('')||'<tr><td colspan="6" class="muted">No graded games yet.</td></tr>';
})();
// At a glance: the three things worth knowing, as headline numbers with one plain sentence each
(()=>{const o=B.overall||{},S=B.seasons||[],H=S.find(x=>x.holdout)||S[S.length-1]||{};
 const nb=(L.ats?L.ats.w+L.ats.l+L.ats.p:0)+(L.ou?L.ou.w+L.ou.l+L.ou.p:0);
 const bw=(L.ats?L.ats.w:0)+(L.ou?L.ou.w:0),bl=(L.ats?L.ats.l:0)+(L.ou?L.ou.l:0),bp=(L.ats?L.ats.p:0)+(L.ou?L.ou.p:0);
 const units=bw*100/110-bl;
 const blk=(lab,big,txt)=>`<div class="gblock"><div class="l">${lab}</div><div class="big2">${big}</div><div class="gtxt">${txt}</div></div>`;
 const b1=L.graded?blk('This season',nb?`${bw}–${bl}${bp?`–${bp}`:''} <span class="when" style="display:inline">bets · ${units>=0?'+':'−'}${Math.abs(units).toFixed(2)} units</span>`:'No bets yet',
   `${L.graded} games graded. Picks on every game: ${rec(L.ats_all)} against the spread, ${rec(L.ou_all)} on totals. `+(L.clv_n?`Bets beat the last line before kickoff ${pct(L.clv_beat_pct)} of the time (${L.clv_n} bets), the best early sign of a real edge. `:'')+(()=>{const w=wil({w:bw,l:bl});if(!w)return '';const r=`${Math.round(w[0]*100)}–${Math.round(w[1]*100)}%`;return w[0]>0.524?`Bets are winning more than break-even even at the low end of their range (${r}).`:w[1]<0.524?`Bets are below break-even even at the high end of their range (${r}).`:`${bw+bl} bets is too few to tell skill from luck yet (true rate anywhere from ${r}).`;})()):blk('This season','—','No games graded yet.');
 const b2=H.margin_mae!=null&&H.mkt_margin_mae!=null?blk('Model vs market',`${f1(H.margin_mae)} <span class="when" style="display:inline">vs ${f1(H.mkt_margin_mae)} pts</span>`,
   `Average miss on the final margin in ${H.season}, a season the model never trained on, vs the closing line. The line is still more accurate overall: it sees injury and depth-chart news the model can't.`):'';
 const w1=o.ats_open&&wil(o.ats_open),w2=o.ou_open&&wil(o.ou_open);
 const N0=B.nested||{},na=N0.spread&&N0.spread.total,nt=N0.total&&N0.total.total,hon=na&&nt&&na.pct!=null&&nt.pct!=null;
 const b3=o.ats_open&&(o.ats_open.w+o.ats_open.l)?blk('Where the edge is',hon?`${pct(na.pct)} <span class="when" style="display:inline">spreads</span> · ${pct(nt.pct)} <span class="when" style="display:inline">totals</span>`:`${pct(o.ats_open.pct)} <span class="when" style="display:inline">spreads</span> · ${pct(o.ou_open.pct)} <span class="when" style="display:inline">totals</span>`,
   (hon?`Realistic win rates for flagged bets against opening lines, testing each season with a rule chosen only from earlier seasons (${rec(na)} and ${rec(nt)}; 52.4% breaks even). The full backtest shows ${pct(o.ats_open.pct)} and ${pct(o.ou_open.pct)} (${rec(o.ats_open)}, ${rec(o.ou_open)}), but those seasons also helped choose the 7-point rule, so it flatters.`:`Backtest record of flagged bets against opening lines (${rec(o.ats_open)} and ${rec(o.ou_open)}; 52.4% breaks even); these seasons also shaped the rules, so expect a bit less.`)+` Against closing lines it's about 50%, so the edge is in betting early.`):'';
 const W2=B.win_prob||{},cov=B.coverage||{};
 const b4=W2.brier!=null?blk('Calibration',`${W2.brier.toFixed(3)} <span class="when" style="display:inline">Brier · line ${W2.brier_market!=null?W2.brier_market.toFixed(3):'—'}</span>`,
   `Win-probability accuracy in the backtest (0 is perfect, 0.25 is a coin flip; a fixed home-team guess scores ${W2.brier_naive!=null?W2.brier_naive.toFixed(3):'—'}). The actual margin landed inside the model's 80% range ${pct(cov.margin80)} of the time, close to the 80% it should.`):'';
 document.getElementById('glance').innerHTML=`<div class="glance">${b1}${b2}${b4}${b3}</div>`;})();
// Every bet on this page must add up to the season totals at the top
(()=>{const t={spread:{w:0,l:0,p:0},total:{w:0,l:0,p:0}};let un=0;
 HS.forEach(p=>{const r=p.result||{};for(const [k,c] of [['spread','ats'],['total','ou']]){const v=r[c];if(!v)continue;t[k][v==='win'?'w':v==='loss'?'l':'p']++;un+=v==='win'?100/110:v==='loss'?-1:0;}});
 const same=k=>{const a=L[k]||{};return (a.w||0)===t[k].w&&(a.l||0)===t[k].l&&(a.p||0)===t[k].p;};
 const lu=(L.spread&&L.spread.units||0)+(L.total&&L.total.units||0),n=['spread','total'].reduce((a,k)=>a+t[k].w+t[k].l+t[k].p,0);
 const ok=same('spread')&&same('total')&&Math.abs(un-lu)<0.02;
 document.getElementById('recon').innerHTML=!L.graded?'':`<div class="okbox${ok?'':' bad'}">${ok?`✓ Checked: all ${n} graded bets in the game list add up to the season totals (${HS.length} graded games).`:`⚠ The game list (${t.spread.w}–${t.spread.l} spreads, ${t.total.w}–${t.total.l} totals) doesn't match the season totals. The next update should fix it; if not, something is off in the data.`}</div>`;})();
// Data health: what the latest run had to work with
(()=>{const h=D.health||{};const el=document.getElementById('health');if(!h.run){el.innerHTML='<div class="sub">Not available yet.</div>';return;}
 const age=D.lines_updated?Math.round((Date.now()-new Date(D.lines_updated))/36e5):null;
 const row=(a,b,warn)=>`<tr><td>${a}</td><td class="${warn?'flag':''}">${b}</td></tr>`;
 el.innerHTML=`<div class="tbl"><table><tbody>
  ${row('Last run',`${new Date(h.run).toLocaleString('en-US',{timeZone:'America/Chicago',dateStyle:'medium',timeStyle:'short'})} CT · ${h.mode==='odds'?'lines-only update':'full update'}${h.build?` · build ${esc(h.build)}`:''}`)}
  ${row('Model version',`v${h.model_version} · bets need ${(D.thresholds||{}).spread}+ points (spreads) / ${(D.thresholds||{}).total}+ (totals)${h.cautions?` · early season ${h.cautions.early_season_blocks_spreads?'blocks':'doesn\'t block'} spread bets · FCS games ${h.cautions.fcs_blocks_spreads?'block':'don\'t block'} spread bets`:''}`)}
  ${row('This week\'s games',`${h.slate_games} on the slate · ${h.projected} projected${h.skipped_started?` · ${h.skipped_started} already started`:''}`)}
  ${row('Lines',`${h.with_spread} with a spread, ${h.with_total} with a total${age!=null?` · newest snapshot ${age} h old`:''}`,age!=null&&age>24&&h.slate_games>0)}
  ${row('Odds feed',`${h.odds_events} events, ${h.odds_rows} book lines this run${h.unmatched_odds?` · ${h.unmatched_odds} couldn't be matched to a game`:''}`,h.slate_games>0&&!h.odds_events)}
  ${row('Teams and games',`${h.rated_teams} FBS teams rated · ${h.completed_games} of ${h.schedule_games} season games final`)}
  ${row('Graded this season',`${h.graded} games · ${h.bets_graded} bets`)}
  ${row('Lines that can\'t become bets',`${h.backup_lines||0} games on the backup line only · ${h.stale_quotes||0} with quotes over 36 h old`,(h.backup_lines||0)+(h.stale_quotes||0)>0)}
  ${row('Odds-matching notes',(h.oddities||[]).length?(h.oddities||[]).map(esc).join('<br>'):'none',(h.oddities||[]).length>0)}
  ${row('Automatic corrections',h.academy&&h.academy.n?`Service academies ${h.academy.margin>=0?'+':'−'}${Math.abs(h.academy.margin).toFixed(1)} pts margin, ${h.academy.total>=0?'+':'−'}${Math.abs(h.academy.total).toFixed(1)} total (from ${h.academy.n} out-of-sample games)`:'none')}
  ${row('Model fit',h.fit_id?`fingerprint ${esc(h.fit_id)} (saved with every prediction and archived daily)`:'—')}
 </tbody></table></div><p class="sub">Before publishing, every run checks that each projection is a real number in a sensible range and that team ratings exist; if not, it stops and keeps the last good version of the site.</p>`;})();
// What the model is, and what changed when
document.getElementById('changelog').innerHTML=`<ul>
 <li><b>Current model (v${(D.health||{}).model_version||5})</b>: opponent-adjusted team ratings (points, efficiency, pace) with preseason priors from returning production, QB quality and transfers; a points conversion; probabilities from a normal curve sized by past out-of-sample misses, and a score simulation for likely scores and push chances. Sportsbook lines are never inputs.</li>
 <li><b>Bet rules</b>: a 7+ point gap with the line on the spread or total, no blocking caution (QB change, manual adjustment, FCS team, early season, wind/rain, or a 10+ point gap). Bets keep the line they were first flagged at.</li>
 <li><b>Oct 23, 2026 (later)</b>: new keep rule for model inputs (the gain must be real, must not cost bets against opening lines, and must hold up on the newest season; the current inputs pass, so projections are unchanged); QB changes replayed in the backtest's caution table; Sunday line snapshots to catch opening lines; edge column shows today's edge for moved bets; moved lines say whether they went for or against the bet. Bet rules unchanged.</li>
 <li><b>Oct 23, 2026</b>: service-academy correction (they beat projections in all 9 backtest seasons); spreads and totals now equal the expected points the backtest validated (the simulation had been adding about half a point to totals); probabilities use out-of-sample error sizes; bets require a live sportsbook quote under 36 hours old; best book chosen by expected value; decision log, rating ranges, accuracy and profit ranges, shadow thresholds. Bet rules unchanged (7 points, same cautions).</li>
 <li><b>Oct 2026</b>: Matchup page; Performance page with spread/total scorecards, ROI, drawdown, edge-size and caution analysis, an out-of-sample threshold test, win-probability calibration and data health; best available book on every bet; sportsbook prices recorded with each line; complete season history; rating history by week. No change to the model or the bet rules.</li>
</ul>`;
document.getElementById('settings').innerHTML=`Prior strength ${st.lam} (games' worth, tuned ${st.tuned_at?new Date(st.tuned_at).toLocaleDateString():'—'}) · Game-script correlation ${f1(st.rho)} · Margin error SD ${f1(st.margin_sd)} · Total error SD ${f1(st.total_sd)}<br>Tuning results (margin MAE by prior strength): ${Object.entries(R.tune||{}).map(([k,v])=>`${k}: ${f2(v)}`).join(' · ')||'—'}`;
"""

MODEL_FN = r"""
// Same formula as the weekly predictions: expected stat for each side, then the points conversion
function sideX(f,side,h){const MD=window.MODEL||{};const g=k=>f[k+'_'+side];const P=g('p'),Q=g('q'),S=g('s'),z=v=>v==null||isNaN(v)?0:v;
 const cols=[1,P,P*Q,P*S,h];
 for(const grp of MD.groups||[]){if(grp==='score')cols.push(z(g('r')));
  if(grp==='pbp')cols.push(z(g('d'))*P/10,z(g('fn')),z(g('fp')));}
 return cols;}
function project(home,away,neutral){const h=neutral?0:1,f={};const MD=window.MODEL||{};
 for(const [k,m] of Object.entries(MD.metrics||{})){
  const o=t=>m.off[t]??0,d=t=>m.deff[t]??0;
  f[k+'_h']=m.mu+o(home)+d(away)+m.hfa*h;f[k+'_a']=m.mu+o(away)+d(home)-m.hfa*h;}
 const dot=x=>x.reduce((a,v,i)=>a+v*MD.conv[i],0);
 let ph=dot(sideX(f,'h',h)),pa=dot(sideX(f,'a',-h));
 const tm=MD.total_model;  // optional separate totals model (matches the weekly predictions when switched on)
 if(tm&&tm.beta){const P=f.p_h+f.p_a,PQ=f.p_h*f.q_h+f.p_a*f.q_a,PS=f.p_h*f.s_h+f.p_a*f.s_a,R=(f.r_h||0)+(f.r_a||0);
  const th=MD.tiers[home],ta=MD.tiers[away];const x=[1,P,PQ,PS,R,(th==='P4'&&ta==='P4')?1:0,(th==='FCS'||ta==='FCS')?1:0];
  let t=x.reduce((acc,v,i)=>acc+v*tm.beta[i],0);if(tm.recal)t=tm.recal[0]+tm.recal[1]*t;const m=ph-pa;ph=(t+m)/2;pa=(t-m)/2;}
 // Service-academy correction, same as the weekly projections
 const ac=MD.academy||{};if(ac.n&&ac.teams){const ah=ac.teams.includes(home),aa=ac.teams.includes(away);
  if(ah||aa){const m=ac.margin*((ah?1:0)-(aa?1:0)),t=ac.total;ph+=(m+t)/2;pa+=(t-m)/2;}}
 return {ph,pa};}
const Phi=x=>{const t=1/(1+0.2316419*Math.abs(x)),d=0.3989423*Math.exp(-x*x/2),p=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274))));return x>0?1-p:p;};
// Win chance from a projected margin, using how far off past predictions have been (normal approximation)
const winP=m=>Phi(m/((window.MODEL||{}).margin_sd||16));
// Fair (no-vig) American odds for a win chance
const fairOdds=p=>{p=Math.min(Math.max(p,0.005),0.995);return p>=0.5?`−${Math.round(100*p/(1-p))}`:`+${Math.round(100*(1-p)/p)}`;};
const known=t=>!!((window.MODEL||{}).tiers&&t in window.MODEL.tiers);
const pc=p=>p>0.995?'>99%':p<0.005?'<1%':Math.round(p*100)+'%';
"""

RATINGS_BODY = NAV("t") + r"""
<script src="model.js?v=__VER__"></script>
<details class="help"><summary>How to read this page</summary>
<ul>
<li><b>Rating</b> is how many points better than an average FBS team each team would be on a neutral field, using the same ratings and formula as the game projections. The gap between two teams' ratings is roughly the neutral-field spread; the Matchup page gives the exact number.</li>
<li><b>Move</b> is the change since last week: places up or down in the ranking, then the rating change in points.</li>
<li><b>Record</b> is the record so far. <b>Expected wins</b> adds each remaining scheduled game's win chance to the wins so far, so 10.4 means the model expects about 10–11 wins (bowls and title games count once they're scheduled). The team panel also shows how hard the remaining schedule is.</li>
<li><b>Conferences</b> compares conferences by their teams' average rating. <b>Schedule left</b> is the average rating of the opponents still to play, adjusted for where the games are.</li>
<li><b>±</b> is the rating's 80% range: how far the true strength could plausibly be from the estimate. It shrinks as a team plays more games.</li>
<li><b>Offense / Defense</b> are opponent-adjusted expected points (EPA) per play with garbage time removed, compared with an average FBS team. The small grey number is the rank. On defense, lower is better.</li>
<li>Tap a column header to sort, and tap a team for its results, rating over the season, upcoming games with projections, and a link to compare it with any opponent.</li>
</ul></details>
<p class="def"><b>Rating</b> = points better (+) or worse (−) than an average FBS team on a neutral field. Higher is better.</p>
<div id="movers"></div>
<div class="bar" id="tfilters"></div>
<div id="confs" hidden></div>
<div class="tbl" id="ttbl"><table class="rt"><thead><tr><th>#</th><th>Team</th><th aria-sort="descending" data-s="power"><button class="sb">Rating</button></th><th>Move</th><th class="hide-sm">Record</th><th class="hide-sm" data-s="pw"><button class="sb">Expected wins</button></th><th class="hide-sm" data-s="sos"><button class="sb">Schedule left</button><span class="when">avg opponent rating</span></th><th class="hide-sm" data-s="off"><button class="sb">Offense</button><span class="when">EPA/play, higher is better</span></th><th class="hide-sm" data-s="deff"><button class="sb">Defense</button><span class="when">EPA/play allowed, lower is better</span></th><th class="hide-sm" data-s="pace"><button class="sb">Plays/game</button></th></tr></thead><tbody id="teams"></tbody></table></div>
"""

RATINGS_JS = MODEL_FN + r"""
document.getElementById('sub').textContent=`${D.season} season · ${ratedThrough()} · ${freshTxt()}`;
const f1=x=>x==null?'—':x.toFixed(1);const s1=x=>x==null?'—':(Math.abs(x)<0.05?'0.0':(x>0?'+':'−')+Math.abs(x).toFixed(1));
const T=(D.teams||[]).map((t,i)=>({...t,rank:i+1}));
const PMAX=Math.max(1,...T.map(t=>Math.abs(t.power||0)));
const HAVE_MODEL=!!(window.MODEL&&window.MODEL.conv);
// Change since last week: places moved, then the rating change in points
const mv=t=>{if(t.prev_rank==null)return '<span class="muted">—</span>';const d=t.prev_rank-t.rank,p=t.power-t.prev_power;
 const a=d>0?`<span class="up">▲${d}</span> `:d<0?`<span class="down">▼${-d}</span> `:'';
 return `<span class="nw">${a}<span class="when">${Math.abs(p)<0.05?'0.0':(p>0?'+':'−')+Math.abs(p).toFixed(1)}</span></span>`;};
const bar=v=>{const w=Math.abs(v||0)/PMAX*50;return `<span class="rbar"><span class="${v>=0?'pos':'neg'}" style="width:${w.toFixed(1)}%;${v>=0?'left:50%':'right:50%'}"></span></span>`;};
const games=t=>(D.schedule||[]).filter(r=>r[2]===t||r[3]===t);
// A scheduled game that hasn't been played. Old unplayed games (postponed or cancelled) don't count.
const NOW=new Date(D.generated).getTime();
const pending=r=>!r[7]&&new Date(r[1]).getTime()>NOW-3*864e5;
// Win chance for team t in a scheduled game, from the same formula as the Matchup page
function gameP(t,r){const [wk,st,h,a,neu]=r,opp=h===t?a:h;if(!known(opp)||!known(t))return null;
 const {ph,pa}=project(h,a,neu),m=h===t?ph-pa:pa-ph;return {p:winP(m),m};}
const PROJ={};
if(HAVE_MODEL)T.forEach(t=>{let w=0,l=0,ew=0,n=0;for(const r of games(t.team)){if(r[7]&&r[5]!=null){((r[2]===t.team)?r[5]>r[6]:r[6]>r[5])?w++:l++;}
  else if(pending(r)){const g=gameP(t.team,r);if(g){ew+=g.p;n++;}}}
 PROJ[t.team]={w,l,ew,n,W:w+ew,L:l+n-ew};t.pw=w+ew;});
const projTxt=t=>{const P=PROJ[t];return P?P.W.toFixed(1):'—';};
// Chance of each final win total: combine each remaining game's win chance (games treated as independent)
function winDist(t){const ps=[];for(const r of games(t).filter(pending)){const g=HAVE_MODEL?gameP(t,r):null;if(g)ps.push(g.p);}
 let d=[1];for(const p of ps){const n=new Array(d.length+1).fill(0);d.forEach((v,k)=>{n[k]+=v*(1-p);n[k+1]+=v*p;});d=n;}return d;}
// Remaining-schedule strength: average opponent rating, adjusted for where each game is played
const HFA=(()=>{const T0=T[0];if(!HAVE_MODEL||!T0)return 0;const a=project(T0.team,T[1].team,false),b=project(T0.team,T[1].team,true);return (a.ph-a.pa)-(b.ph-b.pa);})();
function sos(t){const P=[];for(const r of games(t.team).filter(pending)){const home=r[2]===t.team,opp=home?r[3]:r[2],o=(D.teams||[]).find(x=>x.team===opp);
  if(!o||o.power==null)continue;P.push(o.power+(r[4]?0:(home?-HFA:HFA)));}return P.length?P.reduce((x,y)=>x+y,0)/P.length:null;}
const SOS={};T.forEach(t=>SOS[t.team]=sos(t));
const sosRank=t=>{const v=SOS[t];if(v==null)return null;return 1+Object.values(SOS).filter(x=>x!=null&&x>v).length;};
// Rating over the season (end of each completed week)
function rhist(t){const h=(D.rating_hist||{})[t.team]||[];if(h.length<2)return '';
 const W=280,H=70,vals=h.map(x=>x[1]),lo=Math.min(...vals)-9,hi=Math.max(...vals)+9,X=i=>10+(W-20)*i/(h.length-1),Y=v=>6+(H-22)*(hi-v)/(hi-lo);
 const path=h.map((x,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(x[1]).toFixed(1)}`).join('');
 const dots=h.map((x,i)=>`<circle cx="${X(i).toFixed(1)}" cy="${Y(x[1]).toFixed(1)}" r="2.5" fill="var(--series)"><title>${x[0]?wkName(x[0]):'Preseason'}: ${s1(x[1])}</title></circle>`).join('');
 const M=window.MODEL||{},pm=x=>M.rating_tau2?1.2816*Math.sqrt(M.rating_tau2/(Math.max(0,x[0]-0)+(M.lam||3))):0;
 const band=M.rating_tau2?`<path d="${h.map((x,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(x[1]+pm(x)).toFixed(1)}`).join('')}${h.slice().reverse().map((x,j)=>`L${X(h.length-1-j).toFixed(1)},${Y(x[1]-pm(x)).toFixed(1)}`).join('')}Z" fill="var(--series)" opacity=".15"/>`:'';
 return `<h4 style="margin-top:12px">Rating over the season <span class="when" style="display:inline">shaded = 80% range</span></h4><svg class="rh" viewBox="0 0 ${W} ${H}" role="img" aria-label="Rating by week">${band}${lo<0&&hi>0?`<line x1="10" x2="${W-10}" y1="${Y(0).toFixed(1)}" y2="${Y(0).toFixed(1)}" stroke="var(--line)"/><text x="${W-10}" y="${(Y(0)-3).toFixed(1)}" text-anchor="end" font-size="10" fill="var(--faint)">average FBS team</text>`:''}<path d="${path}" fill="none" stroke="var(--series)" stroke-width="2"/>${dots}<text x="10" y="${H-3}" font-size="10" fill="var(--faint)">Preseason ${s1(h[0][1])}</text><text x="${W-10}" y="${H-3}" text-anchor="end" font-size="10" fill="var(--faint)">${wkName(h[h.length-1][0])} ${s1(h[h.length-1][1])}</text></svg>`;}
// Biggest movers since last week
(()=>{const mv2=T.filter(t=>t.prev_power!=null).map(t=>({t,d:t.power-t.prev_power})).sort((a,b)=>Math.abs(b.d)-Math.abs(a.d)).slice(0,6);
 if(!mv2.length)return;document.getElementById('movers').innerHTML=`<div class="movers"><span class="when" style="display:inline">Biggest movers since last week</span> ${mv2.map(x=>`<button class="chip mvr" data-t="${esc(x.t.team)}">${esc(x.t.team)} <span class="${x.d>0?'up':'down'}">${x.d>0?'+':'−'}${Math.abs(x.d).toFixed(1)}</span></button>`).join('')}</div>`;
 document.querySelectorAll('.mvr').forEach(b=>b.onclick=()=>{open=b.dataset.t;tq='';tc='all';tconf='';document.getElementById('tq').value='';document.getElementById('tconf').value='';drawT(open);
  const el=[...document.querySelectorAll('#teams tr.g')].find(x=>x.dataset.t===open);if(el)el.scrollIntoView({block:'center',behavior:RM?'auto':'smooth'});});})();
let tc='all',tq='',tconf='',open=null,sortK='power';const tf=document.getElementById('tfilters');
const confs=[...new Set(T.map(t=>t.conf).filter(Boolean))].sort();
tf.innerHTML=[['all','All FBS'],['P4','Power 4'],['G5','Group of 5']].map(([k,l])=>`<button class="chip${k==='all'?' on':''}" data-k="${k}">${l}</button>`).join('')+
 `<select id="tconf" class="chip" aria-label="Conference"><option value="">All conferences</option>${confs.map(c=>`<option>${esc(c)}</option>`).join('')}</select><input type="search" id="tq" placeholder="Find a team" aria-label="Find a team"><button class="chip" id="cview" aria-pressed="false">Conferences</button>`;
tf.querySelectorAll('[data-k]').forEach(b=>b.onclick=()=>{tc=b.dataset.k;tf.querySelectorAll('[data-k]').forEach(x=>x.classList.toggle('on',x===b));drawT();});
document.getElementById('tconf').onchange=e=>{tconf=e.target.value;drawT();};
// Conference comparison: average rating, the best and middle team, and the spread of teams
document.getElementById('cview').onclick=e=>{const on=e.currentTarget.getAttribute('aria-pressed')!=='true';e.currentTarget.setAttribute('aria-pressed',String(on));e.currentTarget.classList.toggle('on',on);
 const box=document.getElementById('confs');box.hidden=!on;if(!on)return;
 const rows=confs.map(c=>{const ts=T.filter(t=>t.conf===c&&t.power!=null).sort((a,b)=>b.power-a.power);if(!ts.length)return null;
  const avg=ts.reduce((x,t)=>x+t.power,0)/ts.length,med=ts[Math.floor((ts.length-1)/2)];return {c,n:ts.length,avg,top:ts[0],med,lo:ts[ts.length-1]};}).filter(Boolean).sort((a,b)=>b.avg-a.avg);
 const span=Math.max(1,...rows.map(r=>Math.max(Math.abs(r.top.power),Math.abs(r.lo.power))));
 const rng=r=>{const X=v=>50+50*v/span;return `<span class="crng"><span style="left:${X(r.lo.power).toFixed(1)}%;width:${(X(r.top.power)-X(r.lo.power)).toFixed(1)}%"></span><i style="left:${X(r.avg).toFixed(1)}%"></i></span>`;};
 box.innerHTML=`<div class="tbl" style="margin-bottom:12px"><table><thead><tr><th>Conference</th><th>Teams</th><th>Average rating</th><th class="hide-sm">Range <span class="when">worst to best · tick = average</span></th><th>Best team</th><th class="hide-sm">Middle team</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.c)}</td><td>${r.n}</td><td><b>${s1(r.avg)}</b></td><td class="hide-sm">${rng(r)}</td><td>${esc(r.top.team)} <span class="when" style="display:inline">${s1(r.top.power)}</span></td><td class="hide-sm">${esc(r.med.team)} <span class="when" style="display:inline">${s1(r.med.power)}</span></td></tr>`).join('')}</tbody></table></div>`;};
document.getElementById('tq').oninput=e=>{tq=e.target.value.toLowerCase();drawT();};
document.querySelectorAll('th[data-s] .sb').forEach(bn=>bn.onclick=()=>{sortK=bn.closest('th').dataset.s;drawT();});
const at=(r,t)=>r[4]?'vs':(r[2]===t?'vs':'@');
function teamCard(t){
 const gs=games(t.team);const done=gs.filter(r=>r[7]&&r[5]!=null),next=gs.filter(pending);
 const res=done.map(r=>{const home=r[2]===t.team,opp=home?r[3]:r[2],us=home?r[5]:r[6],them=home?r[6]:r[5];
  return `<div class="kv"><span>${wkName(r[0])} · ${at(r,t.team)} ${esc(opp)}</span><span class="${us>them?'up':'down'}">${us>them?'W':'L'} ${us}–${them}</span></div>`;}).join('')||'<div class="sub">No games yet.</div>';
 const up=next.slice(0,6).map(r=>{const home=r[2]===t.team,opp=home?r[3]:r[2],v=r[4]?0:(home?1:2),g=HAVE_MODEL?gameP(t.team,r):null;
  const txt=g?`${Math.abs(g.m)<0.25?'PK':(g.m>0?'−':'+')+Math.abs(g.m).toFixed(1)} · ${g.p>0.995?'>99':g.p<0.005?'<1':Math.round(g.p*100)}%`:'projection';
  return `<div class="kv"><span>${wkName(r[0])} · ${at(r,t.team)} ${esc(opp)}</span><a class="lnk nw" href="matchup.html#${encodeURIComponent(t.team)}|${encodeURIComponent(opp)}|${v}" title="Spread from ${esc(t.team)}'s side and win chance">${txt} →</a></div>`;}).join('')||'<div class="sub">No games left on the schedule.</div>';
 const P=PROJ[t.team];
 return `<div class="dgrid"><div><h4>Results</h4>${res}</div><div><h4>Next games <span class="when" style="display:inline">spread · win chance</span></h4>${up}
  ${P&&P.n?`<div class="kv" style="margin-top:8px;border-top:1px solid var(--line);padding-top:6px"><span>Expected final record</span><b>${P.W.toFixed(1)}–${P.L.toFixed(1)}</b></div>`:''}
  ${(()=>{const P=PROJ[t.team];if(!P||!P.n)return '';const d=winDist(t.team),w0=P.w,l0=P.l,best=d.indexOf(Math.max(...d)),n=d.length-1;
    return `<div class="kv"><span>Chance to win out</span><span>${pct(d[n])}</span></div><div class="kv"><span>Most likely finish</span><span>${w0+best}–${l0+n-best} <span class="when" style="display:inline">${pct(d[best])}</span></span></div>`;})()}
  ${SOS[t.team]!=null?`<div class="kv"><span>Remaining schedule</span><span>opponents average ${s1(SOS[t.team])} <span class="when" style="display:inline">#${sosRank(t.team)} hardest of ${Object.values(SOS).filter(x=>x!=null).length}</span></span></div>`:''}</div>
  <div><h4>Profile</h4><div class="kv"><span>Score vs an average team</span><span>${t.pf==null?'—':`${f1(t.pf)}–${f1(t.pa)}`}</span></div>
  <div class="kv"><span>Offense rank</span><span>${t.off_rank} of ${T.length}</span></div><div class="kv"><span>Defense rank</span><span>${t.def_rank} of ${T.length}</span></div>
  <div class="kv"><span>Plays per game</span><span>${f1(t.pace)}</span></div>${t.prev_rank!=null?`<div class="kv"><span>Last week</span><span>#${t.prev_rank} · ${s1(t.prev_power)}</span></div>`:''}
  ${rhist(t)}
  <div style="margin-top:10px"><a class="lnk" href="matchup.html#${encodeURIComponent(t.team)}||0">Compare ${esc(t.team)} with any team →</a></div></div></div>`;}
const SORTS={sos:(a,b)=>(SOS[b.team]??-99)-(SOS[a.team]??-99),power:(a,b)=>a.rank-b.rank,pw:(a,b)=>(b.pw??-1)-(a.pw??-1),off:(a,b)=>b.off-a.off,deff:(a,b)=>a.deff-b.deff,pace:(a,b)=>b.pace-a.pace};
function drawT(focus){const rows=T.filter(t=>(tc==='all'||t.tier===tc)&&(!tconf||t.conf===tconf)&&(!tq||t.team.toLowerCase().includes(tq))).sort(SORTS[sortK]||SORTS.power);
 document.querySelectorAll('th[data-s]').forEach(th=>{const on=th.dataset.s===sortK;th.classList.toggle('on',on);if(on)th.setAttribute('aria-sort',sortK==='deff'?'ascending':'descending');else th.removeAttribute('aria-sort');});
 document.getElementById('teams').innerHTML=rows.map(t=>`<tr class="g" data-t="${esc(t.team)}" tabindex="0" aria-expanded="${open===t.team}"><td>${t.rank}</td><td><b>${esc(t.team)}</b> <span class="when">${esc(t.conf||t.tier)}</span><span class="show-sm when">${teamRecord(t.team)}${PROJ[t.team]?` · ${projTxt(t.team)} expected wins`:''}</span></td><td class="nw"><b>${s1(t.power)}</b>${ratingPM(t.team)?` <span class="when" style="display:inline" title="80% range of the rating">±${ratingPM(t.team).toFixed(1)}</span>`:''} ${bar(t.power)}</td><td>${mv(t)}</td><td class="hide-sm">${teamRecord(t.team)}</td><td class="hide-sm">${projTxt(t.team)}</td><td class="hide-sm">${SOS[t.team]!=null?s1(SOS[t.team]):'<span class="muted">—</span>'}</td><td class="hide-sm">${epa(t.off)} <span class="when">${t.off_rank}</span></td><td class="hide-sm">${epa(t.deff)} <span class="when">${t.def_rank}</span></td><td class="hide-sm">${f1(t.pace)}</td></tr>
 <tr class="detail"${open===t.team?'':' hidden'}><td colspan="10">${open===t.team?teamCard(t):''}</td></tr>`).join('')||'<tr><td colspan="10" class="muted">No teams match.</td></tr>';
 document.querySelectorAll('#teams tr.g').forEach(tr=>{const go=()=>{open=open===tr.dataset.t?null:tr.dataset.t;drawT(tr.dataset.t);};tr.onclick=go;tr.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go();}};});
 if(focus){const el=[...document.querySelectorAll('#teams tr.g')].find(x=>x.dataset.t===focus);if(el)el.focus({preventScroll:true});}}
drawT();
"""

MATCHUP_BODY = NAV("m") + r"""
<script src="model.js?v=__VER__"></script>
<details class="help"><summary>How to read this page</summary>
<ul>
<li>Pick any two teams and where they'd play. The projection uses the same ratings and formula as the weekly predictions, as of the latest update. Unlike the weekly projections, it doesn't include QB-change or manual adjustments, and it isn't a betting pick.</li>
<li><b>Spread</b> is from the favorite's side (−12.5 means favored by 12.5). <b>Win chance</b> and the <b>80% range</b> come from how far off past predictions have been (a typical miss of about <span id="msd">16</span> points on the margin), so even a 10-point favorite loses about a quarter of the time.</li>
<li><b>Fair odds</b> are the moneyline that matches the win chance with no sportsbook margin. A book's price longer than this on the same side would be value if the model is right.</li>
<li>The <b>what-if</b> slider shifts Team 1's strength by a number of points; the presets use typical values (a backup QB is worth about 7 points, a key non-QB starter about 3). It's a scenario, not a projection, and the card shows the unadjusted number next to it.</li>
<li><b>±</b> next to a rating is its 80% range: how far the true strength could plausibly be from the estimate, wider early in the season and for teams with few games.</li>
<li>Win chance and the range use the same method as the weekly projections: a normal curve around the projected margin with the size of past out-of-sample misses (it was better calibrated than the score simulation). The weekly numbers can still differ slightly when a manual adjustment (overrides.csv) applies; QB changes are flagged, not adjusted.</li>
<li>Share a matchup by copying the page link: it remembers the teams and the venue.</li>
</ul></details>
<div class="mpick">
 <label>Team 1<input list="tlist" id="t1" placeholder="Type a team" autocomplete="off"></label>
 <button class="chip" id="swap" title="Swap teams" aria-label="Swap teams">⇄</button>
 <label>Team 2<input list="tlist" id="t2" placeholder="Type a team" autocomplete="off"></label>
 <div class="venue" id="venue" role="group" aria-label="Venue"><button class="chip" data-v="1">Team 1 at home</button><button class="chip on" data-v="0">Neutral site</button><button class="chip" data-v="2">Team 2 at home</button></div>
 <datalist id="tlist"></datalist>
</div>
<div class="scen"><label for="adj">What-if: Team 1 is</label><input type="range" id="adj" min="-14" max="14" step="0.5" value="0" aria-describedby="adjv"><span id="adjv" class="nw">0 pts</span><span class="when" style="display:inline">better (for an injury, QB change, etc.)</span><button class="chip" id="adjr" hidden>Reset</button>
<span class="presets"><button class="chip" data-adj="-7" title="Team 1 loses its starting QB">Team 1 backup QB</button><button class="chip" data-adj="7" title="Team 2 loses its starting QB">Team 2 backup QB</button><button class="chip" data-adj="-3" title="Team 1 loses a key non-QB starter">Team 1 key player out</button></span></div>
<div id="mout"></div>
"""

MATCHUP_JS = MODEL_FN + r"""
const MD=window.MODEL||{};
document.getElementById('sub').textContent=`${D.season} season · ${ratedThrough()} · ${freshTxt()}`;
if(MD.margin_sd)document.getElementById('msd').textContent=Math.round(MD.margin_sd);
const TEAMS=(D.teams||[]).map(t=>t.team);const FCS=Object.keys(MD.tiers||{}).filter(t=>MD.tiers[t]==='FCS').sort();
const ALL=[...TEAMS.slice().sort(),...FCS];
document.getElementById('tlist').innerHTML=ALL.map(t=>`<option value="${esc(t)}">`).join('');
const TI=Object.fromEntries((D.teams||[]).map((t,i)=>[t.team,{...t,rank:i+1}]));
// Forgiving team lookup: exact, then case-insensitive, then a unique "starts with" or "contains" match
// ("alab" finds Alabama rather than Alabama State: when every match starts with the shortest one, that one wins)
const cands=x=>{const l=(x||'').trim().toLowerCase();if(!l)return [];
 let m=ALL.filter(t=>t.toLowerCase()===l);if(m.length)return m;
 m=ALL.filter(t=>t.toLowerCase().startsWith(l));if(!m.length)m=ALL.filter(t=>t.toLowerCase().includes(l));
 if(m.length>1){const sh=m.reduce((a,b)=>a.length<=b.length?a:b);if(m.every(t=>t.startsWith(sh)))return [sh];}
 return m;};
function resolve(x){x=(x||'').trim();if(!x)return '';if(known(x))return x;const m=cands(x);return m.length===1?m[0]:null;}
let venue=0,adj=0;
const vb=document.getElementById('venue'),i1=document.getElementById('t1'),i2=document.getElementById('t2');
const adjEl=document.getElementById('adj'),adjR=document.getElementById('adjr');
const setAdj=v=>{adj=v;adjEl.value=v;document.getElementById('adjv').textContent=`${v>0?'+':v<0?'−':''}${Math.abs(v)} pts`;adjR.hidden=!v;};
adjEl.oninput=()=>{setAdj(+adjEl.value);go();};adjR.onclick=()=>{setAdj(0);go();};
document.querySelectorAll('[data-adj]').forEach(b=>b.onclick=()=>{setAdj(+b.dataset.adj);go();});
const setVenue=v=>{venue=v;vb.querySelectorAll('[data-v]').forEach(x=>x.classList.toggle('on',+x.dataset.v===venue));};
vb.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{setVenue(+b.dataset.v);go();});
[i1,i2].forEach(el=>{el.oninput=go;el.onchange=()=>{const r=resolve(el.value);if(r)el.value=r;go();};});
document.getElementById('swap').onclick=()=>{[i1.value,i2.value]=[i2.value,i1.value];if(venue)setVenue(3-venue);go();};
// Keep focus in the text box when a suggestion is tapped (a blur would redraw the suggestions mid-click)
document.getElementById('mout').addEventListener('mousedown',e=>{if(e.target.closest('.chip'))e.preventDefault();});
const rk=t=>TI[t]?`#${TI[t].rank}`:(MD.tiers&&MD.tiers[t]==='FCS'?'FCS':'');
const s1v=x=>x==null?'—':(x>0?'+':x<0?'−':'')+Math.abs(x).toFixed(1);
const vlink=(a,b,v)=>`#${encodeURIComponent(a)}|${encodeURIComponent(b)}|${v}`;
// Before two teams are picked: one-tap examples from this week's slate
function emptyState(msg){
 const wk=(D.games||[]).filter(g=>!g.completed&&TI[g.home]&&TI[g.away]).sort((x,y)=>Math.max(TI[x.home].rank,TI[x.away].rank)-Math.max(TI[y.home].rank,TI[y.away].rank)).slice(0,6);
 const chips=wk.map(g=>`<a class="chip" href="${vlink(g.away,g.home,g.neutral?0:2)}">${esc(g.away)} ${g.neutral?'vs':'@'} ${esc(g.home)}</a>`).join('');
 const top=TEAMS.length>1?`<a class="chip" href="${vlink(TEAMS[0],TEAMS[1],0)}">#1 ${esc(TEAMS[0])} vs #2 ${esc(TEAMS[1])}, neutral</a>`:'';
 return `<div class="mcard"><div class="sub">${msg}</div>${chips||top?`<div class="mtry"><div class="when">This week's biggest games</div><div class="bar" style="margin:6px 0 0">${chips}${top}</div></div>`:''}</div>`;}
function go(){
 const ra=resolve(i1.value),rb=resolve(i2.value),out=document.getElementById('mout');
 if(!i1.value.trim()||!i2.value.trim()){out.innerHTML=emptyState('Pick two teams above, or try one of these:');return;}
 if(ra===null||rb===null){const el=ra===null?i1:i2,bad=el.value.trim(),c=cands(bad).slice(0,8);
  out.innerHTML=`<div class="mcard"><div class="sub">${c.length?`“${esc(bad)}” matches more than one team. Did you mean:`:`No team matches “${esc(bad)}”. Pick a name from the list.`}</div>${c.length?`<div class="bar" style="margin:8px 0 0">${c.map(t=>`<button class="chip" data-fill="${esc(t)}">${esc(t)}</button>`).join('')}</div>`:''}</div>`;
  out.querySelectorAll('[data-fill]').forEach(bn=>bn.onclick=()=>{el.value=bn.dataset.fill;go();});return;}
 const a=ra,b=rb;
 if(a===b){out.innerHTML=emptyState('Pick two different teams.');return;}
 // home/away from the venue choice; neutral keeps Team 2 as the nominal home side
 const home=venue===1?a:b,away=venue===1?b:a,neutral=venue===0;
 const {ph,pa}=project(home,away,neutral),m=ph-pa,T=ph+pa,sd=MD.margin_sd||16;
 const ma0=venue===1?m:-m; // margin from Team 1's side
 const ma=ma0+adj;          // plus the what-if adjustment
 const fav=ma>=0?a:b,spr=Math.abs(ma);
 const p1=winP(ma),pf=Math.max(p1,1-p1);
 const lo=ma-1.2816*sd,hi=ma+1.2816*sd,side=v=>Math.abs(v)<0.5?'a tie':(v>0?`${esc(a)} by ${Math.round(v)}`:`${esc(b)} by ${Math.round(-v)}`);
 const s1=venue===1?ph:pa,s2=venue===1?pa:ph;
 let S1=Math.round(s1+adj/2),S2=Math.round(s2-adj/2);if(S1===S2&&spr>=0.25){if(ma>0)S1++;else S2++;} // the shown score never contradicts the spread
 const live=(D.games||[]).find(g=>!g.completed&&((g.home===a&&g.away===b)||(g.home===b&&g.away===a)));
 const pw=t=>TI[t]&&TI[t].power!=null?TI[t].power:null;
 const row=(l,va,vb,better)=>`<tr><td>${l}</td><td class="${better===1?'hot':''}">${va}</td><td class="${better===2?'hot':''}">${vb}</td></tr>`;
 const cmp=(x,y,hiGood=true)=>x==null||y==null?0:(hiGood?(x>y?1:x<y?2:0):(x<y?1:x>y?2:0));
 const A=TI[a]||{},B=TI[b]||{};
 const rnk=r=>r==null?'':` <span class="when" style="display:inline">#${r}</span>`;
 let liveNote='';
 if(live){const lv=live.neutral?0:(live.home===a?1:2),mk=live.market||{};
  liveNote=`<div class="mnote">They play this week: <b>${esc(live.away)} ${live.neutral?'vs':'@'} ${esc(live.home)}</b>, ${when(live.start)}. Line ${line(live.home,live.away,mk.spread)}${mk.total!=null?`, total ${mk.total}`:''} · weekly model ${line(live.home,live.away,live.model_spread)}${live.model_total!=null?`, total ${fmt(live.model_total)}`:''}.`+
  (lv!==venue?` <a class="lnk" href="${vlink(a,b,lv)}">Use the real venue →</a>`:'')+` <a class="lnk" href="index.html#g${live.game_id}">See the full projection →</a></div>`;}
 const nz=project(home,away,true),mn=venue===1?(nz.ph-nz.pa):-(nz.ph-nz.pa),hf=ma0-mn;
 const by=v=>Math.abs(v)<0.25?'even':`${esc(v>=0?a:b)} by ${Math.abs(v).toFixed(1)}`;
 // The same matchup at each venue (what-if included)
 const at=[[1,`At ${esc(a)}`],[0,'Neutral site'],[2,`At ${esc(b)}`]].map(([v,l])=>{const pr=project(v===1?a:b,v===1?b:a,v===0),mm=(v===1?pr.ph-pr.pa:-(pr.ph-pr.pa))+adj,pp=winP(mm);
  return `<tr class="${v===venue?'on':''}"><td>${l}</td><td>${by(mm)}</td><td>${esc(pp>=0.5?a:b)} ${pc(Math.max(pp,1-pp))}</td><td>${(pr.ph+pr.pa).toFixed(1)}</td></tr>`;}).join('');
 // When they really play this week (at this venue): this page's ratings-only number next to the weekly projection and the line
 let cmpLive='';
 if(live&&(live.neutral?0:(live.home===a?1:2))===venue&&!adj){const mk=live.market||{},hw=live.p_home_win;
  cmpLive=`<div class="tbl" style="margin-top:10px"><table class="mtbl"><thead><tr><th></th><th>Spread</th><th>Total</th><th>${esc(live.home)} wins</th></tr></thead><tbody>
   <tr><td>This page (ratings only)</td><td>${line(live.home,live.away,Math.round(-(live.home===home?m:-m)*10)/10)}</td><td>${T.toFixed(1)}</td><td>${pc(live.home===a?p1:1-p1)}</td></tr>
   <tr><td>Weekly projection <span class="when" style="display:inline">includes any manual adjustment</span></td><td>${line(live.home,live.away,live.model_spread)}</td><td>${fmt(live.model_total)}</td><td>${hw!=null?pc(hw):'—'}</td></tr>
   <tr><td>Sportsbooks <span class="when" style="display:inline">middle of the books</span></td><td>${line(live.home,live.away,mk.spread)}</td><td>${mk.total??'—'}</td><td>—</td></tr></tbody></table></div>`;}
 out.innerHTML=`<div class="mcard">
  <div class="mhead"><div><div class="mteam">${esc(a)} <span class="when" style="display:inline">${rk(a)} · ${teamRecord(a)}</span></div><div class="mscore">${S1}</div></div>
  <div class="mvs"><div class="mat">${venue===2?'@':'vs'}</div><div class="when">${neutral?'neutral site':`at ${esc(home)}`}</div></div>
  <div style="text-align:right"><div class="mteam">${esc(b)} <span class="when" style="display:inline">${rk(b)} · ${teamRecord(b)}</span></div><div class="mscore">${S2}</div></div></div>
  <div class="cards mcards">
   <div class="card"><div class="l">Spread</div><div class="v">${spr<0.25?'Pick’em':esc(fav)+' −'+spr.toFixed(1)}</div></div>
   <div class="card"><div class="l">Total</div><div class="v">${T.toFixed(1)}</div></div>
   <div class="card"><div class="l">Win chance</div><div class="v">${pc(pf)} <span class="when" style="display:inline">${esc(p1>=0.5?a:b)}</span></div>
    <div class="wbar" aria-hidden="true"><span class="${p1>=0.5?'fav':''}" style="width:${(p1*100).toFixed(1)}%"></span><span class="${p1<0.5?'fav':''}" style="width:${((1-p1)*100).toFixed(1)}%"></span></div>
    <div class="kv when"><span>${esc(a)} ${pc(p1)}</span><span>${esc(b)} ${pc(1-p1)}</span></div></div>
   <div class="card" title="The moneyline that exactly matches the win chance, with no sportsbook margin. A book paying more than this on the same side is value if the model is right."><div class="l">Fair odds <span class="when" style="display:inline">no vig</span></div><div class="v" style="font-size:16px">${esc(fav)} ${fairOdds(pf)}</div><div class="when">${esc(fav===a?b:a)} ${fairOdds(1-pf)}</div></div>
   <div class="card"><div class="l">80% range</div><div class="v" style="font-size:15px">${ma>=0?side(hi):side(lo)}<br><span class="sub">to</span> ${ma>=0?side(lo):side(hi)}</div></div></div>
  ${adj?`<div class="mnote"><span class="pill p-warn">What-if</span> ${esc(a)} adjusted by ${adj>0?'+':'−'}${Math.abs(adj)} points. This is a scenario, not the model's projection. Without it: <b>${by(ma0)}</b>, ${esc(winP(ma0)>=0.5?a:b)} ${pc(Math.max(winP(ma0),1-winP(ma0)))} to win.</div>`:''}
  ${liveNote}${cmpLive}
  <h4 class="mh">Why</h4>
  <div class="eq"><span><b>${by(mn)}</b><span class="when">on a neutral field</span></span>${neutral?'':`<span class="op">+</span><span><b>${Math.abs(hf).toFixed(1)} for ${esc(hf>=0?a:b)}</b><span class="when">home field</span></span>`}${adj?`<span class="op">+</span><span><b>${Math.abs(adj).toFixed(1)} for ${esc(adj>0?a:b)}</b><span class="when">what-if</span></span>`:''}<span class="op">=</span><span><b>${by(ma)}</b><span class="when">projected margin</span></span></div>
  <div class="sub">Power ratings ${s1v(pw(a))} vs ${s1v(pw(b))} compare each team with an average opponent; the matchup number also accounts for how these two offenses and defenses meet, and their pace.</div>
  <h4 class="mh">Every venue</h4>
  <div class="tbl"><table class="mtbl vtbl"><thead><tr><th>Venue</th><th>Margin</th><th>Win chance</th><th>Total</th></tr></thead><tbody>${at}</tbody></table></div>
  <h4 class="mh">Side by side</h4>
  <div class="tbl" style="margin-top:10px"><table class="mtbl"><thead><tr><th></th><th>${esc(a)}</th><th>${esc(b)}</th></tr></thead><tbody>
   ${row('Power rating',`${s1v(pw(a))}${ratingPM(a)?` <span class="when" style="display:inline">±${ratingPM(a).toFixed(1)}</span>`:''}${A.rank?rnk(A.rank):''}`,`${s1v(pw(b))}${ratingPM(b)?` <span class="when" style="display:inline">±${ratingPM(b).toFixed(1)}</span>`:''}${B.rank?rnk(B.rank):''}`,cmp(pw(a),pw(b)))}
   ${row('Games played',gamesPlayed(a),gamesPlayed(b),0)}
   ${row('Score vs an average team',A.pf!=null?`${A.pf.toFixed(0)}–${A.pa.toFixed(0)}`:'—',B.pf!=null?`${B.pf.toFixed(0)}–${B.pa.toFixed(0)}`:'—',0)}
   ${row('Offense <span class="when" style="display:inline">EPA/play</span>',`${epa(A.off)}${rnk(A.off_rank)}`,`${epa(B.off)}${rnk(B.off_rank)}`,cmp(A.off,B.off))}
   ${row('Defense <span class="when" style="display:inline">EPA/play allowed</span>',`${epa(A.deff)}${rnk(A.def_rank)}`,`${epa(B.deff)}${rnk(B.def_rank)}`,cmp(A.deff,B.deff,false))}
   ${row('Plays per game',A.pace!=null?A.pace.toFixed(1):'—',B.pace!=null?B.pace.toFixed(1):'—',0)}
   ${row('Record',teamRecord(a),teamRecord(b),0)}
  </tbody></table></div>
  <div style="margin-top:10px"><button class="chip" id="copym">Copy this projection</button> <span class="when" id="copied" style="display:inline"></span></div>
  <div class="when" style="margin-top:8px">Highlighted = better. EPA is opponent-adjusted expected points per play compared with an average FBS team (0 = average), so negative is good on defense. Ranks are among ${TEAMS.length} FBS teams; FCS teams are rated as a group plus their own results.</div></div>`;
 const cb=document.getElementById('copym');if(cb)cb.onclick=()=>{const txt=`${a} ${venue===2?'@':'vs'} ${b}${neutral?' (neutral site)':` (at ${home})`}: ${a} ${S1}, ${b} ${S2}. Spread ${spr<0.25?'pick':fav+' -'+spr.toFixed(1)}, total ${T.toFixed(1)}, ${p1>=0.5?a:b} ${pc(pf)} to win (fair odds ${fairOdds(pf)}).${adj?` What-if: ${a} ${adj>0?'+':''}${adj} pts.`:''} Ratings through ${ratedThrough().replace('ratings through ','')}, ${D.season}. College football model.`;
  try{navigator.clipboard.writeText(txt).then(()=>{document.getElementById('copied').textContent='Copied';});}catch(e){document.getElementById('copied').textContent=txt;}};
 try{history.replaceState(null,'',vlink(a,b,venue));}catch(e){}}
// Prefill from the link (#Team1|Team2|venue), e.g. from the Team ratings page
function fromHash(){const h=decodeURIComponent(location.hash.slice(1));if(!h)return;const [a,b,v]=h.split('|');
 i1.value=a||'';i2.value=b||'';if(v!=null&&v!=='')setVenue(+v);}
fromHash();window.addEventListener('hashchange',()=>{fromHash();go();});
go();
"""

INDEX_BODY = INDEX_BODY.replace("<script>window.PAGE='index'</script>", "")


def _page(body, js, title, ver="0"):
    return PAGE.replace("__VER__", ver).replace("__BODY__", body).replace("__SCRIPT__", js).replace("__TITLE__", title)
