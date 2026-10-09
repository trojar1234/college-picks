"""Builds the static website in docs/ (served by GitHub Pages)."""
import json

import numpy as np
import pandas as pd

import cfb_config as C


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
        pr = M.predict_bt(pd.DataFrame(feats), state["conv"], state.get("groups") or M.BASE_GROUPS)
        df["power"] = (pr.pred_h - pr.pred_a).to_numpy()
        df["pf"] = pr.pred_h.to_numpy()
        df["pa"] = pr.pred_a.to_numpy()
        df["power"] -= df["power"].mean()  # 0 = average FBS team
        df = df.sort_values("power", ascending=False)
    else:
        df = df.sort_values("net", ascending=False)
    df["off_rank"] = df.off.rank(ascending=False).astype(int)
    df["def_rank"] = df.deff.rank(ascending=True).astype(int)
    return df.to_dict("records")


def build(season, wk, slate, store, rep, final, now, unmatched, state=None):
    games = []
    for r in slate.sort_values("start").itertuples():
        p = store.get(str(r.game_id))
        g = dict(game_id=int(r.game_id), start=r.start.isoformat(), home=r.home, away=r.away,
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
    history = sorted(
        [p for p in store.values() if p.get("result")],
        key=lambda p: p["start"], reverse=True,
    )[:400]
    payload = _clean(dict(
        generated=now.isoformat(), season=season, week=wk, games=games, report=rep,
        teams=team_table(final, state), history=history, unmatched=unmatched,
        thresholds=dict(spread=C.SPREAD_EDGE, total=C.TOTAL_EDGE),
    ))
    C.SITE.mkdir(exist_ok=True)
    (C.SITE / "data.js").write_text("window.DATA = " + json.dumps(payload, separators=(",", ":")) + ";")
    # a new version tag each run so browsers never pair a fresh page with stale cached data
    ver = now.strftime("%Y%m%d%H%M%S")
    (C.SITE / "index.html").write_text(_page(INDEX_BODY, INDEX_JS, "Projections", ver))
    (C.SITE / "report.html").write_text(_page(REPORT_BODY, REPORT_JS, "Report card", ver))
    (C.SITE / "ratings.html").write_text(_page(RATINGS_BODY, RATINGS_JS, "Team ratings", ver))
    (C.SITE / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    (C.SITE / ".nojekyll").write_text("")


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<meta http-equiv="Cache-Control" content="no-cache">
<title>__TITLE__ · CFB model</title>
<style>
:root{--bg:#f7f6f2;--card:#fff;--soft:#efede6;--text:#1d1c1a;--muted:#6b6a64;--faint:#9a988f;--line:#e2dfd6;
--acc:#1f5fa8;--accbg:#e4eefa;--good:#2f6a12;--goodbg:#e6f1da;--warn:#7d4a07;--warnbg:#f8ecd6;--bad:#9b2c2c;--badbg:#f8e3e3;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1d1d1b;--soft:#262623;--text:#ecebe6;--muted:#a3a29b;--faint:#77756e;--line:#33322e;
--acc:#8db8ec;--accbg:#1b2c42;--good:#a5d27a;--goodbg:#22311a;--warn:#f0c174;--warnbg:#3a2c12;--bad:#f09b9b;--badbg:#3d1d1d}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:20px 16px 60px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:18px}
h1{font-size:22px;margin:0;font-weight:600;letter-spacing:-.01em}
h2{font-size:17px;margin:28px 0 10px;font-weight:600}
nav a{color:var(--muted);text-decoration:none;margin-left:16px;font-size:14px}
nav a.on{color:var(--text);font-weight:600}
.sub{color:var(--muted);font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px}
.card{background:var(--soft);border-radius:10px;padding:12px 14px}
.card .l{font-size:12px;color:var(--muted)}.card{min-width:0}.card .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.card .v .push{font-size:14px;color:var(--muted);font-weight:500}.card .v .nw{white-space:nowrap}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12px;color:var(--muted);margin:0 0 10px;align-items:center}
.summary{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin-bottom:14px}
.summary h3{font-size:13px;font-weight:600;margin:0 0 8px}.summary .grp+.grp{margin-top:12px}
.sitem{display:grid;grid-template-columns:auto 1fr auto;gap:4px 10px;align-items:baseline;padding:6px 0;border-top:1px solid var(--line);cursor:pointer;font-size:14px}
.sitem:first-of-type{border-top:0}.sitem .g2{color:var(--muted);font-size:12px}.sitem .r{font-size:12px;color:var(--faint);text-align:right}
.sitem:hover .t{text-decoration:underline}
tr.day td{background:var(--soft);font-size:12px;font-weight:600;color:var(--muted);padding:7px 12px;letter-spacing:.02em}
.pick .t.weak{color:var(--muted)}
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
tr.g{cursor:pointer}tr.g:hover td{background:var(--soft)}
tr:last-child td{border-bottom:0}
.home{font-weight:600}.when{font-size:12px;color:var(--faint)}
.pill{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;white-space:nowrap;margin:1px 0}
.p-good{background:var(--goodbg);color:var(--good)}.p-warn{background:var(--warnbg);color:var(--warn)}
.p-bad{background:var(--badbg);color:var(--bad)}.p-acc{background:var(--accbg);color:var(--acc)}
.muted{color:var(--faint)}.up{color:var(--good)}.down{color:var(--bad)}
.detail td{background:var(--soft);padding:14px 16px}
.dgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:18px}
.dgrid h4{margin:0 0 6px;font-size:12px;color:var(--muted);font-weight:500}
.kv{display:flex;justify-content:space-between;font-size:13px;padding:2px 0}
.hist{display:flex;align-items:flex-end;gap:2px;height:60px}
.hist div{flex:1;background:var(--acc);opacity:.75;border-radius:2px 2px 0 0;min-height:1px}
.hist div.z{opacity:.35}
.note{font-size:12px;color:var(--muted);margin-top:10px}
.flag{color:var(--bad);font-weight:600}
@media (max-width:720px){.hide-sm{display:none}td,th{padding:8px 6px;font-size:13px}.card .v{font-size:19px}}
.show-sm{display:none}@media (max-width:720px){.show-sm{display:block}}
.nw{white-space:nowrap}
@media (max-width:720px){table.games thead{display:none}table.games tr.g{display:grid;grid-template-columns:1fr auto;border-bottom:1px solid var(--line)}
table.games tr.g td{border:0;padding:10px 12px 2px}table.games td.c-proj{text-align:right}table.games td.c-picks{grid-column:1/-1;padding:2px 12px 12px}
#cards{grid-template-columns:repeat(3,1fr);gap:8px}#cards .card{padding:8px 10px}#cards .card .v{font-size:17px}#cards .card .l{font-size:11px}table.games tr.g td.hide-sm{display:none}table.games tr.detail td{display:block}table.games tr.day td{display:block}.sitem{grid-template-columns:auto 1fr}.sitem .r{grid-column:1/-1;text-align:left}}
</style></head><body><div class="wrap">
__BODY__
<p class="note" id="foot"></p>
</div>
<script src="data.js?v=__VER__"></script>
<script>
const D=window.DATA;
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
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('foot').innerHTML=`Updated ${new Date(D.generated).toLocaleString('en-US',{timeZone:'America/Chicago'})} CT · Data from CollegeFootballData.com, The Odds API, Open-Meteo.`+(D.unmatched&&D.unmatched.length?` · Unmatched odds games: ${D.unmatched.length}`:'');
</script>
<script>__SCRIPT__</script>
</body></html>"""

NAV = lambda on: f"""<header><div><h1>College football model</h1><div class="sub" id="sub"></div></div>
<nav><a href="index.html" class="{'on' if on=='p' else ''}">Projections</a><a href="ratings.html" class="{'on' if on=='t' else ''}">Team ratings</a><a href="report.html" class="{'on' if on=='r' else ''}">Report card</a></nav></header>"""

INDEX_BODY = NAV("p") + r"""
<div class="cards" id="cards"></div>
<div id="summary"></div>
<div class="bar" id="filters"></div>
<div class="legend"><span><span class="pill p-good">BET</span> model and line 7+ points apart, nothing shaky</span><span><span class="pill p-warn">FLAG</span> 7+ apart, but there's a reason for caution</span><span><b>edge</b> = points between the model and the line</span><span>Home team in <b>bold</b> · scores away–home · tap a game for details</span></div>
<div class="tbl"><table class="games"><thead><tr>
<th>Game</th><th>Projected score</th><th class="hide-sm">Spread <span class="when">model / line</span></th><th class="hide-sm">Total <span class="when">model / line</span></th><th class="hide-sm">Line move <span class="when">open → now</span></th><th>Model picks</th>
</tr></thead><tbody id="rows"></tbody></table></div>
<p class="note"><b>Model picks</b> shows the model's side in every game against the current line (the closing line once a game is final). Most picks are close calls; only <span class="pill p-good">BET</span> games clear the 7-point bar with nothing making the projection shaky. A <span class="pill p-warn">FLAG</span> clears the bar but comes with a caution: a QB change, wind or rain (the model doesn't use weather), or a gap of 8+ points ("check news"), where in past seasons the market was right about two times in three. A bet keeps the line it was first flagged at. The model's edge is against early-week lines, so a bet is worth acting on early and only if the line hasn't already moved to the model's number. Lines are never used as model inputs.</p>
""".replace("</div>\n<p", "</div>\n<p", 1) + "<script>window.PAGE='index'</script>"

REPORT_BODY = NAV("r") + r"""
<div class="summary" id="glance"></div>
<h2>This season (live, graded at kickoff)</h2>
<div class="cards" id="live"></div>
<h2>Backtest by season</h2>
<p class="sub">Walk-forward: each week predicted using only data available before it. "Bets" are games the model flags (7+ point disagreement); they're graded against the opening line (when the model's edges are meant to be bet) and the closing line. "All games" grades the model's side in every game against the closing line. Opening lines exist for only part of the history. Small grey numbers after a record are pushes.</p>
<div class="cards" id="btcards"></div>
<div class="tbl"><table><thead><tr><th>Season</th><th>Games</th><th>Margin MAE</th><th class="hide-sm">Market margin MAE</th><th class="hide-sm">Total MAE</th><th>Bets ATS vs open</th><th>Bets ATS vs close</th><th>Bets O/U vs open</th><th class="hide-sm">Bets O/U vs close</th><th class="hide-sm">All games ATS vs close</th></tr></thead><tbody id="bt"></tbody></table></div>
<h2>Model improvements tested</h2>
<p class="sub" id="expsub"></p>
<div class="tbl"><table><thead><tr><th>Feature</th><th>Margin error change</th><th>Total error change</th><th>Result</th></tr></thead><tbody id="exp"></tbody></table></div>
<p class="sub" id="exphold"></p>
<p class="sub">Tested earlier and removed because they didn't improve accuracy: weather at kickoff, travel and rest, recency weighting, run/pass matchups, fitting the margin directly, a diminishing-returns margin rating, per-team home field, coaching changes, turnover-based preseason weighting, and star-rating-based recruiting/portal data.</p>
<h2>Calibration</h2>
<p class="sub">When the model gives the home team X% to cover, how often do they?</p>
<div class="tbl"><table><thead><tr><th>Model said</th><th>Average</th><th>Actually covered</th><th>Games</th></tr></thead><tbody id="cal"></tbody></table></div>
<p class="sub" id="shrink"></p>
<h2>Error diagnostics</h2>
<p class="sub">Bias = average of (predicted − actual). Red rows are biases unlikely to be chance (100+ games, more than 2.5 standard errors).</p>
<div id="seg"></div>
<h2>Recent results</h2>
<p class="sub">Scores are away–home. Every game shows the model's spread and total picks against the closing line; <span class="pill p-good">BET</span> marks flagged bets, graded at the line when they were first flagged.</p>
<div class="tbl"><table><thead><tr><th>Game</th><th>Final</th><th class="hide-sm">Projected</th><th>Spread pick</th><th>Total pick</th><th class="hide-sm">CLV</th></tr></thead><tbody id="hist"></tbody></table></div>
<h2>Model settings</h2><div class="sub" id="settings"></div>
<script>window.PAGE='report'</script>
"""

INDEX_JS = r"""
const G=D.games;
document.getElementById('sub').textContent=`${D.season} season · Week ${D.week??'—'} · updated ${new Date(D.generated).toLocaleString('en-US',{timeZone:'America/Chicago',weekday:'short',hour:'numeric',minute:'2-digit'})} CT`;
const L=D.report.live||{};
const isLowTotal=g=>(g.low_conf||[]).length||(g.total_caution||[]).length;
const bets=G.filter(g=>!g.started&&(g.spread_bet||g.total_bet));
const withPct=r=>!r||(r.w+r.l+r.p)===0?'—':`${rec(r)}<span class="when">${pct(r.pct)}</span>`;
document.getElementById('cards').innerHTML=[
['Games',G.length],['Bets this week',bets.length],['Bets ATS',withPct(L.ats)],['Bets O/U',withPct(L.ou)],
['All games ATS',withPct(L.ats_all)],['All games O/U',withPct(L.ou_all)]].map(([l,v])=>`<div class="card"><div class="l">${l}</div><div class="v">${v}</div></div>`).join('');
const F=[['all','All games'],['bets','Bets'],['edges','All flags'],['P4','Power 4'],['G5','Group of 5']];
let cur='all',hideLow=false,q='';
const fb=document.getElementById('filters');
fb.innerHTML=F.map(([k,l])=>`<button class="chip${k==='all'?' on':''}" data-k="${k}">${l}</button>`).join('')+
`<button class="chip" id="low">Hide low confidence</button><select id="sort" class="chip"><option value="time">Sort: kickoff</option><option value="edge">Sort: biggest edge</option></select><input type="search" id="q" placeholder="Find a team">`;
let sortBy='time';document.getElementById('sort').onchange=e=>{sortBy=e.target.value;draw();};
fb.querySelectorAll('[data-k]').forEach(b=>b.onclick=()=>{cur=b.dataset.k;fb.querySelectorAll('[data-k]').forEach(x=>x.classList.toggle('on',x===b));draw();});
document.getElementById('low').onclick=e=>{hideLow=!hideLow;e.target.classList.toggle('on',hideLow);draw();};
document.getElementById('q').oninput=e=>{q=e.target.value.toLowerCase();draw();};
const isBet=g=>!!(g.spread_bet||g.total_bet);
function keep(g){
 if(q&&!(g.home+' '+g.away).toLowerCase().includes(q))return false;
 if(hideLow&&(g.low_conf||[]).length)return false;
 if(cur==='bets')return g.completed?!!(g.result&&(g.result.ats||g.result.ou)):isBet(g);
 if(cur==='edges')return g.completed?!!(g.result&&(g.result.ats||g.result.ou)):!!(g.spread_flag||g.total_flag||g.spread_bet||g.total_bet);
 if(cur==='P4')return g.home_tier==='P4'||g.away_tier==='P4';
 if(cur==='G5')return g.home_tier==='G5'||g.away_tier==='G5';
 return true;}
function pickRow(kind,p,status,result){
 if(!p)return `<div class="pick"><span class="e">${kind}: no line</span></div>`;
 if(p.none)return `<div class="pick"><span class="e">${kind}: no pick (model on the line)</span></div>`;
 const txt=kind==='Spread'?`${esc(p.team)} ${sl(p.line)}`:`${p.side} ${p.line}`;
 const tag=status==='bet'?'<span class="pill p-good">BET</span>':status==='caution'?'<span class="pill p-warn">FLAG</span>':'';
 const cls=status==='bet'?'bet':(!status&&!result&&p.edge<2?'weak':'');
 return `<div class="pick">${tag}<span class="t ${cls}">${txt}</span><span class="e">edge ${p.edge.toFixed(1)}</span>${result||''}${p.now?`<span class="e">· line now ${p.now}</span>`:''}</div>`;}
// Plain-English reasons for a caution
const REASON=x=>x.startsWith('big disagreement')?'Model and line 8+ pts apart: check injury/QB news':x.startsWith('wind/rain')?"Wind/rain forecast: the model doesn't use weather":x==='QB change'?'QB change: see details':x==='manual adjustment'?'Manual adjustment':x==='FCS opponent'?'FCS opponent':x==='early season'?'Early season':x;
function picksCell(g){
 const m=g.market||{};
 if(g.model_spread==null)return '<span class="muted">No projection</span>';
 if(g.completed&&g.result){
  const r=g.result;
  const sp=spreadPick(g,r.close_spread,g.model_spread),tp=totalPick(r.close_total,g.model_total);
  const sBet=!!r.ats,tBet=!!r.ou;
  return pickRow('Spread',sp,sBet?'bet':'',resPill(sBet?r.ats:r.ats_all))+pickRow('Total',tp,tBet?'bet':'',resPill(tBet?r.ou:r.ou_all))+
   `<div class="when">vs closing line${sBet||tBet?' · BET graded at the line when first flagged':''}</div>`;}
 let sp=spreadPick(g,m.spread,g.model_spread),tp=totalPick(m.total,g.model_total);
 const low=(g.low_conf||[]).length;
 let sSt=g.spread_flag&&low?'caution':'',tSt=g.total_flag&&isLowTotal(g)?'caution':'';
 // A bet stays at the line where it was first flagged, even if the line has moved since
 const sb=g.spread_bet,tb=g.total_bet;
 if(sb){const home=sb.side==='home';sp={team:home?g.home:g.away,line:home?sb.line:-sb.line,edge:Math.abs(sb.line-g.model_spread),was:Math.abs(sb.line-sb.model)};
  if(m.spread!=null&&m.spread!==sb.line)sp.now=sl(home?m.spread:-m.spread);sSt='bet';}
 if(tb){tp={side:tb.side==='over'?'Over':'Under',line:tb.line,edge:Math.abs(g.model_total-tb.line),was:Math.abs(tb.model-tb.line)};
  if(m.total!=null&&m.total!==tb.line)tp.now=m.total;tSt='bet';}
 const why=[...((sSt||tSt)?(g.low_conf||[]):[]),...((tSt&&(g.total_caution||[]).length)?g.total_caution:[])];
 return pickRow('Spread',sp,sSt)+pickRow('Total',tp,tSt)+(why.length?`<div class="when">${esc([...new Set(why.map(REASON))].join(' · '))}</div>`:'')+
  `<div class="when show-sm">Model ${line(g.home,g.away,g.model_spread)}, ${fmt(g.model_total)} · Line ${line(g.home,g.away,m.spread)}, ${fmt(m.total)}</div>`;}
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
 if((g.low_conf||[]).some(x=>x.startsWith('big disagreement')))s+=' <span class="pill p-warn" title="Model and market differ a lot: check injury/QB news">check news</span>';return s;}
function detail(g){
 if(g.home_med==null)return '<div class="sub">No projection stored for this game.</div>';
 const h=g.hist||[],mx=Math.max(1,...h);const m=g.market||{};
 const bars=h.map((v,i)=>`<div class="${(i*3-45)<0?'z':''}" style="height:${Math.round(v/mx*100)}%" title="${i*3-45} to ${i*3-43}"></div>`).join('');
 const likely=(g.likely||[]).map(([a,b,p])=>`<div class="kv"><span>${esc(g.away)} ${b}, ${esc(g.home)} ${a}</span><span>${pct1(p)}</span></div>`).join('');
 const books=(m.books||[]).map(b=>`<div class="kv"><span>${esc(b.book)}</span><span>${line(g.home,g.away,b.spread)} · ${fmt(b.total)}</span></div>`).join('')||'<div class="sub">No sportsbook snapshot yet.</div>';
 const R=g.ratings||{};const wx=g.weather||{};
 const wtxt=wx.dome?'Dome':wx.wind!=null?`${Math.round(wx.temp)}°F · ${Math.round(wx.wind)} mph wind · ${wx.precip>=0.01?wx.precip.toFixed(2)+' in rain':'no rain'}`:null;
 const fs=g.spread_bet,ft=g.total_bet;
 return `<div class="dgrid">
 <div><h4>Probabilities</h4>
  <div class="kv"><span>${esc(g.away)} wins</span><span>${pct(1-g.p_home_win)}</span></div>
  <div class="kv"><span>${esc(g.home)} wins</span><span>${pct(g.p_home_win)}</span></div>
  ${g.p_side_cover!=null?`<div class="kv"><span>${esc(g.spread_side==='home'?g.home:g.away)} covers</span><span>${pct(g.p_side_cover)}</span></div>`:''}
  ${g.p_side_total!=null?`<div class="kv"><span>${g.total_side==='over'?'Over':'Under'} hits</span><span>${pct(g.p_side_total)}</span></div>`:''}
  <h4 style="margin-top:12px">Most likely final scores</h4>${likely}</div>
 <div><h4>Margin distribution (${esc(g.home)} − ${esc(g.away)})</h4><div class="hist">${bars}</div>
  <div class="kv sub"><span>−45</span><span>0</span><span>+45</span></div>
  <h4 style="margin-top:12px">Inputs</h4>
  <div class="kv"><span>Expected points (away · home)</span><span>${fmt(g.exp_away)} · ${fmt(g.exp_home)}</span></div>
  <div class="kv"><span>Off EPA/play (away · home)</span><span>${epa(R.away_off)} · ${epa(R.home_off)}</span></div>
  <div class="kv"><span>Def EPA/play allowed (away · home)</span><span>${epa(R.away_def)} · ${epa(R.home_def)}</span></div>
  <div class="kv"><span>Expected plays (away · home)</span><span>${fmt(R.away_pace)} · ${fmt(R.home_pace)}</span></div>
  ${wtxt?`<div class="kv"><span>Weather</span><span>${wtxt}</span></div>`:''}
  ${(g.notes||[]).length?`<div class="notes"><b>Notes:</b> ${esc(g.notes.join('; '))}</div>`:''}</div>
 <div><h4>Lines by book</h4>${books}
  <div class="kv" style="margin-top:8px"><span>Open</span><span>${line(g.home,g.away,m.open_spread)} · ${fmt(m.open_total)}</span></div>
  <div class="kv"><span>Now (median of books)</span><span>${line(g.home,g.away,m.spread)} · ${fmt(m.total)}</span></div>
  ${fs?`<div class="kv"><span>Spread bet flagged at</span><span>${esc(fs.side==='home'?g.home:g.away)} ${sl(fs.side==='home'?fs.line:-fs.line)} (edge ${Math.abs(fs.line-fs.model).toFixed(1)})</span></div>`:''}
  ${ft?`<div class="kv"><span>Total bet flagged at</span><span>${ft.side==='over'?'Over':'Under'} ${fmt(ft.line)} (edge ${Math.abs(ft.model-ft.line).toFixed(1)})</span></div>`:''}</div>
 </div>`;}
const day=s=>new Date(s).toLocaleDateString('en-US',{timeZone:'America/Chicago',weekday:'long',month:'short',day:'numeric'});
const winPct=p=>p>0.995?'>99%':p<0.005?'<1%':pct(p);
function rowHtml(g){
  const at=g.neutral?'vs':'@';
  const fav=g.p_home_win==null?'':(g.p_home_win>=0.5?`${winPct(g.p_home_win)} ${esc(g.home)} win`:`${winPct(1-g.p_home_win)} ${esc(g.away)} win`);
  const proj=g.completed?`<span>${g.away_pts}–${g.home_pts}</span><div class="when">final${projScore(g)?` · proj ${projScore(g)}`:''}</div>`:
   (!projScore(g)?'<span class="muted">—</span>':`${projScore(g)}<div class="when">${fav}</div>`);
  const m=g.market||{};
  return `<tr class="g" id="g${g.game_id}" data-id="${g.game_id}"><td class="c-game">${esc(g.away)} ${at} <span class="home">${esc(g.home)}</span>${icons(g)}<div class="when">${when(g.start)}${g.started&&!g.completed?' · in progress':''}${g.neutral?' · neutral site':''}</div></td>
  <td class="c-proj">${proj}</td><td class="hide-sm">${line(g.home,g.away,g.model_spread)}<div class="when">line ${line(g.home,g.away,m.spread)}</div></td>
  <td class="hide-sm">${fmt(g.model_total)}<div class="when">line ${fmt(m.total)}</div></td><td class="hide-sm">${moveCell(g)}</td><td class="c-picks">${picksCell(g)}</td></tr>
  <tr class="detail" id="d${g.game_id}" hidden><td colspan="6">${detail(g)}</td></tr>`;}
// Largest gap between the model and the line in a game (a bet keeps its flagged line)
function maxEdge(g){const m=g.market||{};const e=[];
 if(g.spread_bet)e.push(Math.abs(g.spread_bet.line-g.model_spread));else if(m.spread!=null&&g.model_spread!=null)e.push(Math.abs(m.spread-g.model_spread));
 if(g.total_bet)e.push(Math.abs(g.model_total-g.total_bet.line));else if(m.total!=null&&g.model_total!=null)e.push(Math.abs(g.model_total-m.total));
 return e.length?Math.max(...e):-1;}
function draw(){
 const rows=G.filter(keep);const tb=document.getElementById('rows');
 if(!rows.length){tb.innerHTML='<tr><td colspan="6" class="muted">No games match.</td></tr>';return;}
 // Upcoming games grouped by day, finished games at the bottom
 const up=rows.filter(g=>!g.completed),done=rows.filter(g=>g.completed);
 const groups=[];
 if(sortBy==='edge'){groups.push(['Biggest edge first (spread or total)',[...up].sort((a,b)=>maxEdge(b)-maxEdge(a))]);}
 else up.forEach(g=>{const k=day(g.start);if(!groups.length||groups[groups.length-1][0]!==k)groups.push([k,[]]);groups[groups.length-1][1].push(g);});
 if(done.length)groups.push(['Final',done]);
 const n=k=>`${k.length} game${k.length===1?'':'s'}`;
 tb.innerHTML=groups.map(([k,gs])=>`<tr class="day"><td colspan="6">${k} · ${n(gs)}</td></tr>`+gs.map(rowHtml).join('')).join('');
 tb.querySelectorAll('tr.g').forEach(tr=>tr.onclick=()=>{const d=document.getElementById('d'+tr.dataset.id);d.hidden=!d.hidden;});}
// This week's bets and flags, above the table
function summary(){
 const open=G.filter(g=>!g.started);const items=[];
 open.forEach(g=>{const m=g.market||{};const gm=`${esc(g.away)} ${g.neutral?'vs':'@'} ${esc(g.home)} · ${when(g.start)}`;
  const low=(g.low_conf||[]);
  if(g.spread_bet){const b=g.spread_bet,home=b.side==='home';items.push({k:'bet',g,txt:`${esc(home?g.home:g.away)} ${sl(home?b.line:-b.line)}`,gm,
   edge:Math.abs(b.line-g.model_spread),note:[m.spread!=null&&m.spread!==b.line?`line now ${sl(home?m.spread:-m.spread)}`:'',...low.map(REASON)].filter(Boolean).join(' · ')});}
  else if(g.spread_flag&&low.length){const p=spreadPick(g,m.spread,g.model_spread);if(p&&!p.none)items.push({k:'flag',g,txt:`${esc(p.team)} ${sl(p.line)}`,gm,edge:p.edge,note:low.map(REASON).join(' · ')});}
  if(g.total_bet){const b=g.total_bet;items.push({k:'bet',g,txt:`${b.side==='over'?'Over':'Under'} ${b.line}`,gm,edge:Math.abs(g.model_total-b.line),
   note:[m.total!=null&&m.total!==b.line?`line now ${m.total}`:'',...[...low,...(g.total_caution||[])].map(REASON)].filter(Boolean).join(' · ')});}
  else if(g.total_flag&&isLowTotal(g)){const p=totalPick(m.total,g.model_total);if(p&&!p.none)items.push({k:'flag',g,txt:`${p.side} ${p.line}`,gm,edge:p.edge,note:[...new Set([...low,...(g.total_caution||[])].map(REASON))].join(' · ')});}
 });
 const el=document.getElementById('summary');
 const row=i=>`<div class="sitem" data-id="${i.g.game_id}"><span class="pill ${i.k==='bet'?'p-good':'p-warn'}">${i.k==='bet'?'BET':'FLAG'}</span><span><span class="t ${i.k==='bet'?'bet':''}">${i.txt}</span> <span class="g2">${i.gm}</span></span><span class="r">edge ${i.edge.toFixed(1)}${i.note?` · ${esc(i.note)}`:''}</span></div>`;
 const b=items.filter(i=>i.k==='bet'),f=items.filter(i=>i.k==='flag');
 el.className='summary';
 el.innerHTML=`<div class="grp"><h3>This week's bets</h3>${b.length?b.map(row).join(''):'<div class="sub">No bets right now: no game has a clean 7-point edge.</div>'}</div>`+
  (f.length?`<div class="grp"><h3>Flags: check before betting</h3>${f.map(row).join('')}</div>`:'');
 el.querySelectorAll('.sitem').forEach(x=>x.onclick=()=>{cur='all';q='';hideLow=false;document.getElementById('q').value='';document.getElementById('low').classList.remove('on');
  fb.querySelectorAll('[data-k]').forEach(c=>c.classList.toggle('on',c.dataset.k==='all'));draw();
  const tr=document.getElementById('g'+x.dataset.id);document.getElementById('d'+x.dataset.id).hidden=false;tr.scrollIntoView({behavior:'smooth',block:'center'});
  tr.classList.add('flash');setTimeout(()=>tr.classList.remove('flash'),1500);});}
summary();
draw();
"""

REPORT_JS = r"""
document.getElementById('sub').textContent=`${D.season} season`;
const f1=x=>x==null?'—':x.toFixed(1);const s1=x=>x==null?'—':(Math.abs(x)<0.05?'0.0':(x>0?'+':'−')+Math.abs(x).toFixed(1));
const R=D.report,L=R.live||{},B=R.backtest||{};
const rp=r=>!r||(r.w+r.l+r.p)===0?'<span class="muted">—</span>':`${rec(r)} <span class="when nw">${pct(r.pct)}</span>`;
const card=(l,v)=>`<div class="card"><div class="l">${l}</div><div class="v">${v}</div></div>`;
document.getElementById('live').innerHTML=L.graded?[
 card('Games graded',L.graded),card('Team score MAE',f1(L.team_mae)),card('Margin MAE',f1(L.margin_mae)+(L.mkt_margin_mae?` <span class="sub">mkt ${f1(L.mkt_margin_mae)}</span>`:'')),
 card('Bets ATS',rp(L.ats)),card('Bets O/U',rp(L.ou)),card('ATS, all games',rp(L.ats_all)),card('O/U, all games',rp(L.ou_all)),
 card('Avg CLV',L.clv_avg==null?'—':s1(L.clv_avg)+' pts'),card('Beat closing line',L.clv_beat_pct==null?'—':pct(L.clv_beat_pct)+` <span class="sub">of ${L.clv_n}</span>`)].join('')
 :card('Games graded','0')+'<p class="sub" style="grid-column:1/-1">Results appear here once games the model projected before kickoff are final.</p>';
const bt=(B.seasons||[]).map(s=>`<tr><td>${s.season}${s.holdout?' <span class="pill p-acc">holdout</span>':''}</td><td>${s.games}</td><td>${f1(s.margin_mae)}</td><td class="hide-sm">${f1(s.mkt_margin_mae)}</td><td class="hide-sm">${f1(s.total_mae)}</td><td>${rp(s.ats_open)}</td><td>${rp(s.ats)}</td><td>${rp(s.ou_open)}</td><td class="hide-sm">${rp(s.ou)}</td><td class="hide-sm">${rp(s.ats_all)}</td></tr>`);
const o=B.overall;if(o)bt.push(`<tr><td><b>All</b></td><td>${o.games}</td><td>${f1(o.margin_mae)}</td><td class="hide-sm">${f1(o.mkt_margin_mae)}</td><td class="hide-sm">${f1(o.total_mae)}</td><td><b>${rp(o.ats_open)}</b></td><td><b>${rp(o.ats)}</b></td><td><b>${rp(o.ou_open)}</b></td><td class="hide-sm"><b>${rp(o.ou)}</b></td><td class="hide-sm"><b>${rp(o.ats_all)}</b></td></tr>`);
document.getElementById('bt').innerHTML=bt.join('')||'<tr><td colspan="10" class="muted">Backtest not available yet.</td></tr>';
if(o){const mv=B.line_move||{};document.getElementById('btcards').innerHTML=[
 card('Bets ATS vs open',rp(o.ats_open)),card('Bets O/U vs open',rp(o.ou_open)),
 card('All games ATS vs close',rp(o.ats_all)),card('All games O/U vs close',rp(o.ou_all)),
 card('Lines moving toward model',mv.pct==null?'—':pct(mv.pct)+` <span class="sub">of ${mv.n}</span>`),
 card('Margin MAE (model / market)',`${f1(o.margin_mae)} <span class="sub">/ ${f1(o.mkt_margin_mae)}</span>`)].join('');}
document.getElementById('cal').innerHTML=(B.calibration||[]).map(c=>`<tr><td>${c.bin}</td><td>${pct(c.predicted)}</td><td>${pct(c.actual)}</td><td>${c.n}</td></tr>`).join('')||'<tr><td colspan="4" class="muted">—</td></tr>';
document.getElementById('shrink').textContent=B.shrink_spread!=null?`Learned from this table: cover probabilities shown on the projections page keep ${pct(B.shrink_spread)} of the model's raw confidence for spreads and ${pct(B.shrink_total)} for totals.`:'';
const E=R.experiments||{};
if(E.tests){
 const ch=v=>v==null?'—':`<span class="${v>0?'up':v<0?'down':''}">${v>0?'−':v<0?'+':''}${Math.abs(v).toFixed(3)}</span>`;
 document.getElementById('expsub').textContent=`Each feature is tested on ${E.tune_seasons?E.tune_seasons.join('–'):'the tuning seasons'} against the core model (margin MAE ${f2(E.baseline&&E.baseline.margin)}, total MAE ${f2(E.baseline&&E.baseline.total)}). It's kept only if it lowers error by at least 0.05 points without hurting the other measure. Green means less error. Last tested ${new Date(E.run).toLocaleDateString()}.`;
 document.getElementById('exp').innerHTML=E.tests.map(t=>`<tr><td>${esc(t.feature)}</td><td>${t.skipped?'—':ch(t.gain_margin)}</td><td>${t.skipped?'—':ch(t.gain_total)}</td><td>${t.skipped?`<span class="muted">Not tested: ${esc(t.skipped)}</span>`:t.kept?'<span class="pill p-good">Kept</span>':'<span class="muted">Not kept</span>'}</td></tr>`).join('');
 const H=E.holdout||{};if(H.baseline&&H.selected)document.getElementById('exphold').textContent=`Holdout check on ${H.season}, a season never used for tuning: core model margin MAE ${f2(H.baseline.margin)} vs with kept features ${f2(H.selected.margin)}; total MAE ${f2(H.baseline.total)} vs ${f2(H.selected.total)}.`;
}
const S=B.segments||{};const names={margin_by_week:'Margin by week',margin_by_matchup:'Margin by matchup',margin_by_fav_size:'Margin by projected favorite size',total_by_week:'Total by week',total_by_matchup:'Total by matchup'};
document.getElementById('seg').innerHTML=Object.entries(names).filter(([k])=>S[k]).map(([k,n])=>`<h4 class="sub" style="margin:14px 0 6px">${n}</h4><div class="tbl"><table><thead><tr><th>Group</th><th>Games</th><th>Bias</th><th>MAE</th></tr></thead><tbody>${S[k].map(r=>`<tr><td>${r.group}</td><td>${r.n}</td><td class="${r.flag?'flag':''}">${s1(r.bias)}</td><td>${f1(r.mae)}</td></tr>`).join('')}</tbody></table></div>`).join('');
const pill=v=>v?`<span class="pill ${v==='win'?'p-good':v==='loss'?'p-bad':'p-acc'}">${v}</span>`:'<span class="muted">—</span>';
document.getElementById('hist').innerHTML=(D.history||[]).slice(0,200).map(p=>{const r=p.result;const fs=p.first_spread_flag,ft=p.first_total_flag;
 const clv=[r.clv_spread,r.clv_total].filter(x=>x!=null);
 const sp=spreadPick(p,r.close_spread,p.model_spread),tp=totalPick(r.close_total,p.model_total);
 const sCell=r.ats?`<span class="pill p-good">BET</span> <span class="bet">${esc(fs.side==='home'?p.home:p.away)} ${sl(fs.side==='home'?fs.line:-fs.line)}</span> ${resPill(r.ats)}`:
   (sp&&!sp.none?`${esc(sp.team)} ${sl(sp.line)} ${resPill(r.ats_all)}`:'<span class="muted">—</span>');
 const tCell=r.ou?`<span class="pill p-good">BET</span> <span class="bet">${ft.side==='over'?'Over':'Under'} ${fmt(ft.line)}</span> ${resPill(r.ou)}`:
   (tp&&!tp.none?`${tp.side} ${tp.line} ${resPill(r.ou_all)}`:'<span class="muted">—</span>');
 return `<tr><td>${esc(p.away)} ${p.neutral?'vs':'@'} <span class="home">${esc(p.home)}</span><div class="when">Week ${p.wk}</div></td><td>${r.away_pts}–${r.home_pts}</td><td class="hide-sm">${projScore(p)||'—'}</td>
 <td>${sCell}</td><td>${tCell}</td><td class="hide-sm">${clv.length?clv.map(sgn).join(', '):'—'}</td></tr>`;}).join('')||'<tr><td colspan="6" class="muted">No graded games yet.</td></tr>';
const st=R.state||{};// At a glance: the three things worth knowing, in plain words
(()=>{const o=B.overall||{},S=B.seasons||[],H=S.find(x=>x.holdout)||S[S.length-1]||{};const ph=B.by_phase||[];
 const lines=[];
 lines.push(L.graded?`<b>This season:</b> ${L.graded} games graded. Picks on every game are ${rec(L.ats_all)} against the spread and ${rec(L.ou_all)} on totals; ${(L.ats&&(L.ats.w+L.ats.l+L.ats.p))||(L.ou&&(L.ou.w+L.ou.l+L.ou.p))?`flagged bets are ${rec(L.ats)} ATS and ${rec(L.ou)} O/U.`:'no flagged bets have been graded yet.'}`:
  '<b>This season:</b> no games graded yet.');
 if(H.margin_mae!=null&&H.mkt_margin_mae!=null)lines.push(`<b>Model vs market:</b> on ${H.season}, a season the model never trained on, its average miss on the final margin was ${f1(H.margin_mae)} points vs ${f1(H.mkt_margin_mae)} for the closing line. The closing line is ${H.margin_mae>H.mkt_margin_mae?'still more accurate overall':'no more accurate'}: it sees injury and depth-chart news the model can't.`);
 if(o.ats_open&&(o.ats_open.w+o.ats_open.l))lines.push(`<b>Where the edge is:</b> flagged bets went ${rp(o.ats_open)} ATS and ${rp(o.ou_open)} O/U against opening lines in the backtest${ph.length?` (spreads by part of season: ${ph.map(x=>`${x.label} ${rp(x.ats_open)}`).join(', ')})`:''}. Bet early in the week, before the line moves.`);
 document.getElementById('glance').innerHTML=`<h3>At a glance</h3>${lines.map(x=>`<p class="gl">${x}</p>`).join('')}`;})();
document.getElementById('settings').innerHTML=`Prior strength ${st.lam} (games' worth, tuned ${st.tuned_at?new Date(st.tuned_at).toLocaleDateString():'—'}) · Game-script correlation ${f1(st.rho)} · Margin error SD ${f1(st.margin_sd)} · Total error SD ${f1(st.total_sd)}<br>Tuning results (margin MAE by prior strength): ${Object.entries(R.tune||{}).map(([k,v])=>`${k}: ${f2(v)}`).join(' · ')||'—'}`;
"""

RATINGS_BODY = NAV("t") + r"""
<p class="sub">Ranked by the model's power rating: how many points better than an average FBS team on a neutral field, using the same ratings and formula as the game projections. "Vs average team" is the projected score of that game. EPA is opponent-adjusted expected points per play with garbage time removed; for defense, lower is better.</p>
<div class="bar" id="tfilters"></div>
<div class="tbl"><table><thead><tr><th>#</th><th>Team</th><th>Rating</th><th>Vs average team</th><th class="hide-sm">Off EPA (rank)</th><th class="hide-sm">Def EPA (rank)</th><th class="hide-sm">Plays/game</th></tr></thead><tbody id="teams"></tbody></table></div>
"""

RATINGS_JS = r"""
document.getElementById('sub').textContent=`${D.season} season · through Week ${D.week??'—'}`;
const f1=x=>x==null?'—':x.toFixed(1);const s1=x=>x==null?'—':(Math.abs(x)<0.05?'0.0':(x>0?'+':'−')+Math.abs(x).toFixed(1));
const T=(D.teams||[]).map((t,i)=>({...t,rank:i+1}));
let tc='all',tq='';const tf=document.getElementById('tfilters');
tf.innerHTML=[['all','All FBS'],['P4','Power 4'],['G5','Group of 5']].map(([k,l])=>`<button class="chip${k==='all'?' on':''}" data-k="${k}">${l}</button>`).join('')+'<input type="search" id="tq" placeholder="Find a team">';
tf.querySelectorAll('[data-k]').forEach(b=>b.onclick=()=>{tc=b.dataset.k;tf.querySelectorAll('[data-k]').forEach(x=>x.classList.toggle('on',x===b));drawT();});
document.getElementById('tq').oninput=e=>{tq=e.target.value.toLowerCase();drawT();};
function drawT(){const rows=T.filter(t=>(tc==='all'||t.tier===tc)&&(!tq||t.team.toLowerCase().includes(tq)));
 document.getElementById('teams').innerHTML=rows.map(t=>`<tr><td>${t.rank}</td><td>${esc(t.team)} <span class="when">${t.tier}</span></td><td><b>${s1(t.power)}</b></td><td>${t.pf==null?'—':`${f1(t.pf)}–${f1(t.pa)}`}</td><td class="hide-sm">${epa(t.off)} <span class="when">${t.off_rank}</span></td><td class="hide-sm">${epa(t.deff)} <span class="when">${t.def_rank}</span></td><td class="hide-sm">${f1(t.pace)}</td></tr>`).join('')||'<tr><td colspan="7" class="muted">No teams match.</td></tr>';}
drawT();
"""

INDEX_BODY = INDEX_BODY.replace("<script>window.PAGE='index'</script>", "")
REPORT_BODY = REPORT_BODY.replace("<script>window.PAGE='report'</script>", "")


def _page(body, js, title, ver="0"):
    return PAGE.replace("__VER__", ver).replace("__BODY__", body).replace("__SCRIPT__", js).replace("__TITLE__", title)
