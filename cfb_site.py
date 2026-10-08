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


def team_table(final):
    Rp, Rc = final["ppa"], final["pace"]
    rows = []
    for t, tier in final["tiers"].items():
        if tier == "FCS" or t not in Rp["off"]:
            continue
        rows.append(dict(team=t, tier=tier, off=Rp["off"][t], deff=Rp["deff"][t],
                         net=Rp["off"][t] - Rp["deff"][t], pace=Rc["mu"] + Rc["off"][t]))
    df = pd.DataFrame(rows).sort_values("net", ascending=False)
    df["off_rank"] = df.off.rank(ascending=False).astype(int)
    df["def_rank"] = df.deff.rank(ascending=True).astype(int)
    return df.to_dict("records")


def build(season, wk, slate, store, rep, final, now, unmatched):
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
        games.append(g)
    history = sorted(
        [p for p in store.values() if p.get("result")],
        key=lambda p: p["start"], reverse=True,
    )[:400]
    payload = _clean(dict(
        generated=now.isoformat(), season=season, week=wk, games=games, report=rep,
        teams=team_table(final), history=history, unmatched=unmatched,
        thresholds=dict(spread=C.SPREAD_EDGE, total=C.TOTAL_EDGE),
    ))
    C.SITE.mkdir(exist_ok=True)
    (C.SITE / "data.js").write_text("window.DATA = " + json.dumps(payload, separators=(",", ":")) + ";")
    (C.SITE / "index.html").write_text(_page(INDEX_BODY, INDEX_JS, "Projections"))
    (C.SITE / "report.html").write_text(_page(REPORT_BODY, REPORT_JS, "Report card"))
    (C.SITE / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    (C.SITE / ".nojekyll").write_text("")


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
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
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.card{background:var(--soft);border-radius:10px;padding:12px 14px}
.card .l{font-size:12px;color:var(--muted)}.card .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.bar{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
.chip{font:inherit;font-size:13px;padding:5px 11px;border-radius:999px;border:1px solid var(--line);background:var(--card);color:var(--muted);cursor:pointer}
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
@media (max-width:720px){.hide-sm{display:none}td,th{padding:8px 6px;font-size:13px}}
</style></head><body><div class="wrap">
__BODY__
<p class="note" id="foot"></p>
</div>
<script src="data.js"></script>
<script>
const D=window.DATA;
const fmt=x=>x==null?'—':(Math.round(x*10)/10).toString();
const sgn=x=>x==null?'—':(x>0?'+':x<0?'−':'')+Math.abs(Math.round(x*10)/10);
const pct=x=>x==null?'—':Math.round(x*100)+'%';
const pct1=x=>x==null?'—':(Math.round(x*1000)/10)+'%';
const epa=x=>x==null?'—':(x>0?'+':x<0?'−':'')+Math.abs(x).toFixed(3);
const rec=r=>!r||(r.w+r.l+r.p)===0?'—':`${r.w}–${r.l}${r.p?'–'+r.p:''}`;
const when=s=>new Date(s).toLocaleString('en-US',{timeZone:'America/Chicago',weekday:'short',hour:'numeric',minute:'2-digit'});
function line(home,away,s){if(s==null)return '—';if(s===0)return 'Pick';return s<0?`${home} −${Math.abs(s)}`:`${away} −${s}`;}
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('foot').innerHTML=`Updated ${new Date(D.generated).toLocaleString('en-US',{timeZone:'America/Chicago'})} CT · Data from CollegeFootballData.com, The Odds API, Open-Meteo.`+(D.unmatched&&D.unmatched.length?` · Unmatched odds games: ${D.unmatched.length}`:'');
</script>
<script>__SCRIPT__</script>
</body></html>"""

NAV = lambda on: f"""<header><div><h1>College football model</h1><div class="sub" id="sub"></div></div>
<nav><a href="index.html" class="{'on' if on=='p' else ''}">Projections</a><a href="report.html" class="{'on' if on=='r' else ''}">Report card</a></nav></header>"""

INDEX_BODY = NAV("p") + r"""
<div class="cards" id="cards"></div>
<div class="bar" id="filters"></div>
<div class="tbl"><table><thead><tr>
<th>Game</th><th>Projected</th><th>Spread · model / market</th><th class="hide-sm">Total · model / market</th><th class="hide-sm">Line move</th><th>Edge</th>
</tr></thead><tbody id="rows"></tbody></table></div>
<p class="note">Home team in bold. Click a game for details. Edges show only when the model and the current line disagree by at least the threshold; amber means the edge rests on shaky inputs. "Check news" marks games where the model and market differ by 8+ points: in past seasons the market was closer in about two of three of those games, usually because of injury or QB news the model can't see. The backtest shows the model's edge is against early-week lines; by kickoff the market has usually caught up, so an edge is only worth acting on early, and only if the line hasn't already moved to the model's number. Projected scores are the simulation's most typical result; lines are never used as model inputs.</p>
""".replace("</div>\n<p", "</div>\n<p", 1) + "<script>window.PAGE='index'</script>"

REPORT_BODY = NAV("r") + r"""
<h2>This season (live, graded at kickoff)</h2>
<div class="cards" id="live"></div>
<h2>Backtest by season</h2>
<p class="sub">Walk-forward: each week predicted using only data available before it. Flagged bets are graded against the opening line (when the model's edges are meant to be bet) and the closing line. Opening lines exist for only part of the history.</p>
<div class="cards" id="btcards"></div>
<div class="tbl"><table><thead><tr><th>Season</th><th>Games</th><th>Margin MAE</th><th class="hide-sm">Market margin MAE</th><th class="hide-sm">Total MAE</th><th>ATS vs open</th><th>ATS vs close</th><th>O/U vs open</th><th class="hide-sm">O/U vs close</th></tr></thead><tbody id="bt"></tbody></table></div>
<h2>Model improvements tested</h2>
<p class="sub" id="expsub"></p>
<div class="tbl"><table><thead><tr><th>Feature</th><th>Margin error change</th><th>Total error change</th><th>Result</th></tr></thead><tbody id="exp"></tbody></table></div>
<p class="sub" id="exphold"></p>
<p class="sub">Tested on 2018–2024 and removed because they didn't improve accuracy: weather at kickoff, travel and rest, recency weighting, run/pass matchups, fitting the margin directly, and recruiting/portal/coaching/defensive-returning preseason data. QB changes are shown as alerts instead of being a model input.</p>
<h2>Calibration</h2>
<p class="sub">When the model gives the home team X% to cover, how often do they?</p>
<div class="tbl"><table><thead><tr><th>Model said</th><th>Average</th><th>Actually covered</th><th>Games</th></tr></thead><tbody id="cal"></tbody></table></div>
<p class="sub" id="shrink"></p>
<h2>Error diagnostics</h2>
<p class="sub">Bias = average of (predicted − actual). Red rows are biases unlikely to be chance (100+ games, more than 2.5 standard errors).</p>
<div id="seg"></div>
<h2>Recent results</h2>
<div class="tbl"><table><thead><tr><th>Game</th><th>Final</th><th>Projected</th><th class="hide-sm">Bet line</th><th>Spread</th><th>Total</th><th class="hide-sm">CLV</th></tr></thead><tbody id="hist"></tbody></table></div>
<h2>Team ratings</h2>
<p class="sub">Opponent-adjusted EPA per play (garbage time excluded). Net = offense − defense allowed.</p>
<div class="tbl"><table><thead><tr><th>#</th><th>Team</th><th>Net</th><th>Off (rank)</th><th>Def (rank)</th><th class="hide-sm">Plays/game</th></tr></thead><tbody id="teams"></tbody></table></div>
<h2>Model settings</h2><div class="sub" id="settings"></div>
<script>window.PAGE='report'</script>
"""

INDEX_JS = r"""
const G=D.games;
document.getElementById('sub').textContent=`${D.season} season · Week ${D.week??'—'}`;
const L=D.report.live||{};
const edges=G.filter(g=>(g.spread_flag||g.total_flag)&&!g.started&&!(g.low_conf||[]).length);
document.getElementById('cards').innerHTML=[
['Games',G.length],['Edges this week',edges.length],['Season ATS (flagged)',rec(L.ats)],['Season O/U (flagged)',rec(L.ou)],
['Avg CLV',L.clv_avg==null?'—':sgn(L.clv_avg)+' pts']].map(([l,v])=>`<div class="card"><div class="l">${l}</div><div class="v">${v}</div></div>`).join('');
const F=[['all','All games'],['edges','Edges'],['spread','Spread edges'],['total','Total edges'],['P4','Power 4'],['G5','Group of 5']];
let cur='all',hideLow=false,q='';
const fb=document.getElementById('filters');
fb.innerHTML=F.map(([k,l])=>`<button class="chip${k==='all'?' on':''}" data-k="${k}">${l}</button>`).join('')+
`<button class="chip" id="low">Hide low confidence</button><input type="search" id="q" placeholder="Find a team">`;
fb.querySelectorAll('[data-k]').forEach(b=>b.onclick=()=>{cur=b.dataset.k;fb.querySelectorAll('[data-k]').forEach(x=>x.classList.toggle('on',x===b));draw();});
document.getElementById('low').onclick=e=>{hideLow=!hideLow;e.target.classList.toggle('on',hideLow);draw();};
document.getElementById('q').oninput=e=>{q=e.target.value.toLowerCase();draw();};
function keep(g){
 if(q&&!(g.home+' '+g.away).toLowerCase().includes(q))return false;
 if(hideLow&&(g.low_conf||[]).length)return false;
 if(cur==='edges')return g.spread_flag||g.total_flag;
 if(cur==='spread')return g.spread_flag;if(cur==='total')return g.total_flag;
 if(cur==='P4')return g.home_tier==='P4'||g.away_tier==='P4';
 if(cur==='G5')return g.home_tier==='G5'||g.away_tier==='G5';
 return true;}
function edgeCell(g){
 if(g.completed&&g.result){const r=g.result;const bits=[];if(r.ats)bits.push(['ATS',r.ats]);if(r.ou)bits.push(['O/U',r.ou]);
  return bits.length?bits.map(([k,v])=>`<span class="pill ${v==='win'?'p-good':v==='loss'?'p-bad':'p-acc'}">${k} ${v}</span>`).join(' '):'<span class="muted">—</span>';}
 const low=(g.low_conf||[]).length,cls=low?'p-warn':'p-good',out=[];
 if(g.spread_flag){const t=g.spread_side==='home'?g.home:g.away;const ln=g.market.spread;const tl=g.spread_side==='home'?ln:-ln;out.push(`<span class="pill ${cls}">${esc(t)} ${tl>0?'+':''}${tl===0?'PK':tl} · edge ${g.spread_edge}</span>`);}
 if(g.total_flag)out.push(`<span class="pill ${cls}">${g.total_side==='over'?'Over':'Under'} ${g.market.total} · edge ${g.total_edge}</span>`);
 if(!out.length)return '<span class="muted">None</span>';
 return out.join(' ')+(low?`<div class="when">${(g.low_conf||[]).join(', ')}</div>`:'');}
function moveCell(g){const m=g.market||{};if(m.open_spread==null||m.spread==null)return '<span class="muted">—</span>';
 const d=m.spread-m.open_spread;if(Math.abs(d)<0.25)return `<span class="muted">No move</span>`;
 let toward='';if(g.model_spread!=null){toward=Math.abs(g.model_spread-m.spread)<Math.abs(g.model_spread-m.open_spread)?'up':'down';}
 return `<span class="${toward}">${line(g.home,g.away,m.open_spread)} → ${line(g.home,g.away,m.spread)}</span><div class="when">${toward==='up'?'toward model':'away from model'}</div>`;}
function icons(g){const w=g.weather||{};let s='';if(w.wind>15)s+=` <span class="pill p-warn" title="Wind">${Math.round(w.wind)} mph</span>`;
 const n=g.notes||[];if(n.some(x=>x.includes('expected starter')))s+=' <span class="pill p-bad" title="QB change">QB</span>';
 if(n.some(x=>!x.includes('expected starter')))s+=' <span class="pill p-bad" title="Manual adjustment">adj</span>';
 if((g.low_conf||[]).some(x=>x.startsWith('big disagreement')))s+=' <span class="pill p-warn" title="Model and market differ a lot: check injury/QB news">check news</span>';return s;}
function detail(g){
 if(g.home_med==null)return '<div class="sub">No projection stored for this game.</div>';
 const h=g.hist||[],mx=Math.max(1,...h);const m=g.market||{};
 const bars=h.map((v,i)=>`<div class="${(i*3-45)<0?'z':''}" style="height:${Math.round(v/mx*100)}%" title="${i*3-45} to ${i*3-43}"></div>`).join('');
 const likely=(g.likely||[]).map(([a,b,p])=>`<div class="kv"><span>${esc(g.home)} ${a}, ${esc(g.away)} ${b}</span><span>${pct1(p)}</span></div>`).join('');
 const books=(m.books||[]).map(b=>`<div class="kv"><span>${esc(b.book)}</span><span>${line(g.home,g.away,b.spread)} · ${fmt(b.total)}</span></div>`).join('')||'<div class="sub">No sportsbook snapshot yet.</div>';
 const R=g.ratings||{};const wx=g.weather||{};
 return `<div class="dgrid">
 <div><h4>Probabilities</h4>
  <div class="kv"><span>${esc(g.home)} wins</span><span>${pct(g.p_home_win)}</span></div>
  <div class="kv"><span>${esc(g.away)} wins</span><span>${pct(1-g.p_home_win)}</span></div>
  ${g.p_side_cover!=null?`<div class="kv"><span>${esc(g.spread_side==='home'?g.home:g.away)} covers</span><span>${pct(g.p_side_cover)}</span></div>`:''}
  ${g.p_side_total!=null?`<div class="kv"><span>${g.total_side==='over'?'Over':'Under'} hits</span><span>${pct(g.p_side_total)}</span></div>`:''}
  <h4 style="margin-top:12px">Most likely final scores</h4>${likely}</div>
 <div><h4>Margin distribution (${esc(g.home)} − ${esc(g.away)})</h4><div class="hist">${bars}</div>
  <div class="kv sub"><span>−45</span><span>0</span><span>+45</span></div>
  <h4 style="margin-top:12px">Inputs</h4>
  <div class="kv"><span>Expected points</span><span>${fmt(g.exp_home)} – ${fmt(g.exp_away)}</span></div>
  <div class="kv"><span>Off EPA/play (home · away)</span><span>${epa(R.home_off)} · ${epa(R.away_off)}</span></div>
  <div class="kv"><span>Def EPA/play allowed</span><span>${epa(R.home_def)} · ${epa(R.away_def)}</span></div>
  <div class="kv"><span>Expected plays</span><span>${fmt(R.home_pace)} · ${fmt(R.away_pace)}</span></div>
  ${wx.dome?'<div class="kv"><span>Weather</span><span>Dome</span></div>':wx.wind!=null?`<div class="kv"><span>Weather</span><span>${Math.round(wx.temp)}°F · ${Math.round(wx.wind)} mph · ${fmt(wx.precip)} in</span></div>`:''}
  ${(g.notes||[]).length?`<div class="kv"><span>Adjustments</span><span>${esc(g.notes.join('; '))}</span></div>`:''}</div>
 <div><h4>Lines by book</h4>${books}
  <div class="kv" style="margin-top:8px"><span>Open</span><span>${line(g.home,g.away,m.open_spread)} · ${fmt(m.open_total)}</span></div>
  ${g.first_spread_flag?`<div class="kv"><span>First flagged</span><span>${esc(g.first_spread_flag.side==='home'?g.home:g.away)} at ${fmt(g.first_spread_flag.line)}</span></div>`:''}</div>
 </div>`;}
function draw(){
 const rows=G.filter(keep);const tb=document.getElementById('rows');
 if(!rows.length){tb.innerHTML='<tr><td colspan="6" class="muted">No games match.</td></tr>';return;}
 tb.innerHTML=rows.map(g=>{
  const at=g.neutral?'vs':'@';
  const proj=g.completed?`<span>${g.home_pts}–${g.away_pts}</span><div class="when">final${g.home_med!=null?` · proj ${g.home_med}–${g.away_med}`:''}</div>`:
   (g.home_med==null?'<span class="muted">—</span>':`${g.home_med}–${g.away_med}<div class="when">${pct(g.p_home_win)} ${esc(g.home)}</div>`);
  const m=g.market||{};
  return `<tr class="g" data-id="${g.game_id}"><td>${esc(g.away)} ${at} <span class="home">${esc(g.home)}</span>${icons(g)}<div class="when">${when(g.start)}${g.started&&!g.completed?' · in progress':''}</div></td>
  <td>${proj}</td><td>${line(g.home,g.away,g.model_spread)}<div class="when">mkt ${line(g.home,g.away,m.spread)}</div></td>
  <td class="hide-sm">${fmt(g.model_total)}<div class="when">mkt ${fmt(m.total)}</div></td><td class="hide-sm">${moveCell(g)}</td><td>${edgeCell(g)}</td></tr>
  <tr class="detail" id="d${g.game_id}" hidden><td colspan="6">${detail(g)}</td></tr>`;}).join('');
 tb.querySelectorAll('tr.g').forEach(tr=>tr.onclick=()=>{const d=document.getElementById('d'+tr.dataset.id);d.hidden=!d.hidden;});}
draw();
"""

REPORT_JS = r"""
document.getElementById('sub').textContent=`${D.season} season`;
const R=D.report,L=R.live||{},B=R.backtest||{};
const card=(l,v)=>`<div class="card"><div class="l">${l}</div><div class="v">${v}</div></div>`;
document.getElementById('live').innerHTML=L.graded?[
 card('Games graded',L.graded),card('Team score MAE',fmt(L.team_mae)),card('Margin MAE',fmt(L.margin_mae)+(L.mkt_margin_mae?` <span class="sub">mkt ${fmt(L.mkt_margin_mae)}</span>`:'')),
 card('Flagged ATS',rec(L.ats)+(L.ats&&L.ats.pct!=null?` <span class="sub">${pct(L.ats.pct)}</span>`:'')),card('Flagged O/U',rec(L.ou)),
 card('Avg CLV',L.clv_avg==null?'—':sgn(L.clv_avg)+' pts'),card('Beat closing line',L.clv_beat_pct==null?'—':pct(L.clv_beat_pct)+` <span class="sub">of ${L.clv_n}</span>`)].join('')
 :card('Games graded','0')+'<p class="sub" style="grid-column:1/-1">Results appear here once games the model projected before kickoff are final.</p>';
const rp=r=>r?`${rec(r)} <span class="when">${pct(r.pct)}</span>`:'—';
const bt=(B.seasons||[]).map(s=>`<tr><td>${s.season}${s.holdout?' <span class="pill p-acc">holdout</span>':''}</td><td>${s.games}</td><td>${fmt(s.margin_mae)}</td><td class="hide-sm">${fmt(s.mkt_margin_mae)}</td><td class="hide-sm">${fmt(s.total_mae)}</td><td>${rp(s.ats_open)}</td><td>${rp(s.ats)}</td><td>${rp(s.ou_open)}</td><td class="hide-sm">${rp(s.ou)}</td></tr>`);
const o=B.overall;if(o)bt.push(`<tr><td><b>All</b></td><td>${o.games}</td><td>${fmt(o.margin_mae)}</td><td class="hide-sm">${fmt(o.mkt_margin_mae)}</td><td class="hide-sm">${fmt(o.total_mae)}</td><td><b>${rp(o.ats_open)}</b></td><td><b>${rp(o.ats)}</b></td><td><b>${rp(o.ou_open)}</b></td><td class="hide-sm"><b>${rp(o.ou)}</b></td></tr>`);
document.getElementById('bt').innerHTML=bt.join('')||'<tr><td colspan="9" class="muted">Backtest not available yet.</td></tr>';
if(o){const mv=B.line_move||{};document.getElementById('btcards').innerHTML=[
 card('Flagged ATS vs open',rp(o.ats_open)),card('Flagged O/U vs open',rp(o.ou_open)),
 card('Lines moving toward model',mv.pct==null?'—':pct(mv.pct)+` <span class="sub">of ${mv.n}</span>`),
 card('Margin MAE (model / market)',`${fmt(o.margin_mae)} <span class="sub">/ ${fmt(o.mkt_margin_mae)}</span>`)].join('');}
document.getElementById('cal').innerHTML=(B.calibration||[]).map(c=>`<tr><td>${c.bin}</td><td>${pct(c.predicted)}</td><td>${pct(c.actual)}</td><td>${c.n}</td></tr>`).join('')||'<tr><td colspan="4" class="muted">—</td></tr>';
document.getElementById('shrink').textContent=B.shrink_spread!=null?`Learned from this table: cover probabilities shown on the projections page keep ${pct(B.shrink_spread)} of the model's raw confidence for spreads and ${pct(B.shrink_total)} for totals.`:'';
const E=R.experiments||{};
if(E.tests){
 const ch=v=>v==null?'—':`<span class="${v>0?'up':v<0?'down':''}">${v>0?'−':v<0?'+':''}${Math.abs(v).toFixed(3)}</span>`;
 document.getElementById('expsub').textContent=`Each feature is tested on ${E.tune_seasons?E.tune_seasons.join('–'):'the tuning seasons'} against the baseline (margin MAE ${fmt(E.baseline&&E.baseline.margin)}, total MAE ${fmt(E.baseline&&E.baseline.total)}). It's kept only if it lowers error by at least 0.05 points without hurting the other measure. Green means less error. Last tested ${new Date(E.run).toLocaleDateString()}.`;
 document.getElementById('exp').innerHTML=E.tests.map(t=>`<tr><td>${esc(t.feature)}</td><td>${t.skipped?'—':ch(t.gain_margin)}</td><td>${t.skipped?'—':ch(t.gain_total)}</td><td>${t.skipped?`<span class="muted">Not tested: ${esc(t.skipped)}</span>`:t.kept?'<span class="pill p-good">Kept</span>':'<span class="muted">Not kept</span>'}</td></tr>`).join('');
 const H=E.holdout||{};if(H.baseline&&H.selected)document.getElementById('exphold').textContent=`Holdout check on ${H.season}, a season never used for tuning: baseline margin MAE ${fmt(H.baseline.margin)} vs selected features ${fmt(H.selected.margin)}; total MAE ${fmt(H.baseline.total)} vs ${fmt(H.selected.total)}.`;
}
const S=B.segments||{};const names={margin_by_week:'Margin by week',margin_by_matchup:'Margin by matchup',margin_by_fav_size:'Margin by projected favorite size',total_by_week:'Total by week',total_by_matchup:'Total by matchup'};
document.getElementById('seg').innerHTML=Object.entries(names).filter(([k])=>S[k]).map(([k,n])=>`<h4 class="sub" style="margin:14px 0 6px">${n}</h4><div class="tbl"><table><thead><tr><th>Group</th><th>Games</th><th>Bias</th><th>MAE</th></tr></thead><tbody>${S[k].map(r=>`<tr><td>${r.group}</td><td>${r.n}</td><td class="${r.flag?'flag':''}">${sgn(r.bias)}</td><td>${fmt(r.mae)}</td></tr>`).join('')}</tbody></table></div>`).join('');
const pill=v=>v?`<span class="pill ${v==='win'?'p-good':v==='loss'?'p-bad':'p-acc'}">${v}</span>`:'<span class="muted">—</span>';
document.getElementById('hist').innerHTML=(D.history||[]).slice(0,150).map(p=>{const r=p.result;const fs=p.first_spread_flag;
 const clv=[r.clv_spread,r.clv_total].filter(x=>x!=null);
 return `<tr><td>${esc(p.away)} @ <span class="home">${esc(p.home)}</span><div class="when">Week ${p.wk}</div></td><td>${r.home_pts}–${r.away_pts}</td><td>${p.home_med}–${p.away_med}</td>
 <td class="hide-sm">${fs?esc((fs.side==='home'?p.home:p.away))+' '+fmt(fs.side==='home'?fs.line:-fs.line):'—'}</td><td>${pill(r.ats)}</td><td>${pill(r.ou)}</td><td class="hide-sm">${clv.length?clv.map(sgn).join(', '):'—'}</td></tr>`;}).join('')||'<tr><td colspan="7" class="muted">No graded games yet.</td></tr>';
document.getElementById('teams').innerHTML=(D.teams||[]).map((t,i)=>`<tr><td>${i+1}</td><td>${esc(t.team)} <span class="when">${t.tier}</span></td><td>${epa(t.net)}</td><td>${epa(t.off)} <span class="when">${t.off_rank}</span></td><td>${epa(t.deff)} <span class="when">${t.def_rank}</span></td><td class="hide-sm">${fmt(t.pace)}</td></tr>`).join('');
const st=R.state||{};document.getElementById('settings').innerHTML=`Prior strength ${st.lam} (games' worth, tuned ${st.tuned_at?new Date(st.tuned_at).toLocaleDateString():'—'}) · Game-script correlation ${fmt(st.rho)} · Margin error SD ${fmt(st.margin_sd)} · Total error SD ${fmt(st.total_sd)}<br>Tuning results (margin MAE by prior strength): ${Object.entries(R.tune||{}).map(([k,v])=>`${k}: ${fmt(v)}`).join(' · ')||'—'}`;
"""

INDEX_BODY = INDEX_BODY.replace("<script>window.PAGE='index'</script>", "")
REPORT_BODY = REPORT_BODY.replace("<script>window.PAGE='report'</script>", "")


def _page(body, js, title):
    return PAGE.replace("__BODY__", body).replace("__SCRIPT__", js).replace("__TITLE__", title)
