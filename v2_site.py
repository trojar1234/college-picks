"""Model 2.0 page (docs/v2.html + docs/v2data.js), styled like the rest of the site."""
import json

import cfb_config as C
import cfb_site as site

BODY = site.NAV("v2", "Model 2.0") + r"""
<div class="summary" id="intro"></div>
<h2 class="mh">This week</h2>
<details class="help"><summary>How to read this table</summary><ul>
<li><b>Model 2.0</b> is an ensemble of five new models (Bayesian ratings, a possession model, a matchup model, a personnel model and machine learning) plus the current model, weighted by how accurate each was on past seasons it never saw.</li>
<li><b>Score</b> is the projected score (away–home); <b>Win</b> the favorite's chance; <b>80% range</b> the home team's final margin, which should land in that range 4 times out of 5 (EMU +3 to +20 means Eastern Michigan winning by 3 to 20). Ranges are wider early in the season and for teams the model knows less about.</li>
<li>Each game's detail shows how far apart the five new models are. That spread has <b>not</b> predicted accuracy on past seasons (in 2026 the games where they agreed most had the largest misses), so it is not a confidence rating and changes no probability.</li>
<li><b>Line</b> is the sportsbooks' current number, shown for comparison. Lines are never inputs. <b>No bets are recommended</b>: Model 2.0 is in research mode until it proves itself on games it has not seen.</li>
<li>Tap a game for each model's number, the likeliest scores and the chances at the current line.</li></ul></details>
<div class="tbl"><table id="wk"><thead><tr><th>Game</th><th>Score</th><th>Spread</th><th>Total</th><th>Win</th><th class="hide-sm">80% range</th><th>Line</th></tr></thead><tbody></tbody></table></div>
<h2 class="mh">Live scorecard</h2>
<div id="live"></div>
<h2 class="mh">Backtest</h2>
<details class="help"><summary>How the backtest works</summary><ul>
<li>Every number is a walk-forward forecast: each game predicted only from games before it, and every model refit each season on earlier seasons only.</li>
<li><b>Tune</b> (2017–2022) is where every design choice was made. <b>Validate</b> (2023–2024) checked those choices. <b>Test</b> (2025) was opened once, after the design was fixed. <b>2026</b> is research: it shaped too many decisions on the current model to count as unseen.</li>
<li>Games between two FBS teams only, and every model is scored on the same games; the opening and closing lines on the subset that has them. Average miss is in points; Brier score measures win chances (lower is better); "80% range hit" should be near 80%. For Model 2.0 these come from the same score simulator the page uses.</li>
<li>The opening and closing lines are included as forecasts to beat, not as inputs.</li></ul></details>
<div class="presets" id="splits"></div>
<div id="bt"></div>
<h2 class="mh">How Model 2.0 works</h2>
<div class="summary" id="how"></div>
"""

JS = r"""
const V=window.V2;
const fmt=x=>x==null?'—':(Math.round(x*10)/10).toString();
const f2=x=>x==null?'—':Number(x).toFixed(2);
const f3=x=>x==null?'—':Number(x).toFixed(3);
const pct=x=>x==null?'—':Math.round(x*100)+'%';
const sg=x=>x==null?'—':(Math.round(x)>0?'+':Math.round(x)<0?'−':'')+Math.abs(Math.round(x));
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const when=s=>new Date(s).toLocaleString('en-US',{timeZone:'America/Chicago',weekday:'short',hour:'numeric',minute:'2-digit'});
const line=(h,a,s)=>s==null?'—':s===0?'Pick':s<0?`${esc(h)} −${Math.abs(s)}`:`${esc(a)} −${s}`;
const NAMES={bayes:'Bayesian ratings',poss:'Possession model',matchup:'Matchup model',personnel:'Personnel model',ml:'Machine learning',v1:'Current model (v1)',ens:'Model 2.0 (ensemble)',massey:'Massey (baseline)',elo:'Elo (baseline)',open:'Opening line',close:'Closing line'};
function score(g){const h=Math.round(g.exp_home),a=Math.round(g.exp_away);return `${a}–${h}`;}
function render(){
 document.getElementById('sub').textContent=`Updated ${when(V.generated)} CT · ${V.week>20?'Bowls':'Week '+V.week} · research mode, no bets`;
 const L=V.live||{};
 const age=(Date.now()-new Date(V.generated))/36e5;
 if(age>36)document.getElementById('intro').insertAdjacentHTML('beforebegin',`<div class="summary"><b>These projections are ${Math.round(age/24)} day(s) old.</b> Model 2.0's last update did not finish; the current model's pages are unaffected.</div>`);
 document.getElementById('intro').innerHTML=`<p><b>Research mode.</b> Model 2.0 runs alongside the current model. Its projections are frozen at kickoff and graded, and nothing is recommended as a bet. It replaces the current model only if, on games neither has seen, it is more accurate and its probabilities hold up.</p>`+
 `<p class="muted">On past seasons it has been slightly more accurate than the current model on margins (0.04 to 0.17 points a game; clearly better only in 2017–2022) and about even on totals. Both remain less accurate than the closing line.</p>`;
 const tb=document.querySelector('#wk tbody');
 tb.innerHTML=(V.games||[]).map(g=>{const fav=g.margin>=0?g.home:g.away,pw=g.margin>=0?g.p_home:1-g.p_home;const m=g.market||{};
  const lo=g.m_lo,hi=g.m_hi;
  return `<tr data-id="${g.game_id}" style="cursor:pointer"><td><b>${esc(g.away)}</b> ${g.neutral?'vs':'@'} <b>${esc(g.home)}</b><br><span class="muted">${when(g.start)}</span></td>
  <td class="mono">${score(g)}</td><td class="mono">${line(g.home,g.away,g.spread)}</td><td class="mono">${fmt(g.total)}</td>
  <td>${esc(fav)} ${pct(pw)}</td><td class="hide-sm mono">${esc(g.home)} ${sg(lo)} to ${sg(hi)}</td>
  <td class="mono">${line(g.home,g.away,m.spread)}<br>${m.total==null?'—':'O/U '+m.total}</td></tr>
  <tr class="det" id="d${g.game_id}" hidden><td colspan="7">${detail(g)}</td></tr>`}).join('')||'<tr><td colspan="7">No upcoming games.</td></tr>';
 tb.querySelectorAll('tr[data-id]').forEach(r=>r.onclick=()=>{const d=document.getElementById('d'+r.dataset.id);d.hidden=!d.hidden;});
 document.getElementById('live').innerHTML=liveCard(L);
 const sp=['tune','validate','test','2026'];let cur='test';
 const lab={tune:'Tune 2017–22',validate:'Validate 2023–24',test:'Test 2025',2026:'2026'};
 const bar=document.getElementById('splits');
 const draw=()=>{bar.innerHTML=sp.map(s=>`<button class="chip${s===cur?' on':''}" data-s="${s}">${lab[s]}</button>`).join('');
  bar.querySelectorAll('button').forEach(b=>b.onclick=()=>{cur=b.dataset.s;draw();});
  document.getElementById('bt').innerHTML=btCard((V.backtest||{})[cur]);};
 draw();
 document.getElementById('how').innerHTML=HOW;
}
function detail(g){const c=g.comp||{};const m=g.market||{};
 const rows=Object.keys(c).map(k=>`<div class="kv"><span>${NAMES[k]||k}</span><span class="mono">${line(g.home,g.away,c[k][0]==null?null:-c[k][0])} · ${fmt(c[k][1])}</span></div>`).join('');
 const lk=(g.likely||[]).map(x=>`${x[1]}–${x[0]} (${Math.round(x[2]*1000)/10}%)`).join(', ');
 let mk='';if(m.spread!=null&&g.p_home_cover!=null){const hc=g.p_home_cover;mk+=`<div class="kv"><span>Chance ${esc(g.home)} covers ${m.spread>0?'+':''}${m.spread}</span><span>${pct(hc)}</span></div>`;}
 if(m.total!=null&&g.p_over!=null)mk+=`<div class="kv"><span>Chance of over ${m.total}</span><span>${pct(g.p_over)}</span></div>`;
 return `<div class="dgrid"><div><b>Each model (spread · total)</b>${rows}<p class="muted">Spread of the five new models: ${fmt(g.models_sd)} points (not a confidence rating).</p></div><div><b>Likeliest scores (away–home)</b><p>${lk||'—'}</p>
 <div class="kv"><span>80% range, total</span><span>${fmt(g.t_lo)} to ${fmt(g.t_hi)}</span></div>${mk}
 <p class="muted">Chances at the current line come straight from the model and are not adjusted toward the market. Research only.</p></div></div>`;}
function liveCard(L){if(!L||!L.n)return `<div class="summary"><p>No graded games yet. Model 2.0's first frozen projections are graded as games finish; this card fills in from there.</p></div>`;
 const r=(x)=>x&&x[0]!=null?`${f2(x[0])} <span class="muted">(${x[1]})</span>`:'—';
 const rc=x=>x&&(x.w+x.l)?`${x.w}–${x.l}${x.p?'–'+x.p:''} (${pct(x.pct)})`:'—';
 return `<div class="tbl"><table><thead><tr><th>Average miss since ${new Date(L.since).toLocaleDateString('en-US')}</th><th>Margin</th><th>Total</th></tr></thead><tbody>
 <tr><td>Model 2.0</td><td>${r(L.v2.margin)}</td><td>${r(L.v2.total)}</td></tr>
 <tr><td>Current model (v1)</td><td>${r(L.v1.margin)}</td><td>${r(L.v1.total)}</td></tr>
 <tr><td>Opening line</td><td>${r(L.open.margin)}</td><td>${r(L.open.total)}</td></tr>
 <tr><td>Last line before kickoff</td><td>${r(L.close.margin)}</td><td>${r(L.close.total)}</td></tr></tbody></table></div>
 <p class="muted">Games graded: ${L.n} (FBS vs FBS). Win-chance Brier score: ${f3(L.v2.brier)}. Model 2.0's side against the opening line: spreads ${rc(L.leans.spread_open)}, totals ${rc(L.leans.total_open)}; against the last line: spreads ${rc(L.leans.spread_close)}, totals ${rc(L.leans.total_close)}. These are research records, not bets: break-even at −110 is 52.4%.</p>`;}
function btCard(R){if(!R)return '<p>Backtest not available yet.</p>';
 const A=R.accuracy||[];const order=['ens','v1','bayes','poss','matchup','personnel','ml','massey','elo','open','close'];
 const rows=order.map(k=>A.find(a=>a.model===k)).filter(Boolean).map(a=>`<tr${a.model==='ens'?' style="font-weight:600"':''}><td>${NAMES[a.model]||a.model}</td><td>${f2(a.margin_mae)}${a.model==='open'||a.model==='close'?` <span class="muted">(${a.n})</span>`:''}</td><td>${a.total_mae==null?'—':f2(a.total_mae)}</td><td>${fmt(a.margin_bias)}</td><td>${a.total_bias==null?'—':fmt(a.total_bias)}</td><td>${a.brier==null?'—':f3(a.brier)}</td><td>${a.cover80_margin==null?'—':pct(a.cover80_margin)}</td></tr>`).join('');
 const n=(A[0]||{}).n;
 const pv=(R.vs_v1||[]).filter(Boolean).map(p=>`${p.kind}: ${p.gain>=0?'+':''}${f2(p.gain)} points a game (95% range ${p.range95?f2(p.range95[0])+' to '+f2(p.range95[1]):'—'})`).join('; ');
 const mi=R.market_info||{};const ml=k=>mi[k]?`${f2(mi[k].w)} ± ${f2(mi[k].se)}`:'—';
 const C=R.confidence||{};
 const pk=(R.picks||[]).filter(x=>x.gap===0||x.gap===5).map(x=>`<tr><td>${x.kind==='margin'?'Spread':'Total'} vs ${x.line.includes('open')?'opening':'closing'} line</td><td>${x.gap?x.gap+'+':'all'}</td><td>${x.w}–${x.l}</td><td>${pct(x.pct)}</td><td>${x.units>=0?'+':''}${x.units}</td><td>${x.kind==='total'?(x.over_share==null?'—':pct(x.over_share)+' overs'):(x.home_share==null?'—':pct(x.home_share)+' home')}</td></tr>`).join('');
 return `<div class="tbl"><table><thead><tr><th>Forecast</th><th>Margin miss</th><th>Total miss</th><th>Margin bias</th><th>Total bias</th><th>Brier</th><th>80% range hit</th></tr></thead><tbody>${rows}</tbody></table></div>
 <p class="muted">${n||0} games. Bias = actual minus projected (positive: the forecast was too low on the home margin or total).</p>
 <div class="summary"><p><b>Model 2.0 vs current model:</b> ${pv||'—'} (positive = Model 2.0 more accurate; ranges resample whole weeks).</p>
 <p><b>What it knows beyond the line</b> (share of the model's gap that shows up in results; 0 = the line already knows it all): spreads ${ml('line_m_open')} vs opening, ${ml('line_m_close')} vs closing; totals ${ml('line_t_open')} vs opening, ${ml('line_t_close')} vs closing. A share of about 0.3 on a 5-point gap is worth 1.5 points, roughly 54% against the line; below about 0.2 a gap rarely clears the bookmaker's cut.</p>
 <p><b>Does model agreement predict accuracy?</b> Log score with it ${C.logscore_with??'—'}, without ${C.logscore_without??'—'} (higher is better). Average miss by agreement: ${Object.entries(C.by_agreement||{}).map(([k,v])=>`${k} ${f2(v.mae)} (${v.n})`).join(', ')||'—'}.</p></div>
 <h3>Research only: Model 2.0's side against the line</h3><div class="tbl"><table><thead><tr><th>Line</th><th>Gap</th><th>Record</th><th>Win %</th><th>Units</th><th>Lean</th></tr></thead><tbody>${pk}</tbody></table></div>
 <p class="muted">Not bets. Shown so any lean toward overs or home teams is visible. Break-even at −110 is 52.4%.</p>`;}
const HOW=`<p><b>1. Bayesian ratings.</b> Fourteen Kalman filters (points, points per drive, drives, rushing and passing efficiency and success, explosiveness, stuff rate, line yards, run share, scoring chances, points per scoring chance, field position) rate every team's offense and defense, each with its own uncertainty. Games are absorbed week by week; every forecast uses only earlier games. Between seasons ratings restart from a preseason prior fitted on earlier seasons (last season's rating, talent, recruiting, returning production, quarterback, transfers, roster experience), with wider uncertainty for teams returning less.</p>
<p><b>2. Possession model.</b> Points as drives times what each drive is worth (points per drive, how often drives reach the opponent's 40 and score there, field position, passing and rushing efficiency). It also powers the score simulator: drives end in touchdowns, field goals or nothing, so scores are always whole, never negative, and land on 3s and 7s as real games do.</p>
<p><b>3. Matchup model.</b> Ridge regression on unit-versus-unit ratings and their interactions (a run-heavy offense against a weak run defense, pass efficiency against pass defense, and so on).</p>
<p><b>4. Personnel model.</b> Model 1 corrected for roster talent, recruiting, transfers, returning production and quarterback changes, with preseason information fading as games accumulate. This data has no injury reports or depth charts; in-season quarterback changes come from the previous game's box score.</p>
<p><b>5. Machine learning.</b> Gradient-boosted trees on every rating and personnel input.</p>
<p><b>Ensemble.</b> Non-negative weights on the five models plus the current model, fitted each season on earlier seasons' unseen forecasts. Error sizes are game-specific (rating uncertainty, time of season, game size), also fitted only on earlier seasons.</p>
<p><b>Never an input:</b> sportsbook lines. They are used only to grade and compare. Model 2.0 makes no bets.</p>`;
render();
"""

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>Model 2.0 · CFB Model</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:wght@500;600;700&display=swap" rel="stylesheet">
<script>try{var t=localStorage.getItem('theme');if(t==='light'||t==='dark')document.documentElement.dataset.theme=t;}catch(e){}</script>
<style>__CSS__
.det td{background:var(--soft)}.dgrid{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:700px){.dgrid{grid-template-columns:1fr}}
.mh{font:600 22px/1.2 var(--disp);margin:28px 0 10px}</style></head><body>
__BODY__
<p class="note">Model 2.0 is a separate research model. Sportsbook lines are never inputs. Nothing here is betting advice.</p>
</main></div>
<script src="v2data.js?v=__VER__"></script>
<script>
(function(){var b=document.getElementById('theme');if(!b)return;var m=['auto','light','dark'];
function cur(){return document.documentElement.dataset.theme||'auto';}
function lab(){b.textContent=cur()[0].toUpperCase()+cur().slice(1);}
b.onclick=function(){var i=(m.indexOf(cur())+1)%3;if(m[i]==='auto'){delete document.documentElement.dataset.theme;try{localStorage.removeItem('theme')}catch(e){}}else{document.documentElement.dataset.theme=m[i];try{localStorage.setItem('theme',m[i])}catch(e){}}lab();};lab();})();
if(!window.V2){document.getElementById('intro').innerHTML='<b>Data unavailable.</b> Model 2.0 has not published yet.';}else{
__SCRIPT__
}
</script></body></html>"""


def _clean(o):
    return site._clean(o)


def build(data):
    ver = str(abs(hash(data.get("generated", ""))) % 10 ** 8)
    (C.SITE / "v2data.js").write_text("window.V2 = " + json.dumps(_clean(data), separators=(",", ":"), default=str) + ";\n")
    html = PAGE.replace("__CSS__", site.CSS).replace("__BODY__", BODY).replace("__SCRIPT__", JS).replace("__VER__", ver)
    (C.SITE / "v2.html").write_text(html)
