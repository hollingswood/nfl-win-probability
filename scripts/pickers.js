// Published pickers: summary strip, table and pop-up. ONE copy shared by the NFL and college pages (build_dashboard.py
// inlines it at __PICKERS_JS__), so a fix can't land on one page and not the other. Everything shown comes from
// history/picks_scores.json, whose table rows and pop-up lists are built from the same lists and checked by
// picks_score.audit before the file is written.
function renderPickers(K0, sport, A, opts){
  const el=id=>document.getElementById(id);
  const E=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  if(!K0 || !el('pickers')) return;
  const K={...((K0.by_sport||{})[sport]||{})};
  const ALL=K.pickers||[];
  if(!ALL.length && !(A && A.ok===false)) return;
  el('pickers').hidden=false;
  const t=K0.hot_test||{}, o=K.overall||{}, wl=(w,l,p)=>`${w}–${l}${p?`–${p}`:''}`;
  const az=s=>s?new Date(s).toLocaleString('en-US',{timeZone:'America/Phoenix',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})+' AZ':'—';
  const sk=K.left_out||{}, nSkip=Object.values(sk).reduce((a,b)=>a+b,0);
  const audit = A && A.ok===false
    ? `<div class="hu" role="alert">⚠ The newest picker records failed a consistency check at ${az(A.at)} (${A.n_errors||1} problem${(A.n_errors||1)===1?'':'s'}: ${E(String((A.errors||[])[0]||'').slice(0,160))}). The page keeps the last good records${A.last_good?` from ${az(A.last_good)}`:''} until it's fixed.</div>`
    : (K0.audit&&K0.audit.ok ? ` <b>Checked every run:</b> ${K0.audit.checks.toLocaleString()} consistency checks passed at ${az(K0.audit.at)} (each pick counted once per game, results recomputed from the final score and closing line, table = pop-ups).` : '');
  el('pickintro').innerHTML = `${opts.lead} Graded at the <b>closing</b> line; tap a name for their pending, graded and not-counted picks, with the reason for each.${A&&A.ok===false?audit:''}
    <details class="pkhow"><summary>How it's scored and checked</summary><p>${opts.intro} Picks first seen after kickoff don't count. <b>Line value</b> = points the market moved toward the pick after it was posted. Picks tables are read row by row, and every AI-read pick must match the article (quote, analyst, team, the table's row), the real game and its line, and the analyst's other picks on that game; the same pick seen twice counts once, at the first sighting.${nSkip?` ${nSkip.toLocaleString()} items were left out: ${Object.entries(sk).map(([k,v])=>`${v} ${E(k)}`).join('; ')}.`:''}${A&&A.ok===false?'':audit}</p></details>`;
  el('pickstrip').innerHTML=[
    [wl(o.w||0,o.l||0), `${opts.label} vs the close · ${(o.units||0)>0?'+':''}${(o.units||0).toFixed(1)} units`],
    [o.avg_line_value!=null?`${o.avg_line_value>0?'+':''}${o.avg_line_value}`:'—', 'avg. line value, pts'],
    [t.followed?wl(t.w,t.l):'—', `hot-picker test (NFL + college pooled): ${t.followed||0} / 200 followed picks${t.cover_rate!=null?` · ${Math.round(t.cover_rate*100)}% (everyone else ${t.others_rate!=null?Math.round(t.others_rate*100)+'%':'—'})`:''}`],
    [(K.hot_now||[]).length, (K.hot_now||[]).length?`hot now: ${K.hot_now.map(h=>`${E(h.picker)} (${h.record})`).join(', ')}`:'hot pickers now (none yet: needs 8+ graded picks in 4 weeks)'],
  ].map(([b,x])=>`<div class="stat"><b class="num">${E(String(b))}</b><span>${x}</span></div>`).join('');
  const R=ALL.slice(0,40);
  const outlet=x=>E(String(x.outlet||'').replace(/^bsky:.*/,'Bluesky').replace(/_/g,' '));
  const row=x=>`<tr><td style="text-align:left"><button type="button" class="pkbtn" data-picker="${E(x.picker)}" title="See their pending, graded and not-counted picks">${E(x.picker)}</button></td><td data-l="Outlet" style="text-align:left;font-family:var(--body)">${outlet(x)}</td><td data-l="Graded">${x.graded}${x.pending?`<small> · ${x.pending} pending</small>`:''}</td><td data-l="W–L vs close">${x.graded?wl(x.wins,x.losses,x.pushes):'—'}</td><td data-l="Units" class="${x.units>=0?'win':'loss'}">${x.graded?(x.units>0?'+':'')+x.units.toFixed(2):'—'}</td><td data-l="Line value">${x.avg_line_value!=null?(x.avg_line_value>0?'+':'')+x.avg_line_value:'—'}</td></tr>`;
  el('picktbl').innerHTML = R.length ? `<caption style="text-align:left;caption-side:top;font:13px var(--body);color:var(--muted);padding-bottom:6px">${ALL.length} ${opts.who} tracked · graded records first, best units on top · picks are graded once their games are final${ALL.length>R.length?` · top ${R.length} shown`:''}</caption><thead><tr><th style="text-align:left">Picker</th><th style="text-align:left">Outlet</th><th>Graded</th><th>W–L vs close</th><th>Units</th><th title="Points the market moved toward the pick after it was posted">Line value</th></tr></thead><tbody>`+R.map(row).join('')+'</tbody>'
    : `<tbody><tr><td style="text-align:left;font-family:var(--body)">No ${opts.who} with counted picks yet.</td></tr></tbody>`;
  el('picktbl').onclick = e=>{ const b=e.target.closest('.pkbtn'); if(b) pickerDialog(K, b.dataset.picker); };

  function pickerDialog(K, name){
    let d=el('pkdlg');
    if(!d){ d=document.createElement('dialog'); d.id='pkdlg'; d.className='pkdlg'; document.body.appendChild(d);
      d.addEventListener('click', e=>{ if(e.target===d || e.target.closest('.pkx')) d.close(); }); }
    const P=(K.picks_by_picker||{})[name]||{pending:[],graded:[],not_counted:[]}, x=(K.pickers||[]).find(p=>p.picker===name)||{};
    const sg=v=>v==null?'—':(v>0?'+':'')+v;
    const dt=s=>s?new Date(s).toLocaleString('en-US',{timeZone:'America/Phoenix',weekday:'short',month:'short',day:'numeric'}):'—';
    const game=p=>`${E(p.away||'')} @ ${E(p.home||'')}`;
    const pick=p=>`<b>${E(p.pick||'')}</b>${p.market==='moneyline'?' ML':''}${p.line!=null&&p.market!=='moneyline'?` ${sg(p.line)}`:p.line!=null?` (${sg(p.line)})`:''}`;
    const src=p=>p.link?` <a href="${E(p.link)}" target="_blank" rel="noopener noreferrer">source</a>`:'';
    // why they picked it: AI summary (kept only if its words are in the article) + their own words; shown in full
    const whyRow=(p,n)=>{ const r=(p.reason||'').trim(), q0=(p.quote||'').trim(), q=q0.split(/\s+/).length>=4?q0:'';
      if(!r && !q) return '';
      const same=r && q && r.toLowerCase().replace(/\W/g,'')===q.toLowerCase().replace(/\W/g,'');
      return `<tr class="pkwhy"><td colspan="${n}"><b>Why:</b> ${r?E(r):''}${q&&!same?`${r?' ':''}<i>“${E(q.replace(/^["“”]+|["“”]+$/g,''))}”</i>`:''}</td></tr>`; };
    const started=p=>p.kick && Date.parse(p.kick)<Date.now();
    const pend=P.pending.length?`<table><thead><tr><th>Game day</th><th>Game</th><th>Pick</th></tr></thead><tbody>${P.pending.map(p=>`<tr${whyRow(p,3)?' class="hw"':''}><td>${dt(p.kick||p.seen_at)}${started(p)?'<br><small>awaiting final score</small>':''}</td><td>${game(p)}</td><td>${pick(p)}${src(p)}</td></tr>${whyRow(p,3)}`).join('')}</tbody></table>`:'<p class="q">No pending picks right now.</p>';
    const grd=P.graded.length?`<table><thead><tr><th>Game day</th><th>Game</th><th>Pick</th><th title="Closing line the pick is graded against">Close</th><th>Final</th><th>Result</th><th class="n">Units</th></tr></thead><tbody>${P.graded.map(p=>`<tr${whyRow(p,7)?' class="hw"':''}><td>${dt(p.kick)}</td><td>${game(p)}</td><td>${pick(p)}${src(p)}</td><td class="n" data-l="close">${sg(p.close_line)}</td><td class="n" data-l="final">${E(p.final||'')}</td><td class="${p.result==='win'?'win':p.result==='loss'?'loss':''}">${E(p.result||'')}</td><td class="n" data-l="units">${p.units==null?'—':sg(Number(p.units).toFixed(2))}</td></tr>${whyRow(p,7)}`).join('')}</tbody></table>`:'<p class="q">Nothing graded yet.</p>';
    const NC=P.not_counted||[];
    const nc=NC.length?`<p class="q">Logged from their articles but left out of the record by the checks, with the reason.</p><table><thead><tr><th>Game day</th><th>Game</th><th>Pick</th><th>Why not counted</th></tr></thead><tbody>${NC.map(p=>`<tr><td>${dt(p.kick||p.seen_at)}</td><td>${game(p)}</td><td>${pick(p)}${src(p)}</td><td class="q" data-l="why">${E(p.why||'')}</td></tr>`).join('')}</tbody></table>`:'';
    d.innerHTML=`<div class="pkh"><div><h3>${E(name)}</h3><small>${outlet(x)} · ${x.graded?`${wl(x.wins,x.losses,x.pushes)} vs the close · ${x.units>0?'+':''}${(x.units||0).toFixed(2)} units`:'no graded picks yet'}${x.pending?` · ${x.pending} pending`:''}</small></div><button class="pkx" aria-label="Close">×</button></div>
      <div class="pkb"><span class="subh">Pending (${x.pending||0})</span>${pend}<span class="subh">Graded (${x.graded||0}${(x.graded||0)>P.graded.length?`, latest ${P.graded.length} shown`:''})</span>${grd}${NC.length?`<span class="subh">Not counted (${x.not_counted||NC.length}${(x.not_counted||0)>NC.length?`, latest ${NC.length} shown`:''})</span>${nc}`:''}</div>`;
    d.showModal(); d.scrollTop=0;
  }
}
