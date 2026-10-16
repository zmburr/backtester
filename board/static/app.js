'use strict';
// Setup Board: renders /api/snapshot (written by board/refresh.py) and polls it.
// Phone: one pane at a time (list, then a detail pane with Back).
// Desktop: list sidebar + one reading column. Keys: ↑/↓ names, ←/→ or 1-4 tabs,
// Enter opens, Esc backs out.

const S = {
  tab: 'bounce', snap: null, sel: { bounce: null, reversal: null, history: null },
  showAll: { bounce: false, reversal: false }, lessons: {}, lessonSide: 'bounce',
  history: null, shots: { date: null, entries: [] }, chart: null, chartTf: 'intraday', bars: {},
};
const TABS = ['bounce', 'reversal', 'playbook', 'history'];
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x, d = 1) => x == null ? '–' : `${x >= 0 ? '+' : '−'}${Math.abs(x * 100).toFixed(d)}%`;
const upct = (x, d = 0) => x == null ? '–' : `${(x * 100).toFixed(d)}%`;
const px = x => x == null ? '–' : Number(x).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const wide = () => matchMedia('(min-width: 900px)').matches;
const recCls = r => r === 'GO' ? 'go' : r === 'CAUTION' ? 'caution' : 'nogo';
const isActive = r => r.rec === 'GO' || r.rec === 'CAUTION';

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

function dayLabel(iso) {
  const d = new Date(`${iso}T12:00:00`);
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'numeric', day: 'numeric' });
}
function ago(iso) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return 'just now';
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  return `${Math.round(s / 3600)} h ago`;
}
function activeNow() {   // ET weekday 04:00-20:00, when a refresh should be running
  const et = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const h = et.getHours() + et.getMinutes() / 60;
  return et.getDay() >= 1 && et.getDay() <= 5 && h >= 4 && h < 20;
}

// ── status + tabs ────────────────────────────────────────────────────────
function renderStatus() {
  const s = S.snap, el = $('#status');
  if (!s || s.empty) { el.textContent = 'No snapshot yet. The first refresh runs at 4:00 AM or 4:30 PM ET.'; return; }
  const word = { premarket: 'Premarket', market: 'Market hours', evening: 'Evening' }[s.mode] || s.mode;
  const stale = Date.now() - new Date(s.generated_at).getTime() > 30 * 60e3 && activeNow();
  el.innerHTML = `<b>${esc(word)}</b> · plan for <b>${esc(dayLabel(s.session))}</b> · data ${esc(s.data_time || '–')} ET, ${esc(s.source)} · <span class="${stale ? 'warn' : ''}">updated ${esc(ago(s.generated_at))}</span>`;
  for (const side of ['bounce', 'reversal']) {
    const c = s[side]?.counts || {};
    $(`#n-${side}`).textContent = c.GO || c.CAUTION ? `${c.GO || 0}·${c.CAUTION || 0}` : '';
  }
}

function setTab(tab) {
  if (tab === 'playbook' && (S.tab === 'bounce' || S.tab === 'reversal')) S.lessonSide = S.tab;
  S.tab = tab;
  document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === tab));
  document.body.classList.toggle('wide-only', tab === 'playbook');
  document.body.classList.toggle('phone-detail', tab === 'playbook');
  render();
}

function render() {
  renderStatus();
  if (S.tab === 'bounce' || S.tab === 'reversal') {
    const rows = S.snap?.[S.tab]?.rows || [];
    if (wide() && !rows.some(r => r.ticker === S.sel[S.tab])) S.sel[S.tab] = (rows.find(isActive) || rows[0])?.ticker || null;
    renderSideList(S.tab);
    if (wide() || document.body.classList.contains('phone-detail')) renderDetail(S.tab);
  } else if (S.tab === 'playbook') {
    renderPlaybook();
  } else {
    renderHistory();
  }
}

// ── shots (one-tap counter; rules: 3 shots per idea, 2 stops per name) ─────
function shotStats(side) {
  const es = S.shots.entries.filter(e => e.side === side);
  const stops = {};
  es.filter(e => e.kind === 'stop').forEach(e => { stops[e.ticker] = (stops[e.ticker] || 0) + 1; });
  return { shots: es.filter(e => e.kind === 'shot').length, stops };
}
const locked = (side, tk) => (shotStats(side).stops[tk] || 0) >= 2;

async function loadShots() {
  try { S.shots = await api('/api/shots'); } catch (e) { /* keep the last copy */ }
}
async function postShot(body) {
  S.shots = await api('/api/shots', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  render();
}
function tallyHtml(side) {
  const st = shotStats(side);
  const stops = Object.entries(st.stops).map(([t, n]) => `${t} ${n}/2${n >= 2 ? ' LOCKED' : ''}`).join(' · ');
  return `<div class="tally">Today: <b>${st.shots}/3 shots</b> on the ${side} idea${stops ? ` · stop-outs: ${esc(stops)}` : ''}</div>` +
    (st.shots >= 3 ? '<div class="warn">Three shots used: done with this idea today.</div>' : '');
}

// ── bounce / reversal list ───────────────────────────────────────────────
function spark(a, b) {
  const n = a.length, max = Math.max(1, ...a, ...b);
  const pts = arr => arr.map((v, i) => `${(i / (n - 1) * 100).toFixed(1)},${(36 - v / max * 32).toFixed(1)}`).join(' ');
  return `<svg class="spark" viewBox="0 0 100 38" preserveAspectRatio="none" aria-hidden="true">` +
    `<polyline fill="none" stroke="var(--caution)" stroke-width="1.5" vector-effect="non-scaling-stroke" points="${pts(b)}"/>` +
    `<polyline fill="none" stroke="var(--go)" stroke-width="2" vector-effect="non-scaling-stroke" points="${pts(a)}"/></svg>`;
}
function breadthBlock(side) {
  const b = S.snap.breadth || [], c = S.snap[side].counts;
  const go = b.map(e => e[`${side}_go`] || 0), ca = b.map(e => e[`${side}_caution`] || 0);
  let line = `<b class="go">${c.GO} GO</b> · <b class="caution">${c.CAUTION} CAUTION</b>`;
  if (b.length > 1) {
    const runDay = (S.snap.generated_at || '').slice(0, 10);
    const since = b[0].date && b[0].date !== runDay ? `${b[0].t} last evening` : b[0].t;
    line += ` <span class="dim">(GO ${go[0]} → ${go[go.length - 1]} since ${esc(since)})</span>`;
  }
  return `<div class="block"><h4>Breadth · watchlist</h4>${line}${b.length > 1 ? spark(go, ca) : ''}</div>`;
}
function reportBlock(side) {
  const rep = (S.snap.reports || [])[0];
  if (!rep) return '';
  const [hh, mm] = (rep.time || '0:00').split(':').map(Number);
  const clock = `${((hh + 11) % 12) + 1}:${String(mm).padStart(2, '0')} ${hh < 12 ? 'AM' : 'PM'}`;
  const v = rep[side], when = `${clock} report, ${dayLabel(rep.date)}`;
  const parts = [];
  if (v.GO.length) parts.push(`<b class="go">GO</b> ${esc(v.GO.join(' '))}`);
  if (v.CAUTION.length) parts.push(`<b class="caution">CAUTION</b> ${esc(v.CAUTION.join(' '))}`);
  return `<div class="block"><h4>Priority report</h4><span class="dim">${esc(when)}:</span> ${parts.length ? parts.join(' · ') : `<span class="dim">no ${side} windows</span>`}</div>`;
}
function scenarioBadges(r) {
  if (!r.scenario) return '';
  let h = `<span class="badge b-${r.scenario}">${r.scenario}</span>`;
  if (r.scenario_now && r.scenario_now !== r.scenario) h += `<span class="dim">→</span><span class="badge b-${r.scenario_now}">${r.scenario_now}</span>`;
  return h;
}
function rowHtml(side, r) {
  const mode = S.snap.mode, move = `${pct(r.gap)} ${mode === 'evening' ? 'after hours' : 'vs close'}`;
  const l2 = side === 'bounce'
    ? [move, r.go_gap != null ? `GO ≤ ${px(r.go_price)}` : null,
       r.pm_low != null && mode !== 'evening' ? `PM low ${px(r.pm_low)} @ ${r.pm_low_time}` : null,
       r.down_days != null ? `down ${r.down_days}d` : null]
    : [move, r.pct_from_9ema != null ? `${pct(r.pct_from_9ema, 0)} vs 9EMA` : null, `+4% at ${px(r.gap_level)}`];
  return `<button class="row${S.sel[side] === r.ticker ? ' sel' : ''}" data-side="${side}" data-tk="${esc(r.ticker)}">` +
    `<div class="l1"><span class="tk">${esc(r.ticker)}</span><b class="${recCls(r.rec)}">${r.score}/${r.max_score} ${esc(r.rec)}</b>` +
    `${scenarioBadges(r)}${locked(side, r.ticker) ? '<span class="badge b-lock">LOCKED</span>' : ''}<span class="dim">${esc(r.cap || '')}</span></div>` +
    `<div class="l2">${l2.filter(Boolean).map(esc).join(' · ')}</div></button>`;
}
function renderSideList(side) {
  const list = $('#list');
  if (!S.snap) { list.innerHTML = '<div class="empty">Loading…</div>'; return; }
  if (S.snap.empty) { list.innerHTML = '<div class="empty">No snapshot yet.</div>'; return; }
  const rows = S.snap[side].rows || [], active = rows.filter(isActive), rest = rows.filter(r => !isActive(r));
  let h = breadthBlock(side) + reportBlock(side);
  if (!active.length) h += `<div class="empty">No GO or CAUTION ${side} names right now.</div>`;
  h += active.map(r => rowHtml(side, r)).join('');
  if (rest.length) {
    h += S.showAll[side]
      ? rest.map(r => rowHtml(side, r)).join('') + `<button class="more" data-more="${side}">Hide the ${rest.length} NO-GO names</button>`
      : `<button class="more" data-more="${side}">${rest.length} more names NO-GO ▸</button>`;
  }
  list.innerHTML = h;
  list.querySelector('.row.sel')?.scrollIntoView({ block: 'nearest' });
}

// ── name detail ──────────────────────────────────────────────────────────
function kvHtml(pairs) {
  return `<div class="grid">${pairs.filter(([, v]) => v != null && v !== '–' && v !== '').map(([k, v]) =>
    `<div class="kv"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join('')}</div>`;
}
function scenarioLine(side, r) {
  const txt = S.snap.scenario_text[side], mode = S.snap.mode;
  if (!r.scenario) return '';
  if (mode === 'market') {
    return `Opened <b>${r.scenario}</b>: ${esc(txt[r.scenario])}` +
      (r.scenario_now && r.scenario_now !== r.scenario ? ` · now <b>${r.scenario_now}</b>: ${esc(txt[r.scenario_now])}` : '');
  }
  if (mode === 'evening') return `If it opened at tonight's after-hours price: <b>${r.scenario}</b>, ${esc(txt[r.scenario])}`;
  return `<b>${r.scenario}</b> in play: ${esc(txt[r.scenario])}`;
}
function renderDetail(side) {
  const reader = $('#reader');
  const r = (S.snap?.[side]?.rows || []).find(x => x.ticker === S.sel[side]);
  if (!r) { reader.innerHTML = '<div class="col"><div class="empty">Pick a name from the list.</div></div>'; return; }
  const goLevel = r.go_gap == null ? 'not within −20%' : r.go_gap >= 0 ? 'GO at a flat open' : `≤ ${px(r.go_price)} (${pct(r.go_gap)})`;
  const pairs = side === 'bounce' ? [
    ['Price', px(r.price)], ['vs prior close', pct(r.gap)], ['GO level', goLevel], ['Prior close', px(r.prior_close)],
    ['PM low', r.pm_low != null ? `${px(r.pm_low)} @ ${r.pm_low_time}` : null],
    ['Open', r.open != null ? `${px(r.open)} (${pct(r.open_gap)})` : null], ['Day low', r.day_low != null ? px(r.day_low) : null],
    ['AH low', r.ah_low != null ? px(r.ah_low) : null],
    ['Mid-BB (swing exit)', r.mid_bb != null ? `${px(r.mid_bb)} (${pct(r.to_mid_bb, 0)})` : null],
    ['ATR', r.atr != null ? `${px(r.atr)} (${upct(r.atr_pct)})` : null], ['Down days', r.down_days != null ? String(r.down_days) : null],
    ['Off 30d high', pct(r.off_30d, 0)], ['Selloff', pct(r.selloff, 0)], ['Setup', r.setup],
    ['Odds of a 5%+ bounce', r.odds?.p_bounce5 != null ? upct(r.odds.p_bounce5) : null],
  ] : [
    ['Price', px(r.price)], ['vs prior close', pct(r.gap)], ['+4% gap level', px(r.gap_level)], ['Prior close', px(r.prior_close)],
    ['Above 9EMA', pct(r.pct_from_9ema, 0)], ['9EMA', px(r.ema9)], ['3-day', pct(r.chg_3d, 0)], ['ATR', upct(r.atr_pct)],
    ['Prior-day RVOL', r.rvol != null ? `${Number(r.rvol).toFixed(2)}x` : null], ['Setup', r.setup || null],
    ['Odds of a 5%+ fade', r.odds?.p_fade5 != null ? upct(r.odds.p_fade5) : null],
  ];
  const misses = [...(r.failing || []).map(f => `${f.name}: now ${f.actual}${f.threshold ? ` (needs ${f.threshold})` : ''}`),
    ...(r.gates_failed || []).map(g => `failed the ${g}`)];
  const notes = [...(r.warnings || []).map(w => `⚠ ${w}`), ...(r.bonuses || []).map(b => `+ ${b}`)];
  const comps = (r.analogs?.comps || []).slice(0, 3).map(c =>
    `${c.ticker} ${c.date}: ${side === 'bounce' ? `high ${pct(c.open_high_pct, 0)}, ` : `low ${pct(c.open_low_pct, 0)}, `}close ${pct(c.open_close_pct, 0)}`);
  reader.innerHTML = `<div class="col">
    <button class="back" data-back="1">‹ ${side === 'bounce' ? 'Bounce' : 'Reversal'} list</button>
    <h2>${esc(r.ticker)} <b class="${recCls(r.rec)}">${r.score}/${r.max_score} ${esc(r.rec)}</b> ${scenarioBadges(r)}</h2>
    <div class="sub">${esc(r.cap || '')}${r.report_rec ? ` · priority report: ${esc(r.report_rec)}` : ''}${r.bar_time ? ` · last bar ${esc(r.bar_time)}` : ''}</div>
    ${r.scenario ? `<div class="scen">${scenarioLine(side, r)}</div>` : ''}
    ${r.principal ? `<div class="scen">Leveraged: check the principal (${esc(r.principal)}).</div>` : ''}
    ${kvHtml(pairs)}
    <div class="chart-tabs"><button data-tf="intraday" class="${S.chartTf === 'intraday' ? 'on' : ''}">1-min</button><button data-tf="day" class="${S.chartTf === 'day' ? 'on' : ''}">Daily</button></div>
    <div class="chart" id="chart"></div>
    <h3>Shots</h3>
    <div class="shots"><button class="btn shot" data-shot="shot" data-tk="${esc(r.ticker)}">Took a shot</button>
      <button class="btn stop" data-shot="stop" data-tk="${esc(r.ticker)}">Stopped out</button>
      <button class="btn" data-undo="1">Undo</button></div>
    ${tallyHtml(side)}${locked(side, r.ticker) ? `<div class="warn">${esc(r.ticker)} is locked: two stop-outs today.</div>` : ''}
    ${misses.length ? `<h3>What's missing</h3><ul class="items">${misses.map(m => `<li>${esc(m)}</li>`).join('')}</ul>` : ''}
    ${notes.length ? `<h3>Flags</h3><ul class="items">${notes.map(m => `<li>${esc(m)}</li>`).join('')}</ul>` : ''}
    ${comps.length ? `<h3>Closest past setups</h3><ul class="items">${comps.map(m => `<li>${esc(m)}</li>`).join('')}</ul>` : ''}
  </div>`;
  drawChart(side, r);
}

async function drawChart(side, r) {
  const tk = r.ticker;
  let data = S.bars[tk];
  if (!data || Date.now() - data._t > 300e3) {
    try { data = await api(`/api/bars?ticker=${encodeURIComponent(tk)}`); } catch (e) { return; }
    data._t = Date.now(); S.bars[tk] = data;
  }
  const el = $('#chart');
  if (!el || S.sel[side] !== tk) return;
  if (S.chart) { S.chart.remove(); S.chart = null; }
  const daily = S.chartTf === 'day';
  const bars = daily ? data.daily : data.intraday;
  if (!bars?.length) { el.innerHTML = '<div class="empty">No bars.</div>'; return; }
  const c = LightweightCharts.createChart(el, {
    autoSize: true, layout: { background: { color: '#111418' }, textColor: '#8a94a3' },
    grid: { vertLines: { color: '#1c222a' }, horzLines: { color: '#1c222a' } },
    rightPriceScale: { borderColor: '#2a313b' },
    // 960 one-minute bars (4 AM-8 PM) need to fit a phone-width chart
    timeScale: { borderColor: '#2a313b', timeVisible: !daily, secondsVisible: false, rightOffset: 3, minBarSpacing: 0.05 },
  });
  const candles = c.addCandlestickSeries({ upColor: '#26a69a', downColor: '#ef5350', wickUpColor: '#26a69a', wickDownColor: '#ef5350', borderVisible: false });
  const vol = c.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol', color: '#2a3440' });
  c.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
  candles.setData(bars.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
  vol.setData(bars.map(b => ({ time: b.time, value: b.value })));
  const line = (price, color, title) => { if (price != null) candles.createPriceLine({ price, color, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title }); };
  const goAtClose = side === 'bounce' && r.go_price != null && Math.abs(r.go_price - r.prior_close) < 1e-6;
  line(r.prior_close, goAtClose ? '#3fb950' : '#8a94a3', goAtClose ? 'prior close = GO' : 'prior close');
  if (side === 'bounce') {
    if (!goAtClose) line(r.go_price, '#3fb950', 'GO');
    line(r.mid_bb, '#4f8cff', 'mid-BB');
    if (!daily) line(r.pm_low, '#d29922', 'PM low');
  } else {
    line(r.gap_level, '#ef5350', '+4%');
    line(r.ema9, '#4f8cff', '9EMA');
  }
  // autoSize gives the chart its width a frame later; fit once it has one
  const fit = () => daily ? c.timeScale().setVisibleLogicalRange({ from: bars.length - 90, to: bars.length + 2 })
    : c.timeScale().fitContent();
  fit();
  requestAnimationFrame(() => requestAnimationFrame(fit));
  S.chart = c;
}

// ── playbook ─────────────────────────────────────────────────────────────
async function renderPlaybook() {
  const side = S.lessonSide;
  const reader = $('#reader');
  if (!S.lessons[side]) {
    reader.innerHTML = '<div class="col"><div class="empty">Loading your lessons…</div></div>';
    try { S.lessons[side] = await api(`/api/lessons?side=${side}`); } catch (e) {
      reader.innerHTML = `<div class="col"><p class="warn">Couldn't load the lessons: ${esc(e.message)}</p></div>`; return;
    }
    if (S.tab !== 'playbook' || S.lessonSide !== side) return;
  }
  const L = S.lessons[side];
  let h = `<div class="col"><div class="toggle">` +
    `<button class="btn ${side === 'bounce' ? 'on' : ''}" data-lside="bounce">Bounce</button>` +
    `<button class="btn ${side === 'reversal' ? 'on' : ''}" data-lside="reversal">Reversal</button></div>`;
  h += `<div class="scen">${tallyHtml(side)}<div class="shots"><button class="btn" data-undo="1">Undo last</button></div></div>`;
  if (!L.found) h += `<p class="warn">Lessons note not found (${esc(L.note)}).</p>`;
  for (const sec of L.sections) {
    const items = sec.heading.toLowerCase().startsWith('risk budget') && L.tiers
      ? [`<i>${esc(L.tiers)}</i>`, ...sec.items] : sec.items;
    h += `<h3>${esc(sec.heading)}</h3><ul class="items">${items.map(i => `<li>${i}</li>`).join('')}</ul>`;
  }
  h += `<p class="dim">From your Obsidian note ${esc(L.note)}. Edit it there; this page reads it fresh.</p></div>`;
  reader.innerHTML = h;
}

// ── history ──────────────────────────────────────────────────────────────
function epSummary(ep) {
  const go = ep.names.filter(n => n.rec === 'GO' && n.d0_pct != null);
  if (!go.length) return 'no outcome yet';
  const sign = ep.side === 'bounce' ? 1 : -1;          // a short's best name fell the most
  const best = go.reduce((a, b) => (sign * b.d0_pct > sign * a.d0_pct ? b : a));
  const worked = go.filter(n => sign * n.d0_pct > 0).length;
  return `${worked}/${go.length} GO names moved the right way open→close · best ${best.ticker} ${pct(best.d0_pct)}`;
}
async function renderHistory() {
  const list = $('#list');
  if (!S.history) {
    list.innerHTML = '<div class="empty">Loading…</div>';
    try { S.history = (await api('/api/history')).episodes; } catch (e) {
      list.innerHTML = `<div class="empty warn">${esc(e.message)}</div>`; return;
    }
    if (S.tab !== 'history') return;
  }
  if (S.sel.history == null && wide() && S.history.length) S.sel.history = 0;
  list.innerHTML = `<div class="block"><h4>Setup days</h4><span class="dim">Days with 2+ GO on one side in the priority report</span></div>` +
    (S.history.length ? S.history.map((ep, i) =>
      `<button class="row${S.sel.history === i ? ' sel' : ''}" data-ep="${i}"><div class="l1"><span class="tk">${esc(dayLabel(ep.target))}</span>` +
      `<span class="badge ${ep.side === 'bounce' ? 'b-A' : 'b-lock'}">${ep.side}</span><span class="dim">${ep.n_go} GO · ${esc(ep.names.slice(0, 5).map(n => n.ticker).join(' '))}</span></div>` +
      `<div class="l2">${esc(epSummary(ep))}</div></button>`).join('') : '<div class="empty">No setup days yet.</div>');
  if (wide() || document.body.classList.contains('phone-detail')) renderEpisode();
}
function renderEpisode() {
  const ep = S.history?.[S.sel.history];
  const reader = $('#reader');
  if (!ep) { reader.innerHTML = '<div class="col"><div class="empty">Pick a day.</div></div>'; return; }
  const rows = ep.names.map(n => {
    const bits = [n.d0_pct != null ? `open→close ${pct(n.d0_pct)}` : 'no outcome yet',
      n.mfe_atr_d0 != null ? `best move ${Number(n.mfe_atr_d0).toFixed(1)} ATR` : null,
      n.exit_r != null ? `rules ${n.exit_r >= 0 ? '+' : '−'}${Math.abs(Number(n.exit_r)).toFixed(1)}R` : null,
      n.d1_pct != null ? `next day ${pct(n.d1_pct)}` : null].filter(Boolean);
    return `<div class="nm"><b>${esc(n.ticker)}</b> <b class="${recCls(n.rec)}">${esc(n.score || '')} ${n.rec}</b> <span class="dim">${esc(n.cap || '')}</span><br>${esc(bits.join(' · '))}</div>`;
  }).join('');
  reader.innerHTML = `<div class="col ep"><button class="back" data-back="1">‹ Setup days</button>
    <h2>${esc(dayLabel(ep.target))} · ${esc(ep.side)}</h2>
    <div class="sub">From ${esc(ep.reports.join(', '))}. "Rules R" = the live entry + trail rules replayed on that day.</div>${rows}</div>`;
}

// ── events ───────────────────────────────────────────────────────────────
function openDetail() { document.body.classList.add('phone-detail'); render(); $('#reader').scrollTop = 0; }
function back() { if (S.tab !== 'playbook') { document.body.classList.remove('phone-detail'); render(); } }

document.addEventListener('click', e => {
  const t = e.target.closest('button');
  if (!t) return;
  if (t.dataset.tab) return setTab(t.dataset.tab);
  if (t.dataset.tk && t.dataset.side) { S.sel[t.dataset.side] = t.dataset.tk; return wide() ? render() : openDetail(); }
  if (t.dataset.ep != null) { S.sel.history = Number(t.dataset.ep); return wide() ? render() : openDetail(); }
  if (t.dataset.more) { S.showAll[t.dataset.more] = !S.showAll[t.dataset.more]; return render(); }
  if (t.dataset.back) return back();
  if (t.dataset.tf) { S.chartTf = t.dataset.tf; return render(); }
  if (t.dataset.lside) { S.lessonSide = t.dataset.lside; return render(); }
  if (t.dataset.shot) return postShot({ side: S.tab === 'playbook' ? S.lessonSide : S.tab, ticker: t.dataset.tk, kind: t.dataset.shot });
  if (t.dataset.undo) return postShot({ kind: 'undo' });
});

document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const i = TABS.indexOf(S.tab);
  if (/^[1-4]$/.test(e.key)) return setTab(TABS[Number(e.key) - 1]);
  if (e.key === 'ArrowRight') { e.preventDefault(); return setTab(TABS[(i + 1) % TABS.length]); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); return setTab(TABS[(i + TABS.length - 1) % TABS.length]); }
  if (e.key === 'Escape') return back();
  if (e.key === 'Enter' && !wide()) return openDetail();
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    const step = e.key === 'ArrowDown' ? 1 : -1;
    if (S.tab === 'history' && S.history?.length) {
      S.sel.history = Math.max(0, Math.min(S.history.length - 1, (S.sel.history ?? -1) + step));
      return render();
    }
    if (S.tab === 'bounce' || S.tab === 'reversal') {
      const rows = (S.snap?.[S.tab]?.rows || []).filter(r => isActive(r) || S.showAll[S.tab]);
      if (!rows.length) return;
      const at = rows.findIndex(r => r.ticker === S.sel[S.tab]);
      S.sel[S.tab] = rows[Math.max(0, Math.min(rows.length - 1, at + step))].ticker;
      render();
    }
  }
});

let lastWide = wide();
addEventListener('resize', () => { if (wide() !== lastWide) { lastWide = wide(); render(); } });

async function poll() {
  try {
    const snap = await api('/api/snapshot');
    await loadShots();
    const changed = !S.snap || snap.generated_at !== S.snap.generated_at;
    S.snap = snap;
    if (changed) render(); else renderStatus();
  } catch (e) {
    $('#status').innerHTML = `<span class="warn">Can't reach the board: ${esc(e.message)}</span>`;
  }
}

setTab('bounce');
poll();
setInterval(poll, 60e3);
setInterval(renderStatus, 30e3);
