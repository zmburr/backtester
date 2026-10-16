'use strict';
// Method: renders /api/snapshot (written by method/refresh.py) and polls it.
// Routes: #/bounce[/TICKER]  #/reversal[/TICKER]  #/playbook  #/history[/N]
// Phone: list, then a detail pane (Back / swipe back). Desktop (≥1024px): rail +
// list + detail. Keys: ↑/↓ names, ←/→ or 1-4 sections, Esc back.

const S = {
  route: 'bounce', snap: null, sel: { bounce: null, reversal: null, history: null },
  showAll: { bounce: false, reversal: false }, lessons: {}, lessonSide: 'bounce',
  history: null, shots: { date: null, entries: [] }, chart: null, chartTf: 'intraday', bars: {}, reachable: true,
};
const ROUTES = ['bounce', 'reversal', 'playbook', 'history'];
const app = document.getElementById('app');
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x, d = 1) => x == null ? '–' : `${x >= 0 ? '+' : '−'}${Math.abs(x * 100).toFixed(d)}%`;
const upct = (x, d = 0) => x == null ? '–' : `${(x * 100).toFixed(d)}%`;
const px = x => x == null ? '–' : Number(x).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const rr = x => `${x >= 0 ? '+' : '−'}${Math.abs(Number(x)).toFixed(1)}R`;
const wide = () => matchMedia('(min-width: 1024px)').matches;
const recCls = r => r === 'GO' ? 'go' : r === 'CAUTION' ? 'caution' : 'nogo';
const isActive = r => r.rec === 'GO' || r.rec === 'CAUTION';
const MODE_WORD = { premarket: 'Premarket', market: 'Market hours', evening: 'Evening' };

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
function dayLabel(iso) {
  return new Date(`${iso}T12:00:00`).toLocaleDateString('en-US', { weekday: 'short', month: 'numeric', day: 'numeric' });
}
function clock12(hhmm) {
  const [h, m] = (hhmm || '0:00').split(':').map(Number);
  return `${((h + 11) % 12) + 1}:${String(m).padStart(2, '0')} ${h < 12 ? 'AM' : 'PM'}`;
}
function ago(iso) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return 'just now';
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  return `${Math.round(s / 3600)} h ago`;
}
function activeNow() {     // ET weekday 04:00-20:00, when refreshes should be landing
  const et = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const h = et.getHours() + et.getMinutes() / 60;
  return et.getDay() >= 1 && et.getDay() <= 5 && h >= 4 && h < 20;
}
const recBadge = r => `<span class="badge ${recCls(r.rec)}">${esc(r.score)}/${esc(r.max_score)} ${esc(r.rec)}</span>`;
function scenBadges(r) {
  if (!r.scenario) return '';
  let h = `<span class="badge ${r.scenario}">${r.scenario}</span>`;
  if (r.scenario_now && r.scenario_now !== r.scenario) h += `<span class="muted">→</span><span class="badge ${r.scenario_now}">${r.scenario_now}</span>`;
  return h;
}

// ── routing ──────────────────────────────────────────────────────────────
function parseHash() {
  const [, route, arg] = (location.hash || '#/bounce').split('/');
  return { route: ROUTES.includes(route) ? route : 'bounce', arg: arg ? decodeURIComponent(arg) : null };
}
function go(route, arg = null, replace = false) {
  const h = `#/${route}${arg != null ? `/${encodeURIComponent(arg)}` : ''}`;
  if (location.hash === h) return applyRoute();
  if (replace) { history.replaceState(null, '', h); applyRoute(); } else location.hash = h;
}
function applyRoute() {
  const { route, arg } = parseHash();
  if (route === 'playbook' && (S.route === 'bounce' || S.route === 'reversal')) S.lessonSide = S.route;
  S.route = route;
  if (route === 'bounce' || route === 'reversal') S.sel[route] = arg || (wide() ? S.sel[route] : null);
  if (route === 'history') S.sel.history = arg != null ? Number(arg) : (wide() ? S.sel.history : null);
  app.dataset.route = route;
  app.dataset.view = arg != null || route === 'playbook' ? 'detail' : 'list';
  $$('[data-nav]').forEach(a => a.classList.toggle('on', a.dataset.nav === route));
  render();
}
function back() { if (S.route !== 'playbook') go(S.route); }

// ── status ───────────────────────────────────────────────────────────────
function renderStatus() {
  const s = S.snap;
  let short = 'Connecting…', dot = '';
  if (!S.reachable) { short = 'Offline'; dot = 'off'; }
  else if (s && !s.empty) {
    const stale = Date.now() - new Date(s.generated_at).getTime() > 30 * 60e3 && activeNow();
    short = `${MODE_WORD[s.mode] || s.mode} · ${s.data_time || '–'}`;
    dot = stale ? 'warn' : 'ok';
  } else if (s?.empty) { short = 'No data yet'; dot = 'warn'; }
  $$('[data-status-short]').forEach(e => { e.textContent = short; });
  $$('[data-dot]').forEach(e => { e.className = `dot ${dot}`; });
  for (const side of ['bounce', 'reversal']) {
    const n = s?.[side]?.counts?.GO || 0;
    $$(`[data-count="${side}"]`).forEach(e => { e.hidden = !n; e.textContent = n; });
  }
}

function render() {
  renderStatus();
  const r = S.route;
  if (r === 'bounce' || r === 'reversal') {
    const rows = S.snap?.[r]?.rows || [];
    if (wide() && !rows.some(x => x.ticker === S.sel[r])) S.sel[r] = (rows.find(isActive) || rows[0])?.ticker || null;
    renderSideList(r);
    renderDetail(r);
  } else if (r === 'playbook') {
    renderPlaybook();
  } else {
    renderHistory();
  }
}

// ── shots: 3 per idea, 2 stop-outs lock a name ──────────────────────────
function shotStats(side) {
  const es = S.shots.entries.filter(e => e.side === side);
  const stops = {};
  es.filter(e => e.kind === 'stop').forEach(e => { stops[e.ticker] = (stops[e.ticker] || 0) + 1; });
  return { shots: es.filter(e => e.kind === 'shot').length, stops };
}
const locked = (side, tk) => (shotStats(side).stops[tk] || 0) >= 2;
async function loadShots() { try { S.shots = await api('/api/shots'); } catch (e) { /* keep last copy */ } }
async function postShot(body) {
  S.shots = await api('/api/shots', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  render();
}
function shotsCard(side, tk = null) {
  const st = shotStats(side);
  const stops = Object.entries(st.stops).map(([t, n]) => `${t} ${n}/2`).join(' · ');
  const btns = tk
    ? `<button class="btn primary small" data-shot="shot" data-tk="${esc(tk)}">Took a shot</button>
       <button class="btn danger small" data-shot="stop" data-tk="${esc(tk)}">Stopped out</button>`
    : '';
  return `<div class="card"><div class="eyebrow">Shots today · ${side} idea</div>
    <div class="tally" style="margin-top:8px"><b class="num">${st.shots}/3</b><span class="muted">${stops ? `stop-outs: ${esc(stops)}` : 'no stop-outs'}</span></div>
    ${st.shots >= 3 ? '<div class="warn-text">Three shots used. Done with this idea today.</div>' : ''}
    ${tk && locked(side, tk) ? `<div class="lock-text">${esc(tk)} is locked: two stop-outs today.</div>` : ''}
    <div class="shots-row">${btns}<button class="btn ghost small" data-undo="1">Undo last</button></div></div>`;
}

// ── bounce / reversal list ───────────────────────────────────────────────
function spark(a, b) {
  const n = a.length, max = Math.max(1, ...a, ...b);
  const pts = arr => arr.map((v, i) => `${(i / (n - 1) * 100).toFixed(1)},${(42 - v / max * 36).toFixed(1)}`).join(' ');
  return `<svg class="spark" viewBox="0 0 100 44" preserveAspectRatio="none" aria-hidden="true">` +
    `<polyline fill="none" stroke="var(--caution)" stroke-width="1.5" vector-effect="non-scaling-stroke" points="${pts(b)}"/>` +
    `<polyline fill="none" stroke="var(--go)" stroke-width="2.2" vector-effect="non-scaling-stroke" points="${pts(a)}"/></svg>`;
}
function listHead(eyebrow, title, sub) {
  return `<div class="list-head"><div class="eyebrow">${eyebrow}</div><h2>${title}</h2>${sub ? `<p>${sub}</p>` : ''}</div>`;
}
function renderSideList(side) {
  const list = $('#list'), s = S.snap;
  if (!s) { list.innerHTML = '<div class="empty">Loading…</div>'; return; }
  if (s.empty) { list.innerHTML = listHead('Method', 'No data yet', 'The first refresh lands at 4:00 AM or 4:30 PM ET.'); return; }
  const c = s[side].counts, rows = s[side].rows || [];
  const active = rows.filter(isActive), rest = rows.filter(r => !isActive(r));
  let h = listHead(`${esc(MODE_WORD[s.mode] || s.mode)} · plan for`, esc(dayLabel(s.session)),
    `Data ${esc(s.data_time || '–')} ET · ${esc(s.source)} · updated ${esc(ago(s.generated_at))}`);

  const b = s.breadth || [], go = b.map(e => e[`${side}_go`] || 0), ca = b.map(e => e[`${side}_caution`] || 0);
  const runDay = (s.generated_at || '').slice(0, 10);
  const since = b.length ? (b[0].date && b[0].date !== runDay ? `${b[0].t} last evening` : b[0].t) : '';
  h += `<div class="kpis"><div class="kpi"><div class="eyebrow">GO</div><b class="go num">${c.GO}</b>` +
    `<small>${b.length > 1 ? `${go[0]} → ${go[go.length - 1]} since ${esc(since)}` : 'watchlist'}</small></div>` +
    `<div class="kpi"><div class="eyebrow">Caution</div><b class="caution num">${c.CAUTION}</b><small>watchlist</small></div></div>`;
  if (b.length > 1) h += `<div class="card"><div class="eyebrow">Breadth since ${esc(since)}</div>${spark(go, ca)}</div>`;

  const rep = (s.reports || [])[0];
  if (rep) {
    const v = rep[side];
    const chips = [...v.GO.map(t => `<span class="badge go">${esc(t)}</span>`), ...v.CAUTION.map(t => `<span class="badge caution">${esc(t)}</span>`)];
    h += `<div class="card"><div class="eyebrow">Priority report · ${esc(clock12(rep.time))} ${esc(dayLabel(rep.date))}</div>` +
      (chips.length ? `<div class="tickers">${chips.join('')}</div>` : `<div class="muted" style="margin-top:6px;font-size:13px">No ${side} windows</div>`) + '</div>';
  }
  if (!active.length) h += `<div class="empty">No GO or CAUTION ${side} names right now.</div>`;
  h += active.map(r => nameCard(side, r)).join('');
  if (rest.length) {
    h += S.showAll[side]
      ? rest.map(r => nameCard(side, r)).join('') + `<button class="more" data-more="${side}">Hide the ${rest.length} NO-GO names</button>`
      : `<button class="more" data-more="${side}">${rest.length} more names NO-GO</button>`;
  }
  list.innerHTML = h;
  if (wide()) $('.name.sel', list)?.scrollIntoView({ block: 'nearest' });
}
function nameCard(side, r) {
  const mode = S.snap.mode, move = `${pct(r.gap)} ${mode === 'evening' ? 'after hours' : 'vs close'}`;
  const l2 = side === 'bounce'
    ? [move, r.go_gap != null ? `GO ≤ ${px(r.go_price)}` : null,
       r.pm_low != null ? `PM low ${px(r.pm_low)} @ ${r.pm_low_time}` : null,
       r.down_days != null ? `down ${r.down_days}d` : null]
    : [move, r.pct_from_9ema != null ? `${pct(r.pct_from_9ema, 0)} vs 9EMA` : null, `+4% at ${px(r.gap_level)}`];
  return `<div class="card name${wide() && S.sel[side] === r.ticker ? ' sel' : ''}" role="button" tabindex="0" data-side="${side}" data-tk="${esc(r.ticker)}">` +
    `<div class="l1"><span class="tk">${esc(r.ticker)}</span>${recBadge(r)}${scenBadges(r)}` +
    `${r.principal ? '<span class="badge lev">3x</span>' : ''}${locked(side, r.ticker) ? '<span class="badge lock">LOCKED</span>' : ''}` +
    `<span class="cap">${esc(r.cap || '')}</span></div><div class="l2 num">${l2.filter(Boolean).map(esc).join(' · ')}</div></div>`;
}

// ── name detail ──────────────────────────────────────────────────────────
function setTitle(html, sub) { $('#d-title').innerHTML = html; $('#d-sub').textContent = sub || ''; }
function kpis(pairs) {
  return `<div class="kpis dense">${pairs.filter(([, v]) => v != null && v !== '–' && v !== '').map(([k, v, cls]) =>
    `<div class="kpi"><div class="eyebrow">${esc(k)}</div><b class="num ${cls || ''}">${esc(v)}</b></div>`).join('')}</div>`;
}
function renderDetail(side) {
  const reader = $('#reader');
  const r = (S.snap?.[side]?.rows || []).find(x => x.ticker === S.sel[side]);
  if (!r) {
    setTitle(side === 'bounce' ? 'Bounce' : 'Reversal', '');
    reader.innerHTML = '<div class="col"><div class="empty">Pick a name from the list.</div></div>';
    return;
  }
  setTitle(`${esc(r.ticker)} ${recBadge(r)} ${scenBadges(r)}`,
    [r.cap, r.report_rec ? `priority report ${r.report_rec}` : null, r.bar_time ? `last bar ${r.bar_time}` : null].filter(Boolean).join(' · '));
  const txt = S.snap.scenario_text[side], mode = S.snap.mode;
  let scen = '';
  if (r.scenario) {
    const lead = mode === 'market' ? 'Opened' : mode === 'evening' ? "At tonight's after-hours price" : 'In play now';
    const now = r.scenario_now && r.scenario_now !== r.scenario ? `<div class="muted" style="margin-top:6px">Now <b>${r.scenario_now}</b>: ${esc(txt[r.scenario_now])}</div>` : '';
    scen = `<div class="card callout"><div class="eyebrow">Open scenario · ${esc(lead)}</div><div style="margin-top:8px;display:flex;align-items:center">` +
      `<span class="big">${r.scenario}</span><span>${esc(txt[r.scenario])}</span></div>${now}</div>`;
  }
  const lev = r.principal ? `<div class="card callout warn"><div class="eyebrow">Leveraged</div><div style="margin-top:6px">Check the principal: ${esc(r.principal)}.</div></div>` : '';
  const goLevel = r.go_gap == null ? 'not within −20%' : r.go_gap >= 0 ? 'flat open' : `≤ ${px(r.go_price)}`;
  const pairs = side === 'bounce' ? [
    ['GO level', goLevel, 'go'], ['Price', px(r.price)], ['vs prior close', pct(r.gap), r.gap < 0 ? 'neg' : 'go'],
    ['Prior close', px(r.prior_close)],
    ['PM low', r.pm_low != null ? `${px(r.pm_low)} @ ${r.pm_low_time}` : null], ['Open', r.open != null ? `${px(r.open)} (${pct(r.open_gap)})` : null],
    ['Day low', r.day_low != null ? px(r.day_low) : null], ['AH low', r.ah_low != null ? px(r.ah_low) : null],
    ['Mid-BB · swing exit', r.mid_bb != null ? `${px(r.mid_bb)} (${pct(r.to_mid_bb, 0)})` : null],
    ['ATR', r.atr != null ? `${px(r.atr)} · ${upct(r.atr_pct)}` : null], ['Down days', r.down_days != null ? String(r.down_days) : null],
    ['Off 30d high', pct(r.off_30d, 0)], ['Selloff', pct(r.selloff, 0)],
    ['5%+ bounce odds', r.odds?.p_bounce5 != null ? upct(r.odds.p_bounce5) : null], ['Setup', r.setup ? r.setup.replace('GapFade_', '').replace(/stock$/, ' stock') : null],
  ] : [
    ['+4% gap level', px(r.gap_level), 'neg'], ['Price', px(r.price)], ['vs prior close', pct(r.gap), r.gap > 0 ? 'go' : 'neg'],
    ['Prior close', px(r.prior_close)], ['Above 9EMA', pct(r.pct_from_9ema, 0)], ['9EMA', px(r.ema9)], ['3-day', pct(r.chg_3d, 0)],
    ['ATR', upct(r.atr_pct)], ['Prior-day RVOL', r.rvol != null ? `${Number(r.rvol).toFixed(2)}x` : null], ['Setup', r.setup || null],
    ['5%+ fade odds', r.odds?.p_fade5 != null ? upct(r.odds.p_fade5) : null],
  ];
  const misses = [...(r.failing || []).map(f => [f.name, `now ${f.actual}${f.threshold ? ` · needs ${f.threshold}` : ''}`]),
    ...(r.gates_failed || []).map(g => [`Failed the ${g}`, ''])];
  const flags = [...(r.warnings || []).map(w => ['Warning', w]), ...(r.bonuses || []).map(b => ['Bonus', b])];
  const comps = (r.analogs?.comps || []).slice(0, 3).map(c => [`${c.ticker} · ${c.date}`,
    `${side === 'bounce' ? `high ${pct(c.open_high_pct, 0)}` : `low ${pct(c.open_low_pct, 0)}`} · close ${pct(c.open_close_pct, 0)}`]);
  const lines = (title, rows) => rows.length ? `<div class="card"><div class="eyebrow">${title}</div><div class="lines">` +
    rows.map(([a, b2]) => `<div class="line"><span>${esc(a)}</span><span class="r muted">${esc(b2)}</span></div>`).join('') + '</div></div>' : '';
  reader.innerHTML = `<div class="col">${scen}${lev}${kpis(pairs)}
    <div class="card chart-card"><div class="card-head"><span class="eyebrow">Chart</span>
      <div class="seg small"><button data-tf="intraday" class="${S.chartTf === 'intraday' ? 'on' : ''}">1-min</button><button data-tf="day" class="${S.chartTf === 'day' ? 'on' : ''}">Daily</button></div></div>
      <div class="chart" id="chart"></div></div>
    ${shotsCard(side, r.ticker)}
    ${lines("What's missing", misses)}${lines('Flags', flags)}${lines('Closest past setups', comps)}</div>`;
  drawChart(side, r);
}

const CH = { text: '#6f7873', grid: '#151a1e', up: '#46b98a', down: '#d9615f' };
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
  const daily = S.chartTf === 'day', bars = daily ? data.daily : data.intraday;
  if (!bars?.length) { el.innerHTML = '<div class="empty">No bars.</div>'; return; }
  const c = LightweightCharts.createChart(el, {
    autoSize: true,
    layout: { background: { type: 'solid', color: 'transparent' }, textColor: CH.text, fontFamily: 'Inter, system-ui, sans-serif', fontSize: 11, attributionLogo: false },
    grid: { vertLines: { visible: false }, horzLines: { color: CH.grid } },
    rightPriceScale: { borderVisible: false, minimumWidth: 58 },
    // 960 one-minute bars (4 AM-8 PM) must fit a phone-width chart
    timeScale: { borderVisible: false, timeVisible: !daily, secondsVisible: false, rightOffset: 2, minBarSpacing: 0.05 },
    crosshair: { mode: 0, vertLine: { color: '#3a434b', labelBackgroundColor: '#1a1f24' }, horzLine: { color: '#3a434b', labelBackgroundColor: '#1a1f24' } },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
  });
  const candles = c.addCandlestickSeries({ upColor: CH.up, downColor: CH.down, wickUpColor: CH.up, wickDownColor: CH.down, borderVisible: false, priceLineVisible: false });
  const vol = c.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol', color: '#1c2227', lastValueVisible: false, priceLineVisible: false });
  c.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
  candles.setData(bars.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
  vol.setData(bars.map(b => ({ time: b.time, value: b.value })));
  const line = (price, color, title) => { if (price != null) candles.createPriceLine({ price, color, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title }); };
  const goAtClose = side === 'bounce' && r.go_price != null && Math.abs(r.go_price - r.prior_close) < 1e-6;
  line(r.prior_close, goAtClose ? '#3ddc97' : '#6f7873', goAtClose ? 'prior close = GO' : 'prior close');
  if (side === 'bounce') {
    if (!goAtClose) line(r.go_price, '#3ddc97', 'GO');
    line(r.mid_bb, '#a594ff', 'mid-BB');
    if (!daily) line(r.pm_low, '#ffb547', 'PM low');
  } else {
    line(r.gap_level, '#ff6363', '+4%');
    line(r.ema9, '#7cc4ff', '9EMA');
  }
  const fit = () => daily ? c.timeScale().setVisibleLogicalRange({ from: bars.length - 90, to: bars.length + 2 }) : c.timeScale().fitContent();
  fit();
  requestAnimationFrame(() => requestAnimationFrame(fit));     // autoSize sets the width a frame later
  S.chart = c;
}

// ── playbook ─────────────────────────────────────────────────────────────
async function renderPlaybook() {
  const side = S.lessonSide, reader = $('#reader');
  setTitle('Playbook', 'Your lessons, read live from Obsidian');
  if (!S.lessons[side]) {
    reader.innerHTML = '<div class="col"><div class="empty">Loading your lessons…</div></div>';
    try { S.lessons[side] = await api(`/api/lessons?side=${side}`); } catch (e) {
      reader.innerHTML = `<div class="col"><div class="empty neg">Couldn't load the lessons: ${esc(e.message)}</div></div>`; return;
    }
    if (S.route !== 'playbook' || S.lessonSide !== side) return;
  }
  const L = S.lessons[side];
  let h = `<div class="col"><div class="pb-top"><div class="seg"><button data-lside="bounce" class="${side === 'bounce' ? 'on' : ''}">Bounce</button>` +
    `<button data-lside="reversal" class="${side === 'reversal' ? 'on' : ''}">Reversal</button></div></div>${shotsCard(side)}`;
  if (!L.found) h += `<div class="card callout warn">Lessons note not found (${esc(L.note)}).</div>`;
  for (const sec of L.sections) {
    const risk = sec.heading.toLowerCase().startsWith('risk budget') && L.tiers;
    const items = (risk ? [`<li class="tiers">${esc(L.tiers)}</li>`] : []).concat(sec.items.map(i => `<li>${i}</li>`));
    h += `<div class="card"><h3>${esc(sec.heading)}</h3><ul class="items">${items.join('')}</ul></div>`;
  }
  h += `<p class="foot-note">From ${esc(L.note)}. Edit it in Obsidian; Method reads it fresh.</p></div>`;
  reader.innerHTML = h;
}

// ── history ──────────────────────────────────────────────────────────────
function epSummary(ep) {
  const go = ep.names.filter(n => n.rec === 'GO' && n.d0_pct != null);
  if (!go.length) return 'no outcome yet';
  const sign = ep.side === 'bounce' ? 1 : -1;          // a short's best name fell the most
  const best = go.reduce((a, b) => (sign * b.d0_pct > sign * a.d0_pct ? b : a));
  return `${go.filter(n => sign * n.d0_pct > 0).length}/${go.length} GO worked open→close · best ${best.ticker} ${pct(best.d0_pct)}`;
}
async function renderHistory() {
  const list = $('#list');
  if (!S.history) {
    list.innerHTML = '<div class="empty">Loading…</div>';
    try { S.history = (await api('/api/history')).episodes; } catch (e) {
      list.innerHTML = `<div class="empty neg">${esc(e.message)}</div>`; return;
    }
    if (S.route !== 'history') return;
  }
  if (S.sel.history == null && wide() && S.history.length) S.sel.history = 0;
  list.innerHTML = listHead('History', 'Setup days', 'Days with 2+ GO on one side in the priority report') +
    (S.history.length ? S.history.map((ep, i) =>
      `<div class="card name${wide() && S.sel.history === i ? ' sel' : ''}" role="button" tabindex="0" data-ep="${i}">` +
      `<div class="l1"><span class="tk">${esc(dayLabel(ep.target))}</span><span class="badge ${ep.side === 'bounce' ? 'long' : 'short'}">${ep.side === 'bounce' ? 'Bounce' : 'Reversal'}</span>` +
      `<span class="cap">${ep.n_go} GO</span></div><div class="l2">${esc(ep.names.slice(0, 6).map(n => n.ticker).join(' '))}<br>` +
      `<span class="muted">${esc(epSummary(ep))}</span></div></div>`).join('') : '<div class="empty">No setup days yet.</div>');
  renderEpisode();
}
function renderEpisode() {
  const ep = S.history?.[S.sel.history], reader = $('#reader');
  if (!ep) { setTitle('History', ''); reader.innerHTML = '<div class="col"><div class="empty">Pick a day.</div></div>'; return; }
  setTitle(`${esc(dayLabel(ep.target))} <span class="badge ${ep.side === 'bounce' ? 'long' : 'short'}">${ep.side === 'bounce' ? 'Bounce' : 'Reversal'}</span>`,
    `From ${ep.reports.join(', ')}`);
  const sign = ep.side === 'bounce' ? 1 : -1;
  const rows = ep.names.map(n => {
    const d0 = n.d0_pct != null ? `<b class="${sign * n.d0_pct > 0 ? 'go' : 'neg'}">${pct(n.d0_pct)}</b>` : '<span class="muted">no outcome yet</span>';
    const sub = [n.mfe_atr_d0 != null ? `best ${Number(n.mfe_atr_d0).toFixed(1)} ATR` : null,
      n.exit_r != null ? `rules ${rr(n.exit_r)}` : null, n.d1_pct != null ? `next day ${pct(n.d1_pct)}` : null].filter(Boolean).join(' · ');
    return `<div class="line"><div><b>${esc(n.ticker)}</b> <span class="badge ${recCls(n.rec)}">${esc(n.score || '')} ${esc(n.rec)}</span>` +
      `<small>${esc(n.cap || '')}</small></div><div class="r">${d0}<small>${esc(sub)}</small></div></div>`;
  }).join('');
  reader.innerHTML = `<div class="col"><div class="card"><div class="eyebrow">Open → close · ${ep.side === 'bounce' ? 'up' : 'down'} is right</div>` +
    `<div class="lines" style="margin-top:4px">${rows}</div></div>` +
    `<p class="foot-note">"Rules" R = the live entry and trail rules replayed on that day.</p></div>`;
}

// ── events ───────────────────────────────────────────────────────────────
document.addEventListener('click', e => {
  const card = e.target.closest('.name');
  if (card) {
    if (card.dataset.ep != null) return go('history', card.dataset.ep, wide());
    return go(card.dataset.side, card.dataset.tk, wide());
  }
  const t = e.target.closest('button');
  if (!t) return;
  if (t.dataset.back != null) return back();
  if (t.dataset.more) { S.showAll[t.dataset.more] = !S.showAll[t.dataset.more]; return render(); }
  if (t.dataset.tf) { S.chartTf = t.dataset.tf; return render(); }
  if (t.dataset.lside) { S.lessonSide = t.dataset.lside; return render(); }
  if (t.dataset.shot) return postShot({ side: S.route === 'playbook' ? S.lessonSide : S.route, ticker: t.dataset.tk, kind: t.dataset.shot });
  if (t.dataset.undo) return postShot({ kind: 'undo' });
});
document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'Enter' && e.target.classList?.contains('name')) return e.target.click();
  const i = ROUTES.indexOf(S.route);
  if (/^[1-4]$/.test(e.key)) return go(ROUTES[Number(e.key) - 1]);
  if (e.key === 'ArrowRight') { e.preventDefault(); return go(ROUTES[(i + 1) % ROUTES.length]); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); return go(ROUTES[(i + ROUTES.length - 1) % ROUTES.length]); }
  if (e.key === 'Escape') return back();
  if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
  e.preventDefault();
  const step = e.key === 'ArrowDown' ? 1 : -1;
  if (S.route === 'history' && S.history?.length) {
    return go('history', Math.max(0, Math.min(S.history.length - 1, (S.sel.history ?? -1) + step)), true);
  }
  if (S.route === 'bounce' || S.route === 'reversal') {
    const rows = (S.snap?.[S.route]?.rows || []).filter(r => isActive(r) || S.showAll[S.route]);
    if (!rows.length) return;
    const at = rows.findIndex(r => r.ticker === S.sel[S.route]);
    go(S.route, rows[Math.max(0, Math.min(rows.length - 1, at + step))].ticker, true);
  }
});
addEventListener('hashchange', applyRoute);
let lastWide = wide();
addEventListener('resize', () => { if (wide() !== lastWide) { lastWide = wide(); render(); } });

async function poll() {
  try {
    const snap = await api('/api/snapshot');
    await loadShots();
    S.reachable = true;
    const changed = !S.snap || snap.generated_at !== S.snap.generated_at;
    S.snap = snap;
    if (changed) render(); else renderStatus();
  } catch (e) {
    S.reachable = false;
    renderStatus();
  }
}

applyRoute();
poll();
setInterval(poll, 60e3);
setInterval(renderStatus, 30e3);
