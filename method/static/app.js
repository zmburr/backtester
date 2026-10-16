'use strict';
// Method: renders /api/snapshot (written by method/refresh.py) and polls it.
// Routes: #/bounce[/TICKER]  #/reversal[/TICKER]  #/playbook  #/history[/N[/TICKER]]
// Phone: list, then a detail pane (Back / swipe back). Desktop (≥1024px): rail +
// list + detail. Keys: ↑/↓ names, ←/→ or 1-4 sections, Esc back.
// Bounce grades are the bounce game plan's (analyzers/bounce_ladder.py):
// GO with intensity 75+ = A+, 60-75 = A, under 60 = B; 4/6 = B; 3/6 = C.

const S = {
  route: 'bounce', snap: null, sel: { bounce: null, reversal: null, history: null }, reviewTk: null,
  showAll: { bounce: false, reversal: false }, lessons: {}, lessonSide: 'bounce', history: null,
  shots: { date: null, entries: [] }, chart: null, chartTf: 'intraday', reviewTf: 'day2', bars: {},
  reviews: {}, reachable: true,
};
const ROUTES = ['bounce', 'reversal', 'playbook', 'history'];
const app = document.getElementById('app');
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x, d = 1) => x == null ? '–' : `${x >= 0 ? '+' : '−'}${Math.abs(x * 100).toFixed(d)}%`;
const upct = (x, d = 0) => x == null ? '–' : `${(x * 100).toFixed(d)}%`;
const px = x => x == null ? '–' : Number(x).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const rr = x => x == null ? '–' : `${x >= 0 ? '+' : '−'}${Math.abs(Number(x)).toFixed(1)}R`;
const wide = () => matchMedia('(min-width: 1024px)').matches;
const recCls = r => r === 'GO' ? 'go' : r === 'CAUTION' ? 'caution' : 'nogo';
const isActive = r => r.rec === 'GO' || r.rec === 'CAUTION';
const MODE_WORD = { premarket: 'Premarket', market: 'Market hours', evening: 'Evening', overnight: 'Overnight' };
const MOVE_WORD = { evening: 'after hours', overnight: 'overnight' };
const GRADE_CLS = { 'A+': 'gAp', A: 'gA', B: 'gB', C: 'gC', '-': 'gN' };
const GRADE_LINE = { 'A+': '#3ddc97', A: '#8fe3b8', B: '#ffb547', C: '#ff8a4c', '-': '#6f7873' };
const gradeLabel = g => g === '-' ? 'No trade' : g;
const gradeBadge = g => g ? `<span class="badge ${GRADE_CLS[g] || 'gN'}">${esc(gradeLabel(g))}</span>` : '';
const hhmm = t => new Date(t * 1000).toISOString().slice(11, 16);     // ET-shifted epoch -> "HH:MM"

async function api(path, opts) {
  if (opts?.method === 'POST') opts.headers = { 'Content-Type': 'application/json', 'X-Method': '1', ...(opts.headers || {}) };
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
function dayLabel(iso) {
  return new Date(`${iso}T12:00:00`).toLocaleDateString('en-US', { weekday: 'short', month: 'numeric', day: 'numeric' });
}
function clock12(t) {
  const [h, m] = (t || '0:00').split(':').map(Number);
  return `${((h + 11) % 12) + 1}:${String(m).padStart(2, '0')} ${h < 12 ? 'AM' : 'PM'}`;
}
const lastUpdate = s => [s.generated_at, s.live_at].filter(Boolean).sort().pop();
function ago(iso) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return 'just now';
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  return `${Math.round(s / 3600)} h ago`;
}
function etToday() {
  const et = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }));
  return `${et.getFullYear()}-${String(et.getMonth() + 1).padStart(2, '0')}-${String(et.getDate()).padStart(2, '0')}`;
}
function activeNow() {     // when refreshes should be landing: weekdays 4 AM-8 PM, Blue Ocean nights
  const et = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const h = et.getHours() + et.getMinutes() / 60, d = et.getDay();    // Sunday = 0
  if (d >= 1 && d <= 5 && h >= 4 && h < 20) return true;
  if (h >= 20) return d <= 4;                                          // Sun-Thu evenings
  return h < 4 && d >= 1 && d <= 5;                                    // Mon-Fri early hours
}
const recBadge = r => `<span class="badge ${recCls(r.rec)}">${esc(r.score)}/${esc(r.max_score)} ${esc(r.rec)}</span>`;
function scenBadges(r) {           // reversal names keep the gap scenarios
  if (!r.scenario) return '';
  return `<span class="badge ${r.scenario}">${r.scenario}</span>`;
}
function gradeBadges(r) {          // bounce: the grade at the open in market hours, else at the live price
  if (r.grade_open) return gradeBadge(r.grade_open) + (r.grade !== r.grade_open ? `<span class="muted">→</span>${gradeBadge(r.grade)}` : '');
  return gradeBadge(r.grade);
}

// ── routing ──────────────────────────────────────────────────────────────
function parseHash() {
  const [, route, arg, sub] = (location.hash || '#/bounce').split('/');
  return { route: ROUTES.includes(route) ? route : 'bounce', arg: arg ? decodeURIComponent(arg) : null,
           sub: sub ? decodeURIComponent(sub) : null };
}
function go(route, arg = null, replace = false, sub = null) {
  const h = `#/${route}${arg != null ? `/${encodeURIComponent(arg)}` : ''}${sub ? `/${encodeURIComponent(sub)}` : ''}`;
  if (location.hash === h) return applyRoute();
  if (replace) { history.replaceState(null, '', h); applyRoute(); } else location.hash = h;
}
function applyRoute() {
  const { route, arg, sub } = parseHash();
  if (route === 'playbook' && (S.route === 'bounce' || S.route === 'reversal')) S.lessonSide = S.route;
  S.route = route;
  if (route === 'bounce' || route === 'reversal') S.sel[route] = arg || (wide() ? S.sel[route] : null);
  if (route === 'history') {
    S.sel.history = arg != null ? Number(arg) : (wide() ? S.sel.history : null);
    S.reviewTk = sub;
  }
  app.dataset.route = route;
  app.dataset.view = arg != null || route === 'playbook' ? 'detail' : 'list';
  $$('[data-nav]').forEach(a => a.classList.toggle('on', a.dataset.nav === route));
  render();
}
function back() {
  if (S.route === 'playbook') return;
  if (S.route === 'history' && S.reviewTk) return go('history', S.sel.history);
  go(S.route);
}

// ── status ───────────────────────────────────────────────────────────────
function renderStatus() {
  const s = S.snap;
  let short = 'Connecting…', dot = '';
  if (!S.reachable) { short = 'Offline'; dot = 'off'; }
  else if (s && !s.empty) {
    const stale = Date.now() - new Date(lastUpdate(s)).getTime() > 30 * 60e3 && activeNow();
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
  S.shots = await api('/api/shots', { method: 'POST', body: JSON.stringify(body) });
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

// ── lists ────────────────────────────────────────────────────────────────
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
    `Data ${esc(s.data_time || '–')} ET · ${esc(s.source)} · updated ${esc(ago(lastUpdate(s)))}`);

  const b = s.breadth || [], goN = b.map(e => e[`${side}_go`] || 0), ca = b.map(e => e[`${side}_caution`] || 0);
  const runDay = (s.generated_at || '').slice(0, 10);
  const since = b.length ? (b[0].date && b[0].date !== runDay ? `${b[0].t} last evening` : b[0].t) : '';
  h += `<div class="kpis"><div class="kpi"><div class="eyebrow">GO</div><b class="go num">${c.GO}</b>` +
    `<small>${b.length > 1 ? `${goN[0]} → ${goN[goN.length - 1]} since ${esc(since)}` : 'watchlist'}</small></div>` +
    `<div class="kpi"><div class="eyebrow">Caution</div><b class="caution num">${c.CAUTION}</b><small>watchlist</small></div></div>`;
  if (b.length > 1) h += `<div class="card"><div class="eyebrow">Breadth since ${esc(since)}</div>${spark(goN, ca)}</div>`;

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
  const mode = S.snap.mode, move = `${pct(r.gap)} ${MOVE_WORD[mode] || 'vs close'}`;
  const src = r.src === 'Y' ? `Blue Ocean ${r.bar_time}`
    : r.src && r.src !== (S.snap.src || 'P') ? `${r.src_label || r.src} ${r.bar_time || ''}` : null;
  const nxt = r.next ? `next ${gradeLabel(r.next.grade)} at ${px(r.next.price)} (${pct(r.next.pct_away)})` : (side === 'bounce' && r.grade ? 'best grade reached' : null);
  const l2 = side === 'bounce'
    ? [move, nxt, src, r.pm_low != null ? `PM low ${px(r.pm_low)} @ ${r.pm_low_time}` : null,
       r.down_days != null ? `down ${r.down_days}d` : null]
    : [move, r.pct_from_9ema != null ? `${pct(r.pct_from_9ema, 0)} vs 9EMA` : null, `+4% at ${px(r.gap_level)}`];
  return `<div class="card name${wide() && S.sel[side] === r.ticker ? ' sel' : ''}" role="button" tabindex="0" data-side="${side}" data-tk="${esc(r.ticker)}">` +
    `<div class="l1"><span class="tk">${esc(r.ticker)}</span>${side === 'bounce' ? gradeBadges(r) : ''}${recBadge(r)}${side === 'reversal' ? scenBadges(r) : ''}` +
    `${r.principal ? '<span class="badge lev">3x</span>' : ''}${locked(side, r.ticker) ? '<span class="badge lock">LOCKED</span>' : ''}` +
    `<span class="cap">${esc(r.cap || '')}</span></div><div class="l2 num">${l2.filter(Boolean).map(esc).join(' · ')}</div></div>`;
}

// ── charts ───────────────────────────────────────────────────────────────
const CH = { text: '#6f7873', grid: '#151a1e', up: '#46b98a', down: '#d9615f' };
function snapTime(t, times) {        // markers must sit on a bar that exists
  let best = times[0];
  for (const x of times) { if (x <= t) best = x; else break; }
  return best;
}
function makeChart(el, bars, { daily = false, lines = [], markers = [], segments = [], fitDays = 90, offEl = null } = {}) {
  if (S.chart) { S.chart.remove(); S.chart = null; }
  el.innerHTML = '';
  if (!bars?.length) { el.innerHTML = '<div class="empty">No bars.</div>'; return null; }
  const c = LightweightCharts.createChart(el, {
    autoSize: true,
    layout: { background: { type: 'solid', color: 'transparent' }, textColor: CH.text, fontFamily: 'Inter, system-ui, sans-serif', fontSize: 11, attributionLogo: false },
    grid: { vertLines: { visible: false }, horzLines: { color: CH.grid } },
    rightPriceScale: { borderVisible: false, minimumWidth: 58 },
    // 960 one-minute bars (4 AM-8 PM) must fit a phone-width chart
    timeScale: { borderVisible: false, timeVisible: !daily, secondsVisible: false, rightOffset: daily ? 2 : 8, minBarSpacing: 0.05 },
    crosshair: { mode: 0, vertLine: { color: '#3a434b', labelBackgroundColor: '#1a1f24' }, horzLine: { color: '#3a434b', labelBackgroundColor: '#1a1f24' } },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
  });
  // Intraday charts scale to the candles: a level outside the day's range is listed under the
  // chart instead of stretching the scale (a squashed strip of candles reads as nothing, 10/15).
  // The daily view is the big picture, so it stretches to levels within 30% of the last close.
  const last = bars[bars.length - 1].close;
  const lo = Math.min(...bars.map(b => b.low)), hi = Math.max(...bars.map(b => b.high));
  const pad = Math.max((hi - lo) * 0.25, last * 0.025);    // near levels (prior close, PM low) stay on
  const fits = p => daily ? Math.abs(p / last - 1) <= 0.3 : p >= lo - pad && p <= hi + pad;
  const shown = lines.filter(l => l.price != null && fits(l.price));
  const off = lines.filter(l => l.price != null && !fits(l.price));
  const extra = [...shown.map(l => l.price), ...segments.map(s => s.price).filter(fits)];
  if (offEl) {
    offEl.innerHTML = off.length ? `Off chart: ${off.sort((a, b) => b.price - a.price)
      .map(l => `<span style="color:${l.color}">${esc(l.title)}</span> ${px(l.price)} (${pct(l.price / last - 1, 0)})`).join(' · ')}` : '';
  }
  const candles = c.addCandlestickSeries({
    upColor: CH.up, downColor: CH.down, wickUpColor: CH.up, wickDownColor: CH.down, borderVisible: false,
    priceLineVisible: false, lastValueVisible: false,
    autoscaleInfoProvider: original => {
      const res = original();
      if (!res || !extra.length) return res;
      return { ...res, priceRange: { minValue: Math.min(res.priceRange.minValue, ...extra), maxValue: Math.max(res.priceRange.maxValue, ...extra) } };
    },
  });
  const vol = c.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol', color: '#1c2227', lastValueVisible: false, priceLineVisible: false });
  c.priceScale('vol').applyOptions({ scaleMargins: { top: 0.88, bottom: 0 } });
  candles.setData(bars.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
  vol.setData(bars.map(b => ({ time: b.time, value: b.value })));
  for (const l of shown) {
    candles.createPriceLine({ price: l.price, color: l.color, lineWidth: 1, lineStyle: l.style ?? 2, axisLabelVisible: true, title: l.title || '' });
  }
  const times = bars.map(b => b.time);
  if (!daily) {
    for (const s of segments) {
      const a = snapTime(s.from, times), b2 = snapTime(s.to, times);
      if (b2 <= a) continue;
      const seg = c.addLineSeries({ color: s.color, lineWidth: 1, lineStyle: 2, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
      seg.setData([{ time: a, value: s.price }, { time: b2, value: s.price }]);
    }
    if (markers.length) {
      candles.setMarkers(markers.map(m => ({ ...m, time: snapTime(m.time, times) })).sort((x, y) => x.time - y.time));
    }
  }
  const fit = () => daily ? c.timeScale().setVisibleLogicalRange({ from: bars.length - fitDays, to: bars.length + 2 }) : c.timeScale().fitContent();
  fit();
  requestAnimationFrame(() => requestAnimationFrame(fit));     // autoSize sets the width a frame later
  S.chart = c;
  return c;
}
function gradeLines(ladder) {        // where each better grade starts, moving down
  return (ladder?.zones || []).slice(1).filter(z => z.grade !== '-')
    .map(z => ({ price: z.top, color: GRADE_LINE[z.grade], title: z.grade }));
}
function tradeMarks(trades, side) {
  const long = side === 'bounce', markers = [], segments = [];
  (trades || []).forEach((t, i) => {
    markers.push({ time: t.entry_t, position: long ? 'belowBar' : 'aboveBar', color: long ? '#3ddc97' : '#ff6363',
      shape: long ? 'arrowUp' : 'arrowDown', text: String(i + 1) });
    if (t.half) markers.push({ time: t.half.t, position: 'belowBar', color: '#a594ff', shape: 'circle', text: '½' });
    if (t.kind !== 'open') {
      markers.push({ time: t.exit_t, position: long ? 'aboveBar' : 'belowBar', color: t.r >= 0 ? '#a594ff' : '#ff6363',
        shape: long ? 'arrowDown' : 'arrowUp', text: rr(t.r) });
    }
    segments.push({ from: t.entry_t, to: t.exit_t, price: t.stop, color: '#ff6363' });
  });
  return { markers, segments };
}
function tradeLines(trades, side) {
  const long = side === 'bounce';
  return (trades || []).map((t, i) => {
    const exit = t.kind === 'open' ? 'still open' : `${t.kind === 'stop' ? 'stopped' : t.kind === 'trail' ? 'trail exit' : t.kind === 'tripwire' ? 'tripwire cover' : 'close'} ${hhmm(t.exit_t)} @ ${px(t.exit)}`;
    const half = t.half ? ` · half into the flush ${hhmm(t.half.t)} @ ${px(t.half.price)}` : '';
    return [`#${i + 1} ${long ? 'buy' : 'short'} ${hhmm(t.entry_t)} @ ${px(t.entry)} · stop ${px(t.stop)}`, `${exit}${half}`, t.kind === 'open' ? '' : rr(t.r)];
  });
}
const RULES_NOTE = {
  bounce: 'Rules: buy the first 2-min close above the prior bar’s high after a fresh low; stop at the low; once +1.5 ATR, trail the prior 2-min low; 2 attempts max.',
  reversal: 'Rules: short the first 2-min close below the prior bar’s low, 9:32–10:30; stop at the high of day; cover half into a 4-ATR flush, the rest on the 12:30 tripwire or at the close; 2 stop-outs lock the name.',
};

// ── name detail ──────────────────────────────────────────────────────────
function setTitle(html, sub) { $('#d-title').innerHTML = html; $('#d-sub').textContent = sub || ''; }
function kpis(pairs) {
  return `<div class="kpis dense">${pairs.filter(([, v]) => v != null && v !== '–' && v !== '').map(([k, v, cls]) =>
    `<div class="kpi"><div class="eyebrow">${esc(k)}</div><b class="num ${cls || ''}">${esc(v)}</b></div>`).join('')}</div>`;
}
function linesCard(title, rows, extra = '') {
  if (!rows.length) return '';
  return `<div class="card"><div class="eyebrow">${title}</div><div class="lines">` +
    rows.map(([a, b2, c]) => `<div class="line"><div>${a}${b2 ? `<small>${b2}</small>` : ''}</div><div class="r">${c ?? ''}</div></div>`).join('') +
    `</div>${extra}</div>`;
}
function gradeCard(r) {
  const mode = S.snap.mode, stats = S.snap.grade_stats || {};
  const g = r.grade_open || r.grade, inten = r.grade_open ? r.intensity_open : r.intensity, score = r.grade_open ? r.score_open : r.score;
  const lead = mode === 'market' ? 'at the open' : mode === 'evening' ? "at tonight's after-hours price"
    : mode === 'overnight' ? `at the overnight price (${r.src_label || 'after hours'} ${r.bar_time || ''})` : 'at the premarket price';
  const better = [];
  let seen = { '-': 0, C: 1, B: 2, A: 3, 'A+': 4 }[g] ?? 0;
  for (const lv of r.ladder?.levels || []) {
    const rank = { '-': 0, C: 1, B: 2, A: 3, 'A+': 4 }[lv.grade];
    if (lv.price < r.price && rank > seen) { better.push(lv); seen = rank; }
  }
  const st = stats[g];
  const hist = st && g !== '-' ? `Your ${gradeLabel(g)} trades: median open→close ${pct(st.med_open_close, 0)}, high ${pct(st.med_open_high, 0)}, dipped ${pct(st.med_open_low, 0)} first (${st.n} trades).` : '';
  const ceil = r.ladder?.ceiling?.blocked?.length ? `Ceiling ${gradeLabel(r.ladder.ceiling.grade)}: ${r.ladder.ceiling.blocked.join(', ')} fails at every price.` : '';
  const now = r.grade_open && r.grade !== r.grade_open ? `<div class="muted" style="margin-top:6px">Now ${gradeLabel(r.grade)} at ${px(r.price)}.</div>` : '';
  return `<div class="card callout"><div class="eyebrow">Grade ${esc(lead)}</div>
    <div style="margin-top:8px;display:flex;align-items:center;gap:12px;flex-wrap:wrap"><span class="big ${GRADE_CLS[g]}-text">${esc(gradeLabel(g))}</span>
      <span class="muted">${score}/6 · intensity ${inten != null ? Math.round(inten) : '–'}</span></div>
    ${better.length ? `<div style="margin-top:8px">${better.map(lv => `${gradeBadge(lv.grade)} at <b>${px(lv.price)}</b> <span class="muted">(${pct(lv.price / r.price - 1)})</span>`).join('&nbsp;&nbsp;')}</div>` : '<div style="margin-top:8px" class="muted">Best grade on the ladder already reached.</div>'}
    ${now}${hist ? `<div class="muted small-line">${esc(hist)}</div>` : ''}${ceil ? `<div class="muted small-line">${esc(ceil)}</div>` : ''}</div>`;
}
function ladderCard(r) {
  const levels = r.ladder?.levels || [];
  if (!levels.length) return '';
  const rows = [];
  let placed = false;
  for (const lv of levels) {
    if (!placed && lv.price < r.price) { rows.push(['<b>Now</b>', '', `<b class="num">${px(r.price)}</b>`]); placed = true; }
    rows.push([`${gradeBadge(lv.grade)} <span class="num">${px(lv.price)}</span> <span class="muted">${pct(lv.gap)} gap</span>`,
      esc(lv.events.join(' · ')), `<span class="muted num">${lv.score}/6</span>`]);
  }
  if (!placed) rows.push(['<b>Now</b>', '', `<b class="num">${px(r.price)}</b>`]);
  return linesCard('Ladder · where it changes', rows);
}
function renderDetail(side, soft = false) {
  const reader = $('#reader');
  const r = (S.snap?.[side]?.rows || []).find(x => x.ticker === S.sel[side]);
  if (!r) {
    setTitle(side === 'bounce' ? 'Bounce' : 'Reversal', '');
    reader.innerHTML = '<div class="col"><div class="empty">Pick a name from the list.</div></div>';
    return;
  }
  setTitle(`${esc(r.ticker)} ${side === 'bounce' ? gradeBadges(r) : ''} ${recBadge(r)} ${side === 'reversal' ? scenBadges(r) : ''}`,
    [r.cap, r.report_rec ? `priority report ${r.report_rec}` : null, r.bar_time ? `last bar ${r.bar_time}` : null].filter(Boolean).join(' · '));
  const mode = S.snap.mode;
  let lead = '';
  if (side === 'bounce') lead = gradeCard(r);
  else if (r.scenario) {
    const txt = S.snap.scenario_text.reversal;
    const when = mode === 'market' ? 'Now' : mode === 'evening' ? "At tonight's after-hours price" : 'In play now';
    lead = `<div class="card callout"><div class="eyebrow">Open scenario · ${esc(when)}</div><div style="margin-top:8px;display:flex;align-items:center">` +
      `<span class="big">${r.scenario}</span><span>${esc(txt[r.scenario])}</span></div></div>`;
  }
  const lev = r.principal ? `<div class="card callout warn"><div class="eyebrow">Leveraged</div><div style="margin-top:6px">Check the principal: ${esc(r.principal)}.</div></div>` : '';
  const goLevel = r.go_gap == null ? 'not within −20%' : r.go_gap >= 0 ? 'flat open' : `≤ ${px(r.go_price)}`;
  const odds = r.odds?.p_bounce5 ?? r.odds_pop?.p_bounce5;
  const pairs = side === 'bounce' ? [
    ['Price', px(r.price)], ['vs prior close', pct(r.gap), r.gap < 0 ? 'neg' : 'go'], ['GO level (5/6)', goLevel],
    ['Prior close', px(r.prior_close)],
    ['PM low', r.pm_low != null ? `${px(r.pm_low)} @ ${r.pm_low_time}` : null], ['Open', r.open != null ? `${px(r.open)} (${pct(r.open_gap)})` : null],
    ['Day low', r.day_low != null ? `${px(r.day_low)} @ ${r.day_low_time || ''}` : null], ['AH low', r.ah_low != null ? px(r.ah_low) : null],
    ['Mid-BB · swing exit', r.mid_bb != null ? `${px(r.mid_bb)} (${pct(r.to_mid_bb, 0)})` : null],
    ['ATR', r.atr != null ? `${px(r.atr)} · ${upct(r.atr_pct)}` : null], ['Down days', r.down_days != null ? String(r.down_days) : null],
    ['Off 30d high', pct(r.off_30d, 0)], ['5%+ bounce odds', odds != null ? upct(odds) : null],
    ['Setup', r.setup ? r.setup.replace('GapFade_', '').replace(/stock$/, ' stock') : null],
  ] : [
    ['+4% gap level', px(r.gap_level), 'neg'], ['Price', px(r.price)], ['vs prior close', pct(r.gap), r.gap > 0 ? 'go' : 'neg'],
    ['Prior close', px(r.prior_close)], ['Above 9EMA', pct(r.pct_from_9ema, 0)], ['9EMA', px(r.ema9)], ['3-day', pct(r.chg_3d, 0)],
    ['ATR', upct(r.atr_pct)], ['Prior-day RVOL', r.rvol != null ? `${Number(r.rvol).toFixed(2)}x` : null], ['Setup', r.setup || null],
    ['5%+ fade odds', r.odds?.p_fade5 != null ? upct(r.odds.p_fade5) : null],
  ];
  const misses = [...(r.failing || []).map(f => [esc(f.name), '', `<span class="muted">now ${esc(f.actual)}${f.threshold ? ` · needs ${esc(f.threshold)}` : ''}</span>`]),
    ...(r.gates_failed || []).map(g => [`Failed the ${esc(g)}`, '', ''])];
  const flags = [...(r.warnings || []).map(w => ['Warning', esc(w), '']), ...(r.bonuses || []).map(b => ['Bonus', esc(b), ''])];
  const comps = (r.analogs?.comps || []).slice(0, 3).map(c => [`${esc(c.ticker)} · ${esc(c.date)}`, '',
    `<span class="muted">${side === 'bounce' ? `high ${pct(c.open_high_pct, 0)}` : `low ${pct(c.open_low_pct, 0)}`} · close ${pct(c.open_close_pct, 0)}</span>`]);
  const top = $('#detail-top');
  if (soft && top && top.dataset.tk === r.ticker) {      // live price tick: keep the chart as it is
    top.innerHTML = `${lead}${lev}${kpis(pairs)}`;
    if (side === 'bounce') $('#ladder-slot').innerHTML = ladderCard(r);
    return;
  }
  reader.innerHTML = `<div class="col"><div id="detail-top" data-tk="${esc(r.ticker)}">${lead}${lev}${kpis(pairs)}</div>
    <div class="card chart-card"><div class="card-head"><span class="eyebrow">Chart${mode === 'market' ? ' · rules so far' : ''}</span>
      <div class="seg small"><button data-tf="intraday" class="${S.chartTf === 'intraday' ? 'on' : ''}">1-min</button><button data-tf="day" class="${S.chartTf === 'day' ? 'on' : ''}">Daily</button></div></div>
      <div class="chart" id="chart"></div><div class="off-chart" id="chart-off"></div><div id="live-rules"></div></div>
    <div id="ladder-slot">${side === 'bounce' ? ladderCard(r) : ''}</div>
    ${shotsCard(side, r.ticker)}
    ${linesCard("What's missing", misses)}${linesCard('Flags', flags)}${linesCard('Closest past setups', comps)}</div>`;
  drawDetailChart(side, r);
}
async function drawDetailChart(side, r) {
  const tk = r.ticker;
  let data = S.bars[tk];
  if (!data || Date.now() - data._t > 300e3) {
    try { data = await api(`/api/bars?ticker=${encodeURIComponent(tk)}`); } catch (e) { return; }
    data._t = Date.now(); S.bars[tk] = data;
  }
  if (!$('#chart') || S.sel[side] !== tk) return;
  const daily = S.chartTf === 'day';
  const lines = [{ price: r.prior_close, color: '#6f7873', title: 'prior close' }];
  if (side === 'bounce') {
    lines.push(...gradeLines(r.ladder), { price: r.mid_bb, color: '#a594ff', title: 'mid-BB' });
    if (!daily) lines.push({ price: r.pm_low, color: '#ffb547', title: 'PM low', style: 1 });
  } else {
    lines.push({ price: r.gap_level, color: '#ff6363', title: '+4%' }, { price: r.ema9, color: '#7cc4ff', title: '9EMA' });
  }
  let marks = { markers: [], segments: [] }, live = null;
  if (S.snap.mode === 'market' && !daily) {
    live = await liveReview(side, tk);
    if (live?.trades) marks = tradeMarks(live.trades, side);
    if (!$('#chart') || S.sel[side] !== tk) return;
  }
  makeChart($('#chart'), daily ? data.daily : data.intraday, { daily, lines, ...marks, offEl: $('#chart-off') });
  const box = $('#live-rules');
  if (box && live) {
    box.innerHTML = live.trades?.length
      ? `<div class="lines" style="margin-top:8px">${tradeLines(live.trades, side).map(([a, b2, c]) => `<div class="line"><div>${esc(a)}<small>${esc(b2)}</small></div><div class="r">${esc(c)}</div></div>`).join('')}</div>`
      : '<div class="muted small-line">No rules entry yet today.</div>';
  }
}
async function liveReview(side, tk) {
  const key = `${side}:${tk}:live`, hit = S.reviews[key];
  if (hit && Date.now() - hit._t < 120e3) return hit;
  try {
    const d = await api(`/api/review?side=${side}&ticker=${encodeURIComponent(tk)}&date=${etToday()}`);
    d._t = Date.now(); S.reviews[key] = d; return d;
  } catch (e) { return null; }
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
    `<button data-lside="reversal" class="${side === 'reversal' ? 'on' : ''}">Reversal</button></div></div>${pushCard()}${shotsCard(side)}`;
  if (!L.found) h += `<div class="card callout warn">Lessons note not found (${esc(L.note)}).</div>`;
  for (const sec of L.sections) {
    const risk = sec.heading.toLowerCase().startsWith('risk budget') && L.tiers;
    const items = (risk ? [`<li class="tiers">${esc(L.tiers)}</li>`] : []).concat(sec.items.map(i => `<li>${i}</li>`));
    h += `<div class="card"><h3>${esc(sec.heading)}</h3><ul class="items">${items.join('')}</ul></div>`;
  }
  h += `<p class="foot-note">From ${esc(L.note)}. Edit it in Obsidian; Method reads it fresh.</p></div>`;
  reader.innerHTML = h;
  paintPush().catch(() => {});
}

// ── push: iOS delivers only to the home-screen app, and permission must come from a tap ──
const standalone = () => matchMedia('(display-mode: standalone)').matches || navigator.standalone === true;
function pushCard() {
  return `<div class="card"><div class="eyebrow">Notifications</div><div id="push-status" class="muted" style="margin-top:6px">Checking…</div>
    <div class="shots-row"><button class="btn primary small" id="push-enable" hidden>Turn on notifications</button>
    <button class="btn ghost small" id="push-test" hidden>Send a test</button></div></div>`;
}
async function paintPush() {
  const st = $('#push-status'), en = $('#push-enable'), te = $('#push-test');
  if (!st) return;
  if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
    st.textContent = standalone() ? "This device can't receive notifications."
      : 'Add Method to your home screen (Share → Add to Home Screen) and open it from there to turn these on.';
    return;
  }
  const info = await api('/api/push');
  const reg = await navigator.serviceWorker.ready;
  const sub = await reg.pushManager.getSubscription();
  if (sub && Notification.permission === 'granted') {
    st.textContent = `On. Setups forming and grade upgrades arrive here instead of email (${info.subscriptions} device${info.subscriptions === 1 ? '' : 's'}).`;
    en.hidden = true; te.hidden = false;
  } else if (Notification.permission === 'denied') {
    st.textContent = 'Blocked. Turn them on in iPhone Settings → Notifications → Method.';
    en.hidden = true; te.hidden = true;
  } else {
    st.textContent = 'Off. Setups forming go to your email until you turn these on.';
    en.hidden = false; te.hidden = true;
  }
}
function b64ToBytes(b64) {
  const s = atob(b64.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - b64.length % 4) % 4));
  return Uint8Array.from(s, c => c.charCodeAt(0));
}
async function enablePush() {
  try {
    const perm = await Notification.requestPermission();
    if (perm !== 'granted') return paintPush();
    const { public_key } = await api('/api/push');
    const reg = await navigator.serviceWorker.ready;
    const sub = await reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: b64ToBytes(public_key) });
    await api('/api/push/subscribe', { method: 'POST', body: JSON.stringify(sub.toJSON()) });
  } catch (e) {
    const st = $('#push-status');
    if (st) st.textContent = `Couldn't turn on notifications: ${e.message}`;
    return;
  }
  paintPush();
}

// ── history ──────────────────────────────────────────────────────────────
function epSummary(ep) {
  const go = ep.names.filter(n => n.rec === 'GO' && n.d0_pct != null);
  if (!go.length) return 'no outcome yet';
  const sign = ep.side === 'bounce' ? 1 : -1;          // a short's best name fell the most
  const best = go.reduce((a, b) => (sign * b.d0_pct > sign * a.d0_pct ? b : a));
  const rules = ep.names.filter(n => n.exit_r != null);
  const rulesR = rules.length ? ` · rules ${rr(rules.reduce((a, n) => a + n.exit_r, 0))}` : '';
  return `${go.filter(n => sign * n.d0_pct > 0).length}/${go.length} GO worked open→close · best ${best.ticker} ${pct(best.d0_pct)}${rulesR}`;
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
  list.innerHTML = listHead('History', 'Setup days', 'Days with 2+ GO on one side in the priority report. Tap a name to replay it.') +
    (S.history.length ? S.history.map((ep, i) =>
      `<div class="card name${wide() && S.sel.history === i ? ' sel' : ''}" role="button" tabindex="0" data-ep="${i}">` +
      `<div class="l1"><span class="tk">${esc(dayLabel(ep.target))}</span><span class="badge ${ep.side === 'bounce' ? 'long' : 'short'}">${ep.side === 'bounce' ? 'Bounce' : 'Reversal'}</span>` +
      `<span class="cap">${ep.n_go} GO</span></div><div class="l2">${esc(ep.names.slice(0, 6).map(n => n.ticker).join(' '))}<br>` +
      `<span class="muted">${esc(epSummary(ep))}</span></div></div>`).join('') : '<div class="empty">No setup days yet.</div>');
  if (S.reviewTk) renderReview(); else renderEpisode();
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
    return `<button class="line link" data-review="${esc(n.ticker)}"><div><b>${esc(n.ticker)}</b> <span class="badge ${recCls(n.rec)}">${esc(n.score || '')} ${esc(n.rec)}</span>` +
      `<small>${esc(n.cap || '')} · replay ›</small></div><div class="r">${d0}<small>${esc(sub)}</small></div></button>`;
  }).join('');
  reader.innerHTML = `<div class="col"><div class="card"><div class="eyebrow">Open → close · ${ep.side === 'bounce' ? 'up' : 'down'} is right · tap a name to replay</div>` +
    `<div class="lines" style="margin-top:4px">${rows}</div></div>` +
    `<p class="foot-note">"Rules" R = the live entry and exit rules replayed on that day (the scorecard's number).</p></div>`;
}
async function renderReview() {
  const ep = S.history?.[S.sel.history], tk = S.reviewTk, reader = $('#reader');
  if (!ep || !tk) return renderEpisode();
  const side = ep.side, key = `${side}:${tk}:${ep.target}`;
  const name = ep.names.find(n => n.ticker === tk) || { ticker: tk };
  setTitle(esc(tk), [`${dayLabel(ep.target)} · ${side === 'bounce' ? 'bounce' : 'reversal'} replay`,
    name.rec ? `report: ${name.score || ''} ${name.rec} the night before` : null].filter(Boolean).join(' · '));
  let d = S.reviews[key];
  if (!d) {
    reader.innerHTML = '<div class="col"><div class="empty">Replaying the day…</div></div>';
    try { d = await api(`/api/review?side=${side}&ticker=${encodeURIComponent(tk)}&date=${ep.target}`); } catch (e) {
      reader.innerHTML = `<div class="col"><div class="empty neg">Couldn't replay: ${esc(e.message)}</div></div>`; return;
    }
    S.reviews[key] = d;
    if (S.reviewTk !== tk || S.history?.[S.sel.history] !== ep) return;
  }
  if (d.at_open?.grade) $('#d-title').insertAdjacentHTML('beforeend', ` ${gradeBadge(d.at_open.grade)}`);
  const sess = d.session || {}, o = d.at_open, out = d.outcome || {};
  let head;
  if (!d.candles?.length) head = `<div class="card callout warn"><div class="eyebrow">No bars</div><div style="margin-top:6px">${esc(d.error || 'Polygon has no minute bars for this day yet.')}</div></div>`;
  else if (side === 'bounce' && o) head = `<div class="card callout"><div class="eyebrow">Graded at the open</div>
      <div style="margin-top:8px;display:flex;align-items:center;gap:12px;flex-wrap:wrap"><span class="big ${GRADE_CLS[o.grade]}-text">${esc(gradeLabel(o.grade))}</span>
      <span class="muted">${o.score}/6 · intensity ${o.intensity != null ? Math.round(o.intensity) : '–'} · opened ${px(o.price)} (${pct(o.gap)})</span></div></div>`;
  else head = `<div class="card callout"><div class="eyebrow">The open</div><div style="margin-top:6px">Opened ${px(o?.price)} (${pct(o?.gap)}) vs the +4% gap level ${px(d.gap_level)}.</div></div>`;
  const total = d.rules_r != null ? `<b class="${d.rules_r >= 0 ? 'go' : 'neg'}" style="font:800 24px/1 var(--display)">${rr(d.rules_r)}</b>` : '';
  const trades = tradeLines(d.trades, side).map(([a, b2, c]) => [esc(a), esc(b2), c ? `<b class="num ${c.startsWith('+') ? 'go' : 'neg'}">${esc(c)}</b>` : '']);
  const rulesCard = `<div class="card"><div class="card-head"><span class="eyebrow">Rules replay</span>${total}</div>` +
    (trades.length ? `<div class="lines">${trades.map(([a, b2, c]) => `<div class="line"><div>${a}<small>${b2}</small></div><div class="r">${c}</div></div>`).join('')}</div>`
      : `<div class="muted">No rules entry fired that day.</div>`) + `<p class="foot-note" style="margin:10px 0 0">${esc(RULES_NOTE[side])}</p></div>`;
  const sessRows = [
    sess.pm_low != null ? ['Premarket low', `${px(sess.pm_low)} @ ${sess.pm_low_time}`] : null,
    sess.open != null ? ['Open', px(sess.open)] : null,
    sess.low != null ? ['Low of day', `${px(sess.low)} @ ${sess.low_time}`] : null,
    sess.high != null ? ['High of day', `${px(sess.high)} @ ${sess.high_time}`] : null,
    sess.close != null ? [d.live ? `Last (${sess.last_time})` : 'Close', px(sess.close)] : null,
  ].filter(Boolean).map(([a, b2]) => [esc(a), '', `<span class="num">${esc(b2)}</span>`]);
  const outRows = [
    out.d0_pct != null ? ['Open → close', '', `<b class="${(side === 'bounce' ? 1 : -1) * out.d0_pct > 0 ? 'go' : 'neg'}">${pct(out.d0_pct)}</b>`] : null,
    out.mfe_atr_d0 != null ? ['Best move from the open', '', `${Number(out.mfe_atr_d0).toFixed(1)} ATR`] : null,
    out.exit_r != null ? ['Scorecard rules R', '', rr(out.exit_r)] : null,
    out.d1_pct != null ? ['Next day', '', pct(out.d1_pct)] : null,
  ].filter(Boolean);
  reader.innerHTML = `<div class="col"><button class="btn ghost small" data-back="1" style="margin-bottom:12px">‹ ${esc(dayLabel(ep.target))}</button>${head}
    <div class="card chart-card"><div class="card-head"><span class="eyebrow">${S.reviewTf === 'daily' ? 'Daily · night-before ladder' : '2-min · 9:00–16:00'}</span>
      <div class="seg small"><button data-rtf="day2" class="${S.reviewTf === 'day2' ? 'on' : ''}">2-min</button><button data-rtf="daily" class="${S.reviewTf === 'daily' ? 'on' : ''}">Daily</button></div></div>
      <div class="chart" id="chart"></div><div class="off-chart" id="chart-off"></div></div>
    ${rulesCard}${linesCard('The session', sessRows)}${linesCard('Scorecard', outRows)}</div>`;
  if (!d.candles?.length && S.reviewTf === 'day2') return;
  const daily = S.reviewTf === 'daily';
  const lines = [{ price: d.prior_close, color: '#6f7873', title: 'prior close' }];
  if (side === 'bounce') lines.push(...gradeLines(d.ladder), { price: d.mid_bb, color: '#a594ff', title: 'mid-BB' });
  else lines.push({ price: d.gap_level, color: '#ff6363', title: '+4%' }, { price: d.ema9, color: '#7cc4ff', title: '9EMA' });
  if (!daily && sess.pm_low != null) lines.push({ price: sess.pm_low, color: '#ffb547', title: 'PM low', style: 1 });
  // the replay shows 9:00-16:00: the rules trade the session; scroll for the rest
  const tod = t => { const x = new Date(t * 1000); return x.getUTCHours() * 60 + x.getUTCMinutes(); };
  const session = (d.candles || []).filter(b => tod(b.time) >= 540 && tod(b.time) < 960);
  makeChart($('#chart'), daily ? d.daily : session, { daily, lines, fitDays: 70, offEl: $('#chart-off'),
    ...(daily ? {} : tradeMarks(d.trades, side)) });
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
  if (t.dataset.review) return go('history', S.sel.history, false, t.dataset.review);
  if (t.dataset.back != null) return back();
  if (t.dataset.more) { S.showAll[t.dataset.more] = !S.showAll[t.dataset.more]; return render(); }
  if (t.dataset.tf) { S.chartTf = t.dataset.tf; return render(); }
  if (t.dataset.rtf) { S.reviewTf = t.dataset.rtf; return render(); }
  if (t.dataset.lside) { S.lessonSide = t.dataset.lside; return render(); }
  if (t.dataset.shot) return postShot({ side: S.route === 'playbook' ? S.lessonSide : S.route, ticker: t.dataset.tk, kind: t.dataset.shot });
  if (t.dataset.undo) return postShot({ kind: 'undo' });
  if (t.id === 'push-enable') return enablePush();
  if (t.id === 'push-test') return api('/api/push/test', { method: 'POST', body: '{}' })
    .then(r => { $('#push-status').textContent = r.sent ? 'Test sent.' : 'No device is subscribed.'; });
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
    const ticked = !changed && snap.live_at !== S.snap.live_at;      // the PC posted new prices
    S.snap = snap;
    const board = S.route === 'bounce' || S.route === 'reversal';
    if (changed && board) render();
    else if (ticked && board) { renderStatus(); renderSideList(S.route); renderDetail(S.route, true); }
    else renderStatus();
  } catch (e) {
    S.reachable = false;
    renderStatus();
  }
}

if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(() => {});
applyRoute();
(async function pollLoop() {        // every 15 s while the PC streams Trillium, else every minute
  await poll();
  setTimeout(pollLoop, S.snap?.src === 'T' ? 15e3 : 60e3);
})();
setInterval(renderStatus, 30e3);
