'use strict';
// Method: renders /api/snapshot (written by method/refresh.py) and polls it.
// Routes: #/bounce[/TICKER]  #/reversal[/TICKER]  #/playbook  #/history[/N[/TICKER]]  #/best[/side:TICKER:date]
// Phone: list, then a detail pane (Back / swipe back). Desktop (≥1024px): rail +
// list + detail. Keys: ↑/↓ names, ←/→ or 1-5 sections, Esc back.
// Bounce grades are the bounce game plan's (analyzers/bounce_ladder.py):
// GO with intensity 75+ = A+, 60-75 = A, under 60 = B; 4/6 = B; 3/6 = C.

const S = {
  route: 'bounce', snap: null, sel: { bounce: null, reversal: null, history: null, best: null }, reviewTk: null,
  showAll: { bounce: false, reversal: false }, lessons: {}, lessonSide: 'bounce', history: null, lib: null,
  shots: { date: null, entries: [] }, chart: null, redraw: null, bars: {}, reviews: {}, reviewKey: null, reachable: true,
};
// Per-device chart and filter choices (a convenience: the page works the same without them).
const PREFS = (() => {
  const base = { tf: { live: '1', replay: '2' }, expanded: false,
    ov: { grades: true, trades: true, stops: true, levels: true, vwap: false, volume: true, avgs: true, bands: false, ext: false, next: false },
    best: { side: 'bounce', grades: ['A'], cap: '', setup: '', sort: 'move' } };
  try {
    const v = JSON.parse(localStorage.getItem('method.prefs') || '{}');
    return { ...base, ...v, tf: { ...base.tf, ...v.tf }, ov: { ...base.ov, ...v.ov }, best: { ...base.best, ...v.best } };
  } catch (e) { return base; }
})();
function savePrefs() { try { localStorage.setItem('method.prefs', JSON.stringify(PREFS)); } catch (e) { /* private mode */ } }
const ROUTES = ['bounce', 'reversal', 'playbook', 'history', 'best'];
const app = document.getElementById('app');
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x, d = 1) => x == null ? '–' : `${x >= 0 ? '+' : '−'}${Math.abs(x * 100).toFixed(d)}%`;
const upct = (x, d = 0) => x == null ? '–' : `${(x * 100).toFixed(d)}%`;
const px = x => x == null ? '–' : Number(x).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const rr = x => x == null ? '–' : Math.abs(x) < 0.05 ? '0.0R' : `${x >= 0 ? '+' : '−'}${Math.abs(Number(x)).toFixed(1)}R`;
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
  if (route === 'best') S.sel.best = arg || (wide() ? S.sel.best : null);
  app.dataset.route = route;
  app.dataset.view = arg != null || route === 'playbook' ? 'detail' : 'list';
  $$('[data-nav]').forEach(a => a.classList.toggle('on', a.dataset.nav === route));
  render();
}
function back() {
  if (S.route === 'playbook') return;
  S.reviewKey = null;
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
    const mb = s.market_breadth;
    if (mb?.status === 'ok') short += ` · breadth ${mb.n}`;
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
  } else if (r === 'best') {
    renderBest();
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

  if (side === 'bounce') h += breadthCard(s);
  const b = s.breadth || [], goN = b.map(e => e[`${side}_go`] || 0), ca = b.map(e => e[`${side}_caution`] || 0);
  const runDay = (s.generated_at || '').slice(0, 10);
  const since = b.length ? (b[0].date && b[0].date !== runDay ? `${b[0].t} last evening` : b[0].t) : '';
  h += `<div class="kpis"><div class="kpi"><div class="eyebrow">GO</div><b class="go num">${c.GO}</b>` +
    `<small>${b.length > 1 ? `${goN[0]} → ${goN[goN.length - 1]} since ${esc(since)}` : 'watchlist'}</small></div>` +
    `<div class="kpi"><div class="eyebrow">Caution</div><b class="caution num">${c.CAUTION}</b><small>watchlist</small></div></div>`;
  if (b.length > 1) h += `<div class="card"><div class="eyebrow">Watchlist GO and caution since ${esc(since)}</div>${spark(goN, ca)}</div>`;

  const rep = (s.reports || [])[0];
  if (rep) {
    // Grouped by what the report called; each chip is coloured by the name's live status and
    // carries its live grade, so a report GO that has since faded doesn't still read green.
    const v = rep[side], byTk = Object.fromEntries(rows.map(r => [r.ticker, r]));
    const chip = t => {
      const r = byTk[t];
      if (!r) return `<span class="badge nogo">${esc(t)}</span>`;          // off the board: no live read
      const g = side === 'bounce' ? r.grade_open || r.grade : null;
      return `<button class="badge ${recCls(r.rec)} pick" data-pick="${side}" data-tk="${esc(t)}">${esc(t)}` +
        `${g ? `<small>${g === '-' ? '–' : esc(g)}</small>` : ''}</button>`;
    };
    const group = (call, tks) => tks.length
      ? `<div class="rep-group"><div class="rep-call">Called ${call}</div><div class="tickers">${tks.map(chip).join('')}</div></div>` : '';
    const n = v.GO.length + v.CAUTION.length;
    h += `<div class="card"><div class="eyebrow">Priority report · ${esc(clock12(rep.time))} ${esc(dayLabel(rep.date))}</div>` +
      (n ? group('GO', v.GO) + group('CAUTION', v.CAUTION) +
        `<div class="rep-note">Colors and grades are live, for ${esc(dayLabel(s.session))}. Tap a name for its chart.</div>`
        : `<div class="muted" style="margin-top:6px;font-size:13px">No ${side} windows</div>`) + '</div>';
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
// ── market breadth (method/breadth.py): the day's count and its replayed edge ──
const STATE_WORD = { normal: 'Normal day', elevated: 'Elevated', breadth: 'Breadth day' };
const STATE_CLS = { normal: 'nogo', elevated: 'caution', breadth: 'go' };
const STATE_DAY = { normal: 'a normal day', elevated: 'an elevated day', breadth: 'a breadth day' };
const edgeText = e => e?.mean_r == null ? null : `${rr(e.mean_r)} a name · ${upct(e.win)} win`;
function breadthCard(s) {
  const mb = s.market_breadth;
  if (!mb) return '';
  if (mb.status !== 'ok') {
    return `<div class="card breadth"><div class="eyebrow">Market breadth</div>
      <div class="muted small-line">Counts for ${esc(dayLabel(mb.session))} are built at 5:45 PM ET.</div></div>`;
  }
  const e = mb.edge || {};
  const when = mb.basis === 'open' ? 'at the open'
    : s.mode === 'evening' ? 'if it opened at the after-hours price'
    : s.mode === 'market' ? 'so far, opens still coming in' : 'if it opens here';
  const onBoard = new Set((s.bounce?.rows || []).map(r => r.ticker));
  const detail = mb.detail || {};
  const chip = (t, cls) => onBoard.has(t)
    ? `<button class="badge ${cls}" data-btk="${esc(t)}">${esc(t)}</button>`
    : `<span class="badge ${cls}">${esc(t)}</span>`;
  const counted = mb.names.filter(t => detail[t]?.counted), extra = mb.names.filter(t => !detail[t]?.counted);
  const chips = [...counted.map(t => chip(t, 'go')), ...extra.map(t => chip(t, 'nogo'))];
  const relaxed = mb.state === 'breadth' ? (mb.relaxed || []).map(t => chip(t, 'caution')) : [];
  const near = mb.locked ? [] : (mb.near || []).slice(0, 4).map(x => `${x.ticker} ${px(x.flag_price)} (${pct(-x.away)})`);
  const days = e.traded_days ? `${e.n} names on ${Number(e.traded_days).toLocaleString('en-US')} days like this: ` +
    `${e.pos_days} of ${e.traded_days} days positive, typical day ${rr(e.day_mean_r)}. Your entry and trail rules.` : '';
  const etf = mb.n_etf ? ` · ${mb.n_etf} ${mb.n_etf === 1 ? 'is an ETF' : 'are ETFs'}` : '';
  return `<div class="card breadth st-${esc(mb.state)}">
    <div class="eyebrow">Market breadth · ${esc(when)}${mb.asof ? ` · ${esc(mb.asof.slice(11, 16))}` : ''}</div>
    <div class="bline"><b class="num">${mb.n}</b><span>names B or better market-wide${esc(etf)}</span><span class="badge ${STATE_CLS[mb.state]}">${esc(STATE_WORD[mb.state])}</span></div>
    ${edgeText(e) ? `<div class="edge">Bounce edge today: <b class="num ${e.mean_r >= 0.5 ? 'go' : ''}">${esc(edgeText(e))}</b><small>${esc(days)}</small></div>` : ''}
    ${chips.length ? `<div class="tickers">${chips.join('')}</div>` : ''}
    ${relaxed.length ? `<div class="muted small-line">Relaxed rules (breadth days only, not counted)</div><div class="tickers">${relaxed.join('')}</div>` : ''}
    ${extra.length ? '<div class="muted small-line">Grey: outside the counted universe (under $5 or under 500k shares a day).</div>' : ''}
    ${near.length ? `<div class="muted small-line">Next to flag: ${esc(near.join(' · '))}</div>` : ''}</div>`;
}

function nameCard(side, r) {
  const mode = S.snap.mode, move = `${pct(r.gap)} ${MOVE_WORD[mode] || 'vs close'}`;
  const src = r.src === 'Y' ? `Blue Ocean ${r.bar_time}`
    : r.src && r.src !== (S.snap.src || 'P') ? `${r.src_label || r.src} ${r.bar_time || ''}` : null;
  const nxt = r.next ? `next ${gradeLabel(r.next.grade)} at ${px(r.next.price)} (${pct(r.next.pct_away)})` : (side === 'bounce' && r.grade ? 'best grade reached' : null);
  const l2 = side === 'bounce'
    ? [move, r.edge?.mean_r != null ? `edge ${rr(r.edge.mean_r)}` : null, nxt, src,
       r.pm_low != null ? `PM low ${px(r.pm_low)} @ ${r.pm_low_time}` : null,
       r.down_days != null ? `down ${r.down_days}d` : null]
    : [move, r.pct_from_9ema != null ? `${pct(r.pct_from_9ema, 0)} vs 9EMA` : null, `+4% at ${px(r.gap_level)}`];
  return `<div class="card name${wide() && S.sel[side] === r.ticker ? ' sel' : ''}" role="button" tabindex="0" data-side="${side}" data-tk="${esc(r.ticker)}">` +
    `<div class="l1"><span class="tk">${esc(r.ticker)}</span>${side === 'bounce' ? gradeBadges(r) : ''}${recBadge(r)}${side === 'reversal' ? scenBadges(r) : ''}` +
    `${r.principal ? '<span class="badge lev">3x</span>' : ''}${locked(side, r.ticker) ? '<span class="badge lock">LOCKED</span>' : ''}` +
    `${r.market ? `<span class="badge mkt">${r.breadth_flag === 'relaxed' ? 'market · relaxed' : r.counted === false ? 'market · not counted' : 'market'}</span>` : ''}` +
    `<span class="cap">${esc(r.cap || '')}</span></div><div class="l2 num">${l2.filter(Boolean).map(esc).join(' · ')}</div></div>`;
}

// ── charts ───────────────────────────────────────────────────────────────
// One drawing path for every chart (live names, History, Best): 1-min bars in,
// the chosen timeframe out, overlays switched by the check boxes under the chart.
const CH = { text: '#6f7873', grid: '#151a1e', up: '#46b98a', down: '#d9615f' };
const BAND = { 'A+': 'rgba(61,220,151,.12)', A: 'rgba(143,227,184,.075)', B: 'rgba(255,181,71,.06)', C: 'rgba(255,138,76,.045)' };
const OV = { grades: 'Grades', trades: 'Trades', stops: 'Stops', levels: 'Levels', vwap: 'VWAP', volume: 'Volume',
  avgs: 'Averages', bands: 'Bollinger', ext: 'Extended hours', next: 'Next day' };
const tod = t => { const x = new Date(t * 1000); return x.getUTCHours() * 60 + x.getUTCMinutes(); };   // ET-shifted epoch -> minute of day
const avg = a => a.reduce((s, x) => s + x, 0) / a.length;
const fmtVol = v => v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${Math.round(v / 1e3)}K` : String(Math.round(v || 0));
function snapTime(t, times) {        // markers must sit on a bar that exists
  let best = times[0];
  for (const x of times) { if (x <= t) best = x; else break; }
  return best;
}
function resample(bars, m) {         // 1-min -> m-min, keeping each bar's volume-weighted price for VWAP
  const tp = b => b.vw ?? (b.high + b.low + b.close) / 3;
  if (m <= 1) return bars;
  const s = m * 60, out = [];
  let cur = null;
  const done = c => out.push({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close, value: c.value,
    vw: c.value ? c.pv / c.value : c.close });
  for (const b of bars) {
    const t = Math.floor(b.time / s) * s, v = b.value || 0;
    if (!cur || cur.time !== t) {
      if (cur) done(cur);
      cur = { time: t, open: b.open, high: b.high, low: b.low, close: b.close, value: v, pv: tp(b) * v };
    } else {
      cur.high = Math.max(cur.high, b.high); cur.low = Math.min(cur.low, b.low); cur.close = b.close;
      cur.value += v; cur.pv += tp(b) * v;
    }
  }
  if (cur) done(cur);
  return out;
}
function sessionWindow(bars, ext) {  // 9:00-16:00 once the session has bars; premarket-only days show everything
  if (ext || !bars.some(b => tod(b.time) >= 570 && tod(b.time) < 960)) return bars;
  return bars.filter(b => tod(b.time) >= 540 && tod(b.time) < 960);
}
function vwapDays(bars) {            // the session VWAP from 9:30, one line per day (4.2 joins a line across gaps)
  const days = [];
  let day = null, pv = 0, v = 0;
  for (const b of bars) {
    const m = tod(b.time);
    if (m < 570 || m >= 960) continue;
    const d = Math.floor(b.time / 86400);
    if (d !== day) { day = d; pv = 0; v = 0; days.push([]); }
    pv += (b.vw ?? (b.high + b.low + b.close) / 3) * (b.value || 0); v += b.value || 0;
    if (v) days[days.length - 1].push({ time: b.time, value: pv / v });
  }
  return days.filter(x => x.length);
}
function dailyAverages(bars) {       // 9EMA, 20SMA (= mid-BB) with 2-sigma bands, 50SMA
  const ema = [], s20 = [], s50 = [], up = [], lo = [];
  let e = null;
  bars.forEach((b, i) => {
    e = e == null ? b.close : 0.2 * b.close + 0.8 * e;
    ema.push({ time: b.time, value: e });
    if (i >= 19) {
      const w = bars.slice(i - 19, i + 1).map(x => x.close), m = avg(w);
      const sd = Math.sqrt(w.reduce((s, x) => s + (x - m) ** 2, 0) / 19);
      s20.push({ time: b.time, value: m }); up.push({ time: b.time, value: m + 2 * sd }); lo.push({ time: b.time, value: m - 2 * sd });
    }
    if (i >= 49) s50.push({ time: b.time, value: avg(bars.slice(i - 49, i + 1).map(x => x.close)) });
  });
  return { ema, s20, s50, up, lo };
}
class Bands {                        // the bounce grade zones, painted faintly behind the candles
  constructor(zones) { this.zones = zones; this.series = null; }
  attached({ series }) { this.series = series; }
  detached() { this.series = null; }
  updateAllViews() {}
  paneViews() {
    return [{
      zOrder: () => 'bottom',
      renderer: () => ({
        draw: target => target.useBitmapCoordinateSpace(({ context: ctx, bitmapSize, verticalPixelRatio: vr }) => {
          if (!this.series) return;
          for (const z of this.zones) {
            const a = this.series.priceToCoordinate(z.top), b = this.series.priceToCoordinate(z.bottom);
            if (a == null || b == null) continue;
            ctx.fillStyle = BAND[z.grade];
            ctx.fillRect(0, Math.min(a, b) * vr, bitmapSize.width, Math.abs(b - a) * vr);
          }
        }),
      }),
    }];
  }
}
function makeChart(el, bars, { daily = false, lines = [], markers = [], segments = [], bands = [], series = [], fitDays = 90,
  offEl = null, legendEl = null, volume = true, refs = {} } = {}) {
  if (S.chart) { S.chart.remove(); S.chart = null; }
  el.innerHTML = '';
  if (legendEl) legendEl.innerHTML = '';
  if (offEl) offEl.innerHTML = '';
  if (!bars?.length) { el.innerHTML = '<div class="empty">No bars.</div>'; return null; }
  const c = LightweightCharts.createChart(el, {
    autoSize: true,
    layout: { background: { type: 'solid', color: 'transparent' }, textColor: CH.text, fontFamily: 'Inter, system-ui, sans-serif', fontSize: 11, attributionLogo: false },
    grid: { vertLines: { visible: false }, horzLines: { color: CH.grid } },
    // room at the top for the readout, at the bottom for volume
    rightPriceScale: { borderVisible: false, minimumWidth: 58, scaleMargins: { top: legendEl ? 0.16 : 0.08, bottom: volume ? 0.16 : 0.06 } },
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
  if (bands.length) candles.attachPrimitive(new Bands(bands));
  candles.setData(bars.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
  if (volume) {
    const vol = c.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol', lastValueVisible: false, priceLineVisible: false });
    c.priceScale('vol').applyOptions({ scaleMargins: { top: 0.86, bottom: 0 } });
    vol.setData(bars.map(b => ({ time: b.time, value: b.value || 0, color: b.close >= b.open ? 'rgba(70,185,138,.4)' : 'rgba(217,97,95,.4)' })));
  }
  for (const sr of series) {           // overlays never stretch the scale
    const ls = c.addLineSeries({ color: sr.color, lineWidth: sr.width || 1, lineStyle: sr.style ?? 0, lastValueVisible: false,
      priceLineVisible: false, crosshairMarkerVisible: false, autoscaleInfoProvider: () => null });
    ls.setData(sr.data);
  }
  for (const l of shown) {
    candles.createPriceLine({ price: l.price, color: l.color, lineWidth: 1, lineStyle: l.style ?? 2, axisLabelVisible: true, title: l.title || '' });
  }
  const times = bars.map(b => b.time);
  if (!daily) {
    for (const s of segments) {
      const a = snapTime(s.from, times), b2 = snapTime(s.to, times);
      if (b2 <= a) continue;
      const seg = c.addLineSeries({ color: s.color, lineWidth: 1, lineStyle: 2, lastValueVisible: false, priceLineVisible: false,
        crosshairMarkerVisible: false, autoscaleInfoProvider: () => null });
      seg.setData([{ time: a, value: s.price }, { time: b2, value: s.price }]);
    }
  }
  if (markers.length) {
    candles.setMarkers(markers.map(m => ({ ...m, time: snapTime(m.time, times) })).sort((x, y) => (x.time < y.time ? -1 : x.time > y.time ? 1 : 0)));
  }
  if (legendEl) {                      // the crosshair readout: OHLC and the moves that matter
    const show = i => {
      const b = bars[i];
      const nextDay = refs.nextStart != null && b.time >= refs.nextStart;
      const prior = daily ? (i > 0 ? bars[i - 1].close : null) : nextDay ? refs.dayClose : refs.prior;
      const open = nextDay ? refs.nextOpen : refs.open;
      legendEl.innerHTML = `<b>${daily ? esc(b.time) : hhmm(b.time)}</b> <span class="ohl">O ${px(b.open)} H ${px(b.high)} L ${px(b.low)} </span>C ${px(b.close)}` +
        (prior ? ` · <span class="${b.close >= prior ? 'go' : 'neg'}">${pct(b.close / prior - 1)}</span> vs ${daily ? 'prior day' : 'prior close'}` : '') +
        (!daily && open ? ` · ${pct(b.close / open - 1)} vs open` : '') + ` · vol ${fmtVol(b.value)}`;
    };
    show(bars.length - 1);
    c.subscribeCrosshairMove(p => show(p.point && p.logical != null ? Math.max(0, Math.min(bars.length - 1, Math.round(p.logical))) : bars.length - 1));
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
function tradeMarks(trades, side, m = 2, bars1 = []) {
  // The rules fire on 2-min closes (entry_t / exit_t = that 2-min bar's start). On any
  // timeframe a fill at a bar's close sits on the bar holding that close; a stop or a
  // trail sits on the 1-min bar that touched it.
  const long = side === 'bounce', markers = [], segments = [], s = m * 60;
  const fl = t => Math.floor(t / s) * s;
  const atClose = t => fl(t + 119);
  const touched = (t0, price, up) => {
    const b = bars1.find(x => x.time >= t0 && x.time < t0 + 120 && (up ? x.high >= price : x.low <= price));
    return fl(b ? b.time : t0);
  };
  (trades || []).forEach((t, i) => {
    const inAt = atClose(t.entry_t);
    markers.push({ time: inAt, position: long ? 'belowBar' : 'aboveBar', color: long ? '#3ddc97' : '#ff6363',
      shape: long ? 'arrowUp' : 'arrowDown', text: String(i + 1) });
    if (t.half) markers.push({ time: touched(t.half.t, t.half.price, false), position: 'belowBar', color: '#a594ff', shape: 'circle', text: '½' });
    let outAt = atClose(t.exit_t);
    if (t.kind === 'stop' || t.kind === 'trail') outAt = touched(t.exit_t, t.kind === 'stop' ? t.stop : t.exit, !long);
    if (t.kind !== 'open') {
      markers.push({ time: outAt, position: long ? 'aboveBar' : 'belowBar', color: t.r >= 0 ? '#a594ff' : '#ff6363',
        shape: long ? 'arrowDown' : 'arrowUp', text: rr(t.r) });
    }
    segments.push({ from: inAt, to: outAt, price: t.stop, color: '#ff6363' });
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
function chartCardHtml(scope, title) {
  const tf = PREFS.tf[scope];
  return `<div class="card chart-card${PREFS.expanded ? ' expanded' : ''}"><div class="card-head"><span class="eyebrow" id="chart-title">${esc(title)}</span>
    <div class="chart-tools"><div class="seg small">${['1', '2', '5', 'D'].map(x =>
      `<button data-ctf="${x}" data-scope="${scope}" class="${tf === x ? 'on' : ''}">${x === 'D' ? 'Daily' : `${x}m`}</button>`).join('')}</div>
    <button class="btn ghost small" data-expand="1">${PREFS.expanded ? 'Smaller' : 'Bigger'}</button></div></div>
    <div class="chart-wrap"><div class="chart" id="chart"></div><div class="legend" id="chart-legend"></div></div>
    <div class="off-chart" id="chart-off"></div><div class="toggles" id="chart-toggles"></div><div id="live-rules"></div></div>`;
}
function togglesHtml(keys) {
  return keys.map(k => `<label class="tog"><input type="checkbox" data-ov="${k}"${PREFS.ov[k] ? ' checked' : ''}><span>${OV[k]}</span></label>`).join('');
}
function levelLines(side, r, sess = null) {
  const out = [{ price: r.prior_close, color: '#6f7873', title: 'prior close' }];
  if (side === 'bounce') out.push({ price: r.mid_bb, color: '#a594ff', title: 'mid-BB' });
  else out.push({ price: r.gap_level, color: '#ff6363', title: '+4%' }, { price: r.ema9, color: '#7cc4ff', title: '9EMA' });
  const pmLow = sess ? sess.pm_low : side === 'bounce' ? r.pm_low : null;
  if (pmLow != null) out.push({ price: pmLow, color: '#ffb547', title: 'PM low', style: 1, intraday: true });
  return out;
}
function drawChart({ side, tf, bars1 = [], next1 = [], dailyBars = [], levels = [], ladder = null, trades = null, tradeDay = null, refs = {} }) {
  if (!$('#chart')) return;
  const ov = PREFS.ov, daily = tf === 'D', m = daily ? 1 : Number(tf);
  const hasTrades = !daily && !!trades?.length;
  const keys = [side === 'bounce' && ladder && 'grades', hasTrades && 'trades', hasTrades && 'stops', 'levels',
    !daily && 'vwap', daily && 'avgs', daily && 'bands', 'volume', !daily && 'ext', !daily && next1.length && 'next'].filter(Boolean);
  $('#chart-toggles').innerHTML = togglesHtml(keys);
  const grades = ov.grades && side === 'bounce';
  const lines = [...(ov.levels ? levels.filter(l => !(daily && l.intraday)) : []), ...(grades ? gradeLines(ladder) : [])];
  const bands = grades ? (ladder?.zones || []).filter(z => BAND[z.grade]) : [];
  const r2 = { ...refs }, markers = [], series = [];
  let bars, segments = [];
  if (daily) {
    bars = dailyBars;
    if (bars.length > 1 && (ov.avgs || ov.bands)) {
      const a = dailyAverages(bars);
      if (ov.avgs) series.push({ data: a.ema, color: '#7cc4ff' }, { data: a.s20, color: '#a594ff' }, { data: a.s50, color: 'rgba(242,244,239,.5)' });
      if (ov.bands) series.push({ data: a.up, color: 'rgba(165,148,255,.55)', style: 2 }, { data: a.lo, color: 'rgba(165,148,255,.55)', style: 2 });
    }
    if (tradeDay && bars.some(b => b.time === tradeDay)) {
      markers.push({ time: tradeDay, position: side === 'bounce' ? 'belowBar' : 'aboveBar', color: '#a594ff',
        shape: side === 'bounce' ? 'arrowUp' : 'arrowDown', text: 'trade' });
    }
  } else {
    const nxt = ov.next ? sessionWindow(next1, ov.ext) : [];
    bars = resample([...sessionWindow(bars1, ov.ext), ...nxt], m);
    if (ov.vwap) series.push(...vwapDays(bars).map(data => ({ data, color: '#e5c07b', width: 2 })));
    if (hasTrades) {
      const tm = tradeMarks(trades, side, m, bars1);
      if (ov.trades) markers.push(...tm.markers);
      if (ov.stops) segments = tm.segments;
    }
    if (nxt.length) {
      r2.nextStart = Math.floor(nxt[0].time / (m * 60)) * m * 60;
      markers.push({ time: r2.nextStart, position: 'aboveBar', color: '#aab2ad', shape: 'square', text: 'next day' });
    }
  }
  makeChart($('#chart'), bars, { daily, lines, markers, segments, bands, series, fitDays: 70, offEl: $('#chart-off'),
    legendEl: $('#chart-legend'), volume: ov.volume, refs: r2 });
}

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
  const mb = S.snap.market_breadth;
  const edge = r.edge?.mean_r != null && mb?.state
    ? `On ${STATE_DAY[mb.state]} (${mb.n} names market-wide), ${score >= 4 ? `${Math.min(score, 6)}/6` : '3/6-or-lower'} names averaged ${rr(r.edge.mean_r)} with ${upct(r.edge.win)} wins under your rules (${r.edge.n} names${r.edge.shrunk ? ', a small sample pulled toward the day average' : ''}).`
    : '';
  const mkt = r.market ? `Flagged by the market-wide breadth scan, not on your watchlist${r.counted === false ? ' (outside the counted universe)' : ''}.` : '';
  return `<div class="card callout"><div class="eyebrow">Grade ${esc(lead)}</div>
    <div style="margin-top:8px;display:flex;align-items:center;gap:12px;flex-wrap:wrap"><span class="big ${GRADE_CLS[g]}-text">${esc(gradeLabel(g))}</span>
      <span class="muted">${score}/6 · intensity ${inten != null ? Math.round(inten) : '–'}</span></div>
    ${better.length ? `<div style="margin-top:8px">${better.map(lv => `${gradeBadge(lv.grade)} at <b>${px(lv.price)}</b> <span class="muted">(${pct(lv.price / r.price - 1)})</span>`).join('&nbsp;&nbsp;')}</div>` : '<div style="margin-top:8px" class="muted">Best grade on the ladder already reached.</div>'}
    ${now}${edge ? `<div class="small-line">${esc(edge)}</div>` : ''}${mkt ? `<div class="muted small-line">${esc(mkt)}</div>` : ''}${hist ? `<div class="muted small-line">${esc(hist)}</div>` : ''}${ceil ? `<div class="muted small-line">${esc(ceil)}</div>` : ''}</div>`;
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
    ${chartCardHtml('live', 'Chart')}
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
  let live = null;
  if (S.snap.mode === 'market') {
    live = await liveReview(side, tk);
    if (!$('#chart') || S.sel[side] !== tk) return;
  }
  S.redraw = () => {                   // reads the row fresh: live ticks move the levels
    const row = (S.snap?.[side]?.rows || []).find(x => x.ticker === tk);
    if (!row || !$('#chart') || S.sel[side] !== tk) return;
    const tf = PREFS.tf.live;
    $('#chart-title').textContent = tf === 'D' ? 'Daily' : `${tf}-min${S.snap.mode === 'market' ? ' · rules so far' : ''}`;
    drawChart({ side, tf, bars1: data.intraday || [], dailyBars: data.daily || [], levels: levelLines(side, row),
      ladder: side === 'bounce' ? row.ladder : null, trades: live?.trades, refs: { prior: row.prior_close, open: row.open } });
  };
  S.redraw();
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
function renderReview() {
  const ep = S.history?.[S.sel.history], tk = S.reviewTk;
  if (!ep || !tk) return renderEpisode();
  const name = ep.names.find(n => n.ticker === tk) || { ticker: tk };
  showReview({ side: ep.side, tk, date: ep.target, title: esc(tk),
    sub: [`${dayLabel(ep.target)} · ${ep.side === 'bounce' ? 'bounce' : 'reversal'} replay`,
      name.rec ? `report: ${name.score || ''} ${name.rec} the night before` : null].filter(Boolean).join(' · '),
    back: `<button class="btn ghost small" data-back="1" style="margin-bottom:12px">‹ ${esc(dayLabel(ep.target))}</button>` });
}

// ── one replayed day (History and Best) ─────────────────────────────────
async function showReview(ctx) {
  const { side, tk, date } = ctx, reader = $('#reader'), key = `${side}:${tk}:${date}`;
  S.reviewKey = key;
  setTitle(ctx.title, ctx.sub);
  let d = S.reviews[key];
  if (!d) {
    reader.innerHTML = `<div class="col">${ctx.back || ''}<div class="empty">Replaying the day…</div></div>`;
    try {
      d = await api(`/api/review?side=${side}&ticker=${encodeURIComponent(tk)}&date=${date}${ctx.cap ? `&cap=${encodeURIComponent(ctx.cap)}` : ''}`);
    } catch (e) {
      if (S.reviewKey === key) reader.innerHTML = `<div class="col">${ctx.back || ''}<div class="empty neg">Couldn't replay: ${esc(e.message)}</div></div>`;
      return;
    }
    S.reviews[key] = d;
    if (S.reviewKey !== key) return;
  }
  if (d.at_open?.grade) $('#d-title').insertAdjacentHTML('beforeend', ` ${gradeBadge(d.at_open.grade)}`);
  const sess = d.session || {}, o = d.at_open, out = d.outcome || {};
  let head;
  if (!d.candles?.length) head = `<div class="card callout warn"><div class="eyebrow">No bars</div><div style="margin-top:6px">${
    date > etToday() ? `${esc(dayLabel(date))} hasn't traded yet.` : esc(d.error || 'Polygon has no minute bars for this day yet.')}</div></div>`;
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
  const ns = d.next_session || {}, dc = sess.close;
  const nextRows = d.next_date && ns.open != null ? [
    ['Open', ns.open, dc], ['High', ns.high, ns.open], ['Low', ns.low, ns.open], ['Close', ns.close, dc],
  ].map(([k, v, ref]) => [esc(k), '', `<span class="num">${px(v)}</span> <span class="muted">${ref ? pct(v / ref - 1) : ''} vs ${k === 'High' || k === 'Low' ? 'its open' : 'the close before'}</span>`]) : [];
  const outRows = [
    out.d0_pct != null ? ['Open → close', '', `<b class="${(side === 'bounce' ? 1 : -1) * out.d0_pct > 0 ? 'go' : 'neg'}">${pct(out.d0_pct)}</b>`] : null,
    out.mfe_atr_d0 != null ? ['Best move from the open', '', `${Number(out.mfe_atr_d0).toFixed(1)} ATR`] : null,
    out.exit_r != null ? ['Scorecard rules R', '', rr(out.exit_r)] : null,
    out.d1_pct != null ? ['Next day', '', pct(out.d1_pct)] : null,
  ].filter(Boolean);
  reader.innerHTML = `<div class="col">${ctx.back || ''}${head}${chartCardHtml('replay', '')}${rulesCard}${ctx.cards ? ctx.cards(d) : ''}
    ${linesCard('The session', sessRows)}${linesCard(d.next_date ? `Next day · ${esc(dayLabel(d.next_date))}` : 'Next day', nextRows)}${linesCard('Scorecard', outRows)}</div>`;
  S.redraw = () => {
    if (S.reviewKey !== key || !$('#chart')) return;
    const tf = PREFS.tf.replay;
    $('#chart-title').textContent = tf === 'D' ? 'Daily · night-before ladder' : `${tf}-min${PREFS.ov.ext ? '' : ' · 9:00–16:00'}`;
    if (tf !== 'D' && !d.candles?.length) { $('#chart').innerHTML = '<div class="empty">No bars.</div>'; return; }
    drawChart({ side, tf, bars1: d.candles || [], next1: d.next_candles || [], dailyBars: d.daily || [],
      levels: levelLines(side, d, sess), ladder: side === 'bounce' ? d.ladder : null, trades: d.trades, tradeDay: date,
      refs: { prior: d.prior_close, open: sess.open, dayClose: sess.close, nextOpen: ns.open } });
  };
  S.redraw();
}

// ── best trades: your graded trades, replayed with the systems on them ──
const dayLabelY = iso => new Date(`${iso}T12:00:00`).toLocaleDateString('en-US', { weekday: 'short', month: 'numeric', day: 'numeric', year: '2-digit' });
const shortSetup = s => s.replace(/_?(weak|strong)stock$/, (_, w) => ` · ${w}`).replace(/^GapFade/, 'Gap fade');
// Intraday capitulations form after the open, so the grade at the open doesn't apply to them.
const intradaySetup = r => r.side === 'bounce' && /IntradayCapitch/.test(r.setup || '');
const median = a => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y), h = s.length >> 1; return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2; };
function bestRows() {
  const f = PREFS.best;
  const rows = (S.lib || []).filter(r => r.side === f.side && (!f.grades.length || f.grades.includes(r.grade))
    && (!f.cap || r.cap === f.cap) && (!f.setup || r.setup === f.setup));
  const val = { move: r => r.move_atr, rules: r => r.rules_r }[f.sort];
  return rows.sort(val ? (a, b) => (val(b) ?? -1e9) - (val(a) ?? -1e9) : (a, b) => (a.date < b.date ? 1 : -1));
}
function bestSummary(rows) {
  if (!rows.length) return '';
  const ok = rows.filter(r => r.ok), rs = ok.filter(r => r.rules_r != null).map(r => r.rules_r);
  const gaps = ok.filter(r => !intradaySetup(r)), intra = ok.length - gaps.length;
  const sys = PREFS.best.side === 'bounce'
    ? `system had ${Math.round(100 * gaps.filter(r => r.sys_grade === 'A' || r.sys_grade === 'A+').length / Math.max(1, gaps.length))}% at A or better at the open` +
      (intra ? ` (${intra} intraday capitulation${intra === 1 ? '' : 's'} left out)` : '')
    : `${Math.round(100 * ok.filter(r => r.sys_scenario === 'A').length / Math.max(1, ok.length))}% opened +4% or more`;
  return `${rows.length} trades · ${sys} · rules ${rs.length ? `median ${rr(median(rs))} on the ${rs.length} that fired` : 'never fired'}`;
}
function bestFilters() {
  const f = PREFS.best, side = (S.lib || []).filter(r => r.side === f.side);
  const opts = (vals, cur, all) => `<option value="">${all}</option>` + vals.map(v => `<option value="${esc(v)}"${v === cur ? ' selected' : ''}>${esc(v)}</option>`).join('');
  const setups = [...new Set(side.map(r => r.setup).filter(Boolean))].sort();
  const caps = ['ETF', 'Large', 'Medium', 'Small', 'Micro'].filter(c => side.some(r => r.cap === c));
  return `<div class="filters"><div class="frow"><div class="seg"><button data-bside="bounce" class="${f.side === 'bounce' ? 'on' : ''}">Bounce</button>` +
    `<button data-bside="reversal" class="${f.side === 'reversal' ? 'on' : ''}">Reversal</button></div>` +
    `<div class="chips" role="group" aria-label="Your grade">${['A', 'B', 'C'].map(g => `<button class="chip${f.grades.includes(g) ? ' on' : ''}" data-bgrade="${g}" aria-pressed="${f.grades.includes(g)}">${g}</button>`).join('')}</div></div>` +
    `<div class="frow selects"><select data-bf="cap" aria-label="Cap">${opts(caps, f.cap, 'All caps')}</select>` +
    `<select data-bf="setup" aria-label="Setup">${opts(setups, f.setup, 'All setups')}</select>` +
    `<select data-bf="sort" aria-label="Sort">${[['move', 'By move'], ['newest', 'By date'], ['rules', 'By rules R']].map(([v, l]) =>
      `<option value="${v}"${f.sort === v ? ' selected' : ''}>${l}</option>`).join('')}</select></div></div>`;
}
function bestCard(r) {
  const bounce = r.side === 'bounce';
  const sys = intradaySetup(r) ? '<span class="badge nogo" title="Forms after the open: the open grade doesn’t apply">intraday</span>'
    : bounce ? (r.sys_grade ? gradeBadge(r.sys_grade) : '') : (r.sys_scenario ? `<span class="badge ${r.sys_scenario}">${r.sys_scenario}</span>` : '');
  const move = r.move != null ? `${bounce ? 'open→high' : 'open→low'} ${pct(bounce ? r.move : -r.move, 0)}${r.move_atr != null ? ` (${r.move_atr.toFixed(1)} ATR)` : ''}` : null;
  const rules = !r.ok ? 'no bars' : r.rules_r != null ? `rules ${rr(r.rules_r)}` : 'no rules entry';
  return `<div class="card name${wide() && S.sel.best === r.key ? ' sel' : ''}" role="button" tabindex="0" data-best="${esc(r.key)}">` +
    `<div class="l1"><span class="tk">${esc(r.ticker)}</span><span class="badge mine" title="Your grade">You ${esc(r.grade || '?')}</span>` +
    `${sys ? `<span class="sys">${intradaySetup(r) ? 'setup' : bounce ? 'system' : 'open'}</span>${sys}` : ''}<span class="cap">${esc(r.cap || '')}</span></div>` +
    `<div class="l2 num">${[dayLabelY(r.date), move, rules].filter(Boolean).map(esc).join(' · ')}` +
    `${r.setup ? `<br><span class="muted">${esc(shortSetup(r.setup))}</span>` : ''}</div></div>`;
}
async function renderBest() {
  const list = $('#list');
  if (!S.lib) {
    list.innerHTML = '<div class="empty">Loading…</div>';
    try { S.lib = (await api('/api/library')).rows || []; } catch (e) {
      list.innerHTML = `<div class="empty neg">${esc(e.message)}</div>`; return;
    }
    if (S.route !== 'best') return;
  }
  const linked = S.lib.find(r => r.key === S.sel.best);
  if (linked && linked.side !== PREFS.best.side) Object.assign(PREFS.best, { side: linked.side, setup: '' });   // a link to the other side
  const rows = bestRows();
  if (wide() && rows.length && !linked) S.sel.best = rows[0].key;
  const side = PREFS.best.side === 'bounce' ? 'bounces' : 'reversals';
  list.innerHTML = listHead('Best trades', `Your ${side}`, esc(bestSummary(rows))) + bestFilters() +
    (!S.lib.length ? '<div class="empty">Not built yet. On the mini: python -m method.library</div>'
      : rows.length ? rows.map(bestCard).join('') : '<div class="empty">No trades match these filters.</div>');
  if (wide()) $('.name.sel', list)?.scrollIntoView({ block: 'nearest' });
  renderBestDetail();
}
function renderBestDetail() {
  const r = (S.lib || []).find(x => x.key === S.sel.best), reader = $('#reader');
  if (!r) { setTitle('Best trades', ''); reader.innerHTML = '<div class="col"><div class="empty">Pick a trade.</div></div>'; return; }
  const log = linesCard('Your log', (r.log || []).map(([k, v]) => [esc(k), '', `<span class="num">${esc(v)}</span>`]),
    `<p class="foot-note" style="margin:10px 0 0">From ${r.side === 'bounce' ? 'bounce_data.csv' : 'reversal_data.csv'}. Your grade: ${esc(r.grade || '–')}.</p>`);
  const intra = intradaySetup(r) ? `<div class="card callout warn"><div class="eyebrow">Intraday capitulation</div><div style="margin-top:6px">This setup forms after the open, so the grade at the open and the open-based rules don't apply. Judge it on the intraday flush.</div></div>` : '';
  showReview({ side: r.side, tk: r.ticker, date: r.date, cap: r.cap, title: esc(r.ticker),
    sub: [dayLabelY(r.date), r.side, `your grade ${r.grade || '–'}`, r.cap].filter(Boolean).join(' · '), cards: () => intra + log });
}

// ── events ───────────────────────────────────────────────────────────────
document.addEventListener('click', e => {
  const card = e.target.closest('.name');
  if (card) {
    if (card.dataset.best != null) return go('best', card.dataset.best, wide());
    if (card.dataset.ep != null) return go('history', card.dataset.ep, wide());
    return go(card.dataset.side, card.dataset.tk, wide());
  }
  const t = e.target.closest('button');
  if (!t) return;
  if (t.dataset.btk) return go('bounce', t.dataset.btk, wide());
  if (t.dataset.review) return go('history', S.sel.history, false, t.dataset.review);
  if (t.dataset.pick) return go(t.dataset.pick, t.dataset.tk, wide());
  if (t.dataset.back != null) return back();
  if (t.dataset.more) { S.showAll[t.dataset.more] = !S.showAll[t.dataset.more]; return render(); }
  if (t.dataset.ctf) {                 // timeframe: redraw the chart only
    PREFS.tf[t.dataset.scope] = t.dataset.ctf; savePrefs();
    $$('[data-ctf]').forEach(b => b.classList.toggle('on', b === t));
    return S.redraw?.();
  }
  if (t.dataset.expand) {
    PREFS.expanded = !PREFS.expanded; savePrefs();
    t.closest('.chart-card')?.classList.toggle('expanded', PREFS.expanded);
    t.textContent = PREFS.expanded ? 'Smaller' : 'Bigger';
    return;
  }
  if (t.dataset.bside) {
    if (PREFS.best.side !== t.dataset.bside) Object.assign(PREFS.best, { side: t.dataset.bside, setup: '' });
    savePrefs(); return render();
  }
  if (t.dataset.bgrade) {
    const g = t.dataset.bgrade, gs = PREFS.best.grades;
    PREFS.best.grades = gs.includes(g) ? gs.filter(x => x !== g) : [...gs, g].sort();
    savePrefs(); return render();
  }
  if (t.dataset.lside) { S.lessonSide = t.dataset.lside; return render(); }
  if (t.dataset.shot) return postShot({ side: S.route === 'playbook' ? S.lessonSide : S.route, ticker: t.dataset.tk, kind: t.dataset.shot });
  if (t.dataset.undo) return postShot({ kind: 'undo' });
  if (t.id === 'push-enable') return enablePush();
  if (t.id === 'push-test') return api('/api/push/test', { method: 'POST', body: '{}' })
    .then(r => { $('#push-status').textContent = r.sent ? 'Test sent.' : 'No device is subscribed.'; });
});
document.addEventListener('change', e => {
  const el = e.target;
  if (el.dataset.ov) { PREFS.ov[el.dataset.ov] = el.checked; savePrefs(); return S.redraw?.(); }
  if (el.dataset.bf) { PREFS.best[el.dataset.bf] = el.value; savePrefs(); return render(); }
});
document.addEventListener('keydown', e => {
  if (e.target.matches?.('input, select, textarea')) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'Enter' && e.target.classList?.contains('name')) return e.target.click();
  const i = ROUTES.indexOf(S.route);
  if (/^[1-5]$/.test(e.key)) return go(ROUTES[Number(e.key) - 1]);
  if (e.key === 'ArrowRight') { e.preventDefault(); return go(ROUTES[(i + 1) % ROUTES.length]); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); return go(ROUTES[(i + ROUTES.length - 1) % ROUTES.length]); }
  if (e.key === 'Escape') return back();
  if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
  e.preventDefault();
  const step = e.key === 'ArrowDown' ? 1 : -1;
  if (S.route === 'best') {
    const rows = bestRows();
    if (!rows.length) return;
    const at = rows.findIndex(r => r.key === S.sel.best);
    return go('best', rows[Math.max(0, Math.min(rows.length - 1, at + step))].key, true);
  }
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

// iOS 26 home-screen app with a translucent status bar: the viewport ends short of the screen by
// the top inset (59pt on a Pro), so the tab bar already sits clear of the home indicator. Tell the
// CSS how short it is so it stops padding for the indicator on top of that.
function fitViewport() {
  const full = innerHeight > innerWidth ? Math.max(screen.width, screen.height) : Math.min(screen.width, screen.height);
  const short = navigator.standalone === true ? Math.min(Math.max(0, full - innerHeight), 120) : 0;   // iOS only
  document.documentElement.style.setProperty('--vshort', `${short}px`);
}
fitViewport();
addEventListener('resize', fitViewport);

if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(() => {});
applyRoute();
(async function pollLoop() {        // every 15 s while the PC streams Trillium, else every minute
  await poll();
  setTimeout(pollLoop, S.snap?.src === 'T' ? 15e3 : 60e3);
})();
setInterval(renderStatus, 30e3);
