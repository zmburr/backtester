'use strict';
// Method Practice: old bounce and reversal days, played blind (method/practice.py serves them).
//   #/practice          home: your stats, Play a day, Quick reads
//   #/practice/play     one session on the tape with your real hotkeys (practice_engine.js scores it)
//   #/practice/reads    ten-second cards: a chart frozen at a decision point, one question
// Blind: no ticker, no date, prices rebased so the prior close = 100, until the reveal.
// Uses app.js's globals ($, esc, api, rr, px, hhmm, resample, vwapDays, sessionWindow, CH, setTitle).

const PE = window.PracticeEngine;
const P = {
  blind: (() => { try { return localStorage.getItem('method.practice.blind') !== '0'; } catch (e) { return true; } })(),
  queue: { play: [], reads: [] },       // keys dealt and not yet played
  days: {},                              // key -> payload (the next few preload, so a weak signal still plays)
  game: null, day: null, phase: null, tf: '2', speed: 1, timer: null, chart: null, started: 0,
  cards: [], card: null, cardDay: null, answered: null, readRun: { n: 0, right: 0 },
  stats: null, pending: [],
};
const SPEEDS = [1, 4, 16];                // bars a second on Play
const FLAG_WORD = {
  early: ['Bought before the trigger', 'Shorted before the trigger'], add_no_break: 'Added without a fresh 2-min break',
  added_to_loser: 'Added to a loser', third_try: 'A 3rd try after 2 stop-outs', early_exit: 'Sold before +1.5 ATR',
  early_cover: 'Covered before 12:30 with no 4-ATR flush', over_budget: 'Put on more risk than your budget',
  budget_vs_breadth: 'Budget above what the day’s breadth supports',
};
const flagWord = (type, side) => { const w = FLAG_WORD[type]; return Array.isArray(w) ? w[side === 'bounce' ? 0 : 1] : (w || type); };
const SIDE_WORD = { bounce: 'Bounce long', reversal: 'Short the spike' };
const KEYS = { bounce: [['D', 1, 1, 'Buy 1R'], ['F', 1, 3, 'Buy 3R'], ['J', -1, 3, 'Sell 3R'], ['K', -1, 1, 'Sell 1R']],
  reversal: [['D', 1, 1, 'Cover 1R'], ['F', 1, 3, 'Cover 3R'], ['J', -1, 3, 'Short 3R'], ['K', -1, 1, 'Short 1R']] };

// ── data ─────────────────────────────────────────────────────────────────
// A blind day keeps the scale it was played on (prior close = 100) through the reveal, so the
// chart and the stop prices you answered don't jump; `realPx` is the reveal's "Real prices" tap.
function scaleOf(day) { return P.blind && !day.realPx && day.prior_close ? 100 / day.prior_close : 1; }
function pxNote(d) {
  if (!P.blind || !d.prior_close) return '';
  return `<div class="off-chart pr-pxnote">${d.realPx ? 'Real prices.' : `Prices as you saw them: prior close ${px(d.prior_close)} = 100.`}
    <button class="btn small ghost" data-prealpx>${d.realPx ? 'As you saw them' : 'Real prices'}</button></div>`;
}
function dayMarker(d) {             // the daily after a reveal runs on past the day: mark it
  const long = d.side === 'bounce';
  return { time: d.date, position: long ? 'belowBar' : 'aboveBar', color: '#a594ff', shape: long ? 'arrowUp' : 'arrowDown', text: 'this day' };
}
const sp = (day, x) => x == null ? null : x * scaleOf(day);
function scaleBars(day, bars) {
  const f = scaleOf(day);
  if (f === 1) return bars;
  return bars.map(b => ({ ...b, open: b.open * f, high: b.high * f, low: b.low * f, close: b.close * f, vw: b.vw != null ? b.vw * f : undefined }));
}
async function nextKey(mode) {
  if (P.queue[mode].length < 3) {
    try { const r = await api(`/api/practice/next?mode=${mode}&n=10`); P.queue[mode].push(...r.keys); P.deckN = r.deck; } catch (e) { /* offline: use what's queued */ }
  }
  return P.queue[mode].shift() || null;
}
async function loadDay(key) {
  if (!P.days[key]) P.days[key] = api(`/api/practice/day?key=${encodeURIComponent(key)}`);
  try { return await P.days[key]; } catch (e) { delete P.days[key]; throw e; }
}
function preload(mode) { P.queue[mode].slice(0, 3).forEach(k => { loadDay(k).catch(() => {}); }); }
function forget(key) { delete P.days[key]; }
async function postLog(entry) {
  P.pending.push(entry);
  try { localStorage.setItem('method.practice.pending', JSON.stringify(P.pending)); } catch (e) { /* private mode */ }
  const left = [];
  for (const e of P.pending) {
    try { await api('/api/practice/log', { method: 'POST', body: JSON.stringify(e) }); } catch (err) { left.push(e); }
  }
  P.pending = left;
  try { localStorage.setItem('method.practice.pending', JSON.stringify(left)); } catch (e) { /* */ }
  P.stats = null;
}
try { P.pending = JSON.parse(localStorage.getItem('method.practice.pending') || '[]'); } catch (e) { P.pending = []; }

// ── one chart, kept between steps ────────────────────────────────────────
// Created once per view and fed with setData each step (a fresh chart a bar would stutter at 16x).
// Intraday views span 9:00-16:00 from the first bar (whitespace for the future), like a live screen.
class PChart {
  constructor(el, daily, blind = false) {
    el.innerHTML = '';
    const host = document.createElement('div');
    host.className = 'chart-host';
    el.appendChild(host);
    this.daily = daily;
    this.c = LightweightCharts.createChart(host, {
      autoSize: true,
      layout: { background: { type: 'solid', color: 'transparent' }, textColor: CH.text, fontFamily: 'Inter, system-ui, sans-serif', fontSize: 11, attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: CH.grid } },
      rightPriceScale: { borderVisible: false, minimumWidth: 58, scaleMargins: { top: 0.08, bottom: 0.16 } },
      timeScale: { borderVisible: false, timeVisible: !daily, secondsVisible: false, rightOffset: daily ? 2 : 0, minBarSpacing: 0.05, fixLeftEdge: !daily },
      crosshair: { mode: 0, vertLine: { color: '#3a434b', labelBackgroundColor: '#1a1f24' }, horzLine: { color: '#3a434b', labelBackgroundColor: '#1a1f24' } },
      handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    });
    // Blind: the axis and the crosshair never show a date (intraday shows the clock, daily nothing).
    const clock = t => typeof t === 'number' ? hhmm(t) : '';
    if (blind) this.c.applyOptions({ localization: { timeFormatter: clock }, timeScale: { tickMarkFormatter: clock } });
    else if (!daily) this.c.applyOptions({ timeScale: { tickMarkFormatter: clock } });
    this.k = this.c.addCandlestickSeries({ upColor: CH.up, downColor: CH.down, wickUpColor: CH.up, wickDownColor: CH.down,
      borderVisible: false, priceLineVisible: false, lastValueVisible: true });
    this.v = this.c.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol', lastValueVisible: false, priceLineVisible: false });
    this.c.priceScale('vol').applyOptions({ scaleMargins: { top: 0.86, bottom: 0 } });
    this.w = this.c.addLineSeries({ color: '#e5c07b', lineWidth: 2, lastValueVisible: false, priceLineVisible: false,
      crosshairMarkerVisible: false, autoscaleInfoProvider: () => null });
    this.lines = [];
    this.fitted = false;
  }
  set(bars, { lines = [], markers = [], vwap = [], span = null, follow = 0 } = {}) {
    const data = bars.map(({ time, open, high, low, close }) => ({ time, open, high, low, close }));
    if (span && bars.length) {             // whitespace to the end of the session
      const last = bars[bars.length - 1].time;
      for (let t = last + span.step; t <= span.end; t += span.step) data.push({ time: t });
    }
    this.k.setData(data);
    this.v.setData(bars.map(b => ({ time: b.time, value: b.value || 0, color: b.close >= b.open ? 'rgba(70,185,138,.4)' : 'rgba(217,97,95,.4)' })));
    this.w.setData(vwap);
    this.lines.forEach(l => this.k.removePriceLine(l));
    this.lines = lines.filter(l => l.price != null).map(l => this.k.createPriceLine({ price: l.price, color: l.color, lineWidth: 1,
      lineStyle: l.style ?? 2, axisLabelVisible: true, title: l.title || '' }));
    const times = bars.map(b => b.time);
    this.k.setMarkers(markers.filter(m => times.length && m.time >= times[0]).map(m => ({ ...m, time: snapTime(m.time, times) }))
      .sort((a, b) => a.time - b.time));
    if (follow && bars.length) {           // 1m / 2m: the last stretch of the tape, like a live screen
      const last = bars.length - 1;
      this.c.timeScale().setVisibleLogicalRange({ from: Math.max(-2, last - follow), to: last + Math.round(follow / 3) });
      this.fitted = true;
      return;
    }
    if (!this.fitted) {
      this.fitted = true;
      const fit = () => this.daily ? this.c.timeScale().setVisibleLogicalRange({ from: bars.length - 60, to: bars.length + 2 }) : this.c.timeScale().fitContent();
      fit();
      requestAnimationFrame(() => requestAnimationFrame(fit));
    }
  }
  remove() { try { this.c.remove(); } catch (e) { /* gone */ } }
}

// The bars a view shows: intraday = 9:00 to now on the timeframe; daily = the days before plus
// today's candle so far (no future daily bars until the reveal).
function viewBars(day, tf, upto, full = false) {
  const candles = scaleBars(day, day.candles || []);
  const dayStart = Math.floor((candles[0]?.time || 0) / 86400) * 86400;
  if (tf === 'D') {
    const before = scaleBars(day, (day.daily || []).filter(b => b.time < day.date))
      .map(b => ({ ...b, time: b.time }));
    const rth = candles.filter(b => PE.tod(b.time) >= 570 && PE.tod(b.time) < 960 && (full || b.time <= upto));
    const after = full ? scaleBars(day, (day.daily || []).filter(b => b.time > day.date)) : [];
    if (!rth.length) return [...before, ...after];
    const today = { time: day.date, open: rth[0].open, high: Math.max(...rth.map(b => b.high)), low: Math.min(...rth.map(b => b.low)),
      close: rth[rth.length - 1].close, value: rth.reduce((s, b) => s + (b.value || 0), 0) };
    return [...before, today, ...after];
  }
  const m = Number(tf);
  const shown = sessionWindow(candles.filter(b => full || b.time <= upto), false);
  return { bars: resample(shown, m), span: { step: m * 60, end: dayStart + 959 * 60 }, m };
}

// ── home ─────────────────────────────────────────────────────────────────
async function renderPractice(sub) {
  if (sub === 'play') return renderPlay();
  if (sub === 'reads') return renderReads();
  stopTimer();
  P.chart?.remove(); P.chart = null;
  setTitle('Practice', 'Old days, played blind: your rules on days you won’t remember');
  const reader = $('#reader');
  reader.innerHTML = `<div class="col">${homeHtml()}</div>`;
  if (!P.stats) {
    try { P.stats = await api('/api/practice/stats'); } catch (e) { return; }
    if (S.route === 'practice' && !parseHash().arg) reader.innerHTML = `<div class="col">${homeHtml()}</div>`;
  }
}
function homeHtml() {
  const s = P.stats;
  const modes = `<div class="pr-modes">
    <button class="card pr-mode" data-pgo="play"><div class="eyebrow">Execution</div><h3>Play a day</h3>
      <div class="muted">The whole session on the tape. Your keys: <kbd>⇧D</kbd> <kbd>⇧F</kbd> buy 1R / 3R, <kbd>⇧K</kbd> <kbd>⇧J</kbd> sell 1R / 3R. Scored against your rules.</div></button>
    <button class="card pr-mode" data-pgo="reads"><div class="eyebrow">Recognition</div><h3>Quick reads</h3>
      <div class="muted">A chart frozen at a decision point and one question. About ten seconds a card; made for the phone.</div></button></div>`;
  const blind = `<label class="tog pr-blind"><input type="checkbox" data-pblind${P.blind ? ' checked' : ''}><span>Blind: hide the ticker and date, rebase prices to 100</span></label>`;
  if (!s) return modes + blind + '<div class="empty">Loading your stats…</div>';
  if (!s.plays && !s.reads) return modes + blind + `<div class="card"><div class="eyebrow">Your stats</div><div class="muted small-line">Nothing played yet. Your rule breaks, R against the rules, and quick-read scores show up here.</div></div>`;
  const flags = Object.entries(s.flags || {}).filter(([, f]) => f.days).sort((a, b) => b[1].days - a[1].days);
  const tr = k => { const d = s.trend?.[k]; return d == null ? '' : d < -0.01 ? ' <span class="go">↓</span>' : d > 0.01 ? ' <span class="neg">↑</span>' : ''; };
  const flagRows = flags.length ? flags.map(([k, f]) => `<div class="line"><div>${esc(f.label)}</div><div class="r">${f.days} of ${s.recent} days${tr(k)}</div></div>`).join('')
    : `<div class="line"><div class="go">No rule breaks in your last ${s.recent} days</div><div class="r"></div></div>`;
  const TYPE = { setup: 'Which setup', size: 'What the day is worth', trigger: 'Is this the trigger', stop: 'Where’s the stop', manage: 'Hold or sell', works: 'Will it work' };
  const reads = Object.entries(s.by_type || {}).map(([k, v]) => `<div class="line"><div>${esc(TYPE[k] || k)}</div><div class="r">${v.right}/${v.n} · ${Math.round(100 * v.right / v.n)}%</div></div>`).join('');
  const w = s.works;
  const works = w ? `<div class="muted small-line">“Will it work?”: you ${Math.round(100 * w.you)}% right; always saying yes would be ${Math.round(100 * w.always_yes)}%, always no ${Math.round(100 * w.always_no)}% (${w.n} cards).</div>` : '';
  return modes + blind + `
    <div class="kpis">${[['Days played', s.plays], ['Clean days', s.recent ? `${s.clean_days}/${s.recent}` : '–'],
      ['You', s.recent ? rr(s.you_r) : '–', s.you_r >= 0 ? 'go' : 'neg'], ['The rules', s.recent ? rr(s.rules_r) : '–', s.rules_r >= 0 ? 'go' : 'neg']]
      .map(([k, v, c]) => `<div class="kpi"><div class="eyebrow">${k}</div><b class="num ${c || ''}">${esc(v)}</b></div>`).join('')}</div>
    ${s.recent ? `<div class="card"><div class="eyebrow">Rule breaks, last ${s.recent} days played</div><div class="lines">${flagRows}</div></div>` : ''}
    ${reads ? `<div class="card"><div class="eyebrow">Quick reads (${s.reads})</div><div class="lines">${reads}</div>${works}</div>` : ''}`;
}

// ── play ─────────────────────────────────────────────────────────────────
async function renderPlay(fresh = false) {
  const reader = $('#reader');
  if (!P.day || fresh) {
    stopTimer();
    P.game = null; P.phase = 'loading';
    reader.innerHTML = '<div class="col"><div class="empty">Dealing a day…</div></div>';
    setTitle('Practice', 'Dealing…');
    let day = null;
    for (let tries = 0; tries < 4 && !day; tries++) {
      const key = await nextKey('play');
      if (!key) break;
      try { const d = await loadDay(key); if (d.ok && d.candles?.length) day = d; else forget(key); } catch (e) { /* next */ }
    }
    preload('play');
    if (!day) { reader.innerHTML = '<div class="col"><div class="empty neg">Couldn’t deal a day. Check the connection, or run the deck build on the mini.</div></div>'; return; }
    P.day = { ...day, revealed: false };
    P.phase = 'plan'; P.budget = null; P.tf = '5';
  }
  drawPlay();
}
function playTitle() {
  const d = P.day;
  if (P.phase === 'reveal' || !P.blind) setTitle(`${esc(d.ticker)} <span class="muted" style="font-weight:600">${esc(dayLabelLong(d.date))}</span>`, `${SIDE_WORD[d.side]} · ${d.source_word}`);
  else setTitle(`${SIDE_WORD[d.side]}`, 'Ticker and date hidden');
}
const dayLabelLong = iso => new Date(`${iso}T12:00:00Z`).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
function dayLine(d) {
  const br = d.side === 'bounce'
    ? (d.day_n != null ? `${d.day_n} names B or better at the open (${STATE_WORD[d.day_state] || d.day_state || '–'})` : 'Breadth: no count before 2022')
    : 'Reversal day: no breadth rule';
  return br;
}
function drawPlay() {
  const d = P.day, g = P.game, reader = $('#reader');
  playTitle();
  const tfSeg = `<div class="seg small">${['D', '5', '2', '1'].map(x => `<button data-ptf="${x}" class="${P.tf === x ? 'on' : ''}">${x === 'D' ? 'Daily' : `${x}m`}</button>`).join('')}</div>`;
  let body = '';
  if (P.phase === 'plan') {
    const allowed = d.side === 'bounce' ? PE.allowedFor(d.day_n) : null;
    body = `<div class="card pr-plan"><div class="eyebrow">Before the open</div>
      <div class="small-line">${esc(dayLine(d))}. Premarket and the daily chart are up. Set your risk budget for the day, as you would the night before.</div>
      <div class="pr-budget">${[0, 1, 3, 6, 10].map(b => `<button class="btn small ${P.budget === b ? 'primary' : 'ghost'}" data-pbudget="${b}">${b ? `${b}R` : 'Pass'}</button>`).join('')}</div>
      <button class="btn primary wide" data-pstart ${P.budget == null ? 'disabled' : ''}>${P.budget === 0 ? 'Pass on this day' : 'Start the day'} <kbd>↵</kbd></button>
      ${allowed != null ? '' : ''}</div>`;
  } else if (P.phase === 'live') {
    body = `<div class="pr-pos" id="pr-pos"></div>
      <div class="pr-keys">${KEYS[d.side].map(([k, dir, u, label]) => `<button class="pr-key ${dir > 0 === (d.side === 'bounce') ? 'in' : 'out'}" data-pkey="${k}"><b>${label}</b><kbd>⇧${k}</kbd></button>`).join('')}</div>
      <div class="pr-transport"><button class="btn small ghost" data-pstep>Step <kbd>␣</kbd></button>
        <button class="btn small ghost" data-pplay>${P.timer ? 'Pause' : 'Play'} <kbd>P</kbd></button>
        <div class="seg small">${SPEEDS.map(s => `<button data-pspeed="${s}" class="${P.speed === s ? 'on' : ''}">${s}×</button>`).join('')}</div>
        <button class="btn small ghost" data-pend>End day <kbd>↵</kbd></button></div>`;
  } else if (P.phase === 'reveal') {
    body = revealHtml();
  }
  reader.innerHTML = `<div class="col pr-col">
    <div class="card chart-card pr-chart-card"><div class="card-head"><span class="eyebrow" id="pr-clock">${P.phase === 'plan' ? 'Premarket' : ''}</span>${tfSeg}</div>
      <div class="chart-wrap"><div class="chart" id="pr-chart"></div></div><div class="off-chart" id="pr-off"></div>${P.phase === 'reveal' ? pxNote(d) : ''}</div>
    ${body}</div>`;
  P.chart?.remove();
  P.chart = new PChart($('#pr-chart'), P.tf === 'D', P.blind && !d.revealed);
  paintPlay();
}
function playMarkers(d, g, real) {
  const out = [];
  const long = d.side === 'bounce';
  for (const f of (g?.fills || [])) {
    const buy = f.dir > 0;
    out.push({ time: f.t, position: buy ? 'belowBar' : 'aboveBar', color: f.kind === 'stop' ? '#ff6363' : '#a594ff',
      shape: buy ? 'arrowUp' : 'arrowDown', text: f.kind === 'stop' ? 'stop' : f.kind === 'close' ? 'close' : `${f.u}R` });
  }
  if (real) {
    (d.rules || []).forEach((t, i) => {
      out.push({ time: t.entry_t + 60, position: long ? 'belowBar' : 'aboveBar', color: '#3ddc97', shape: 'circle', text: `rule ${i + 1}` });
      if (t.kind !== 'open') out.push({ time: t.exit_t + 60, position: long ? 'aboveBar' : 'belowBar', color: t.r_live >= 0 ? '#3ddc97' : '#ff6363', shape: 'circle', text: rr(t.r_live) });
    });
    (d.fills || []).forEach(f => {
      if (f.start) out.push({ time: f.start, position: f.side > 0 ? 'belowBar' : 'aboveBar', color: '#ffb547', shape: 'square', text: 'you' });
    });
  }
  return out;
}
function paintPlay() {
  const d = P.day, g = P.game, full = P.phase === 'reveal';
  const rth = PE.rthBars(d.candles);
  const upto = full ? Infinity : g && g.i >= 0 ? g.bars[g.i].time : (rth[0]?.time ?? 0) - 1;
  const lines = [{ price: sp(d, d.prior_close), color: '#6f7873', title: 'prior close' }];
  if (d.side === 'bounce' && d.session?.pm_low != null) lines.push({ price: sp(d, d.session.pm_low), color: '#ffb547', title: 'PM low', style: 1 });
  const v = g ? PE.view(g) : null;
  if (v) lines.push({ price: sp(d, v.stop), color: '#ff6363', title: 'stop', style: 0 }, { price: sp(d, v.avg), color: '#a594ff', title: 'avg', style: 1 });
  if (P.tf === 'D') {
    P.chart.set(viewBars(d, 'D', upto, full), { lines: lines.slice(0, 1), markers: full ? [dayMarker(d)] : [] });
  } else {
    const vb = viewBars(d, P.tf, upto, full);
    const markers = playMarkers(d, g, full).map(m => ({ ...m, time: Math.floor(m.time / (vb.m * 60)) * vb.m * 60 }));
    const follow = full || !g ? 0 : { 1: 100, 2: 70 }[vb.m] || 0;
    P.chart.set(vb.bars, { lines, markers, vwap: vwapDays(vb.bars).flat(), span: full || !g ? null : vb.span, follow });
  }
  const clock = $('#pr-clock');
  if (clock && g && g.i >= 0) clock.textContent = `${hhmm(g.bars[g.i].time + 60)}${g.done ? ' · closed' : ''}`;
  const pos = $('#pr-pos');
  if (pos && g) {
    const real = g.realized;
    pos.innerHTML = v
      ? `<b class="${v.openR >= 0 ? 'go' : 'neg'}">${g.long ? 'Long' : 'Short'} ${v.units}R</b> · ${v.shares.toLocaleString()} sh · avg ${px(sp(d, v.avg))} · stop ${px(sp(d, v.stop))} · open ${rr(v.openR)}${v.armed ? ' · <span class="go">+1.5 ATR reached</span>' : ''}<span class="muted"> · day ${rr(real)}</span>`
      : `<span class="muted">Flat · day ${rr(real)} · ${g.tries} ${g.tries === 1 ? 'try' : 'tries'}, ${g.stops} stopped · ${g.riskUsed}R of ${g.budget}R used</span>`;
  }
}
function startDay() {
  if (P.budget == null) return;
  P.game = PE.newGame(P.day, P.budget);
  P.started = Date.now();
  if (P.budget === 0) { PE.finish(P.game); return reveal(); }
  P.phase = 'live';
  PE.step(P.game, 1);
  if (P.tf === 'D') P.tf = '2';
  drawPlay();
}
// One step = through the next close on the chart's timeframe, by the clock: a thin name has
// missing minutes, and a step must still end on a 2-min close (where the rules fill).
function stepPlay(m) {
  const g = P.game;
  if (!g || P.phase !== 'live') return;
  const s = m * 60, now = g.i >= 0 ? g.bars[g.i].time + 60 : g.bars[0].time;
  const edge = (Math.floor(now / s) + 1) * s;
  let moved = 0;
  while (g.i < g.bars.length - 1 && g.bars[g.i + 1].time < edge && !g.done) { PE.step(g, 1); moved += 1; }
  if (!moved) PE.step(g, 1);
  if (P.game.done) return reveal();
  paintPlay();
}
function stopTimer() { if (P.timer) { clearInterval(P.timer); P.timer = null; } }
function togglePlay() {
  if (P.phase !== 'live') return;
  if (P.timer) stopTimer();
  else P.timer = setInterval(() => stepPlay(1), 1000 / P.speed);
  const b = $('[data-pplay]');
  if (b) b.innerHTML = `${P.timer ? 'Pause' : 'Play'} <kbd>P</kbd>`;
}
function hotkey(k) {
  if (P.phase !== 'live' || !P.game) return;
  const spec = KEYS[P.day.side].find(x => x[0] === k);
  if (!spec) return;
  const res = PE.press(P.game, spec[1], spec[2]);
  if (res.ignored) toast(res.ignored);
  paintPlay();
}
function toast(msg) {
  let t = $('#pr-toast');
  if (!t) { t = document.createElement('div'); t.id = 'pr-toast'; t.className = 'pr-toast'; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add('on');
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('on'), 1800);
}
function reveal() {
  stopTimer();
  const d = P.day, g = P.game;
  PE.finish(g);
  P.phase = 'reveal';
  d.revealed = true;
  const sc = PE.score(g);
  const real = (d.fills || []).reduce((s, f) => s + (f.pnl || 0), 0);
  P.result = { ...sc, your_real_r: d.fills?.length ? real / (d.one_r || 3000) : null };
  postLog({ mode: 'play', key: d.key, year: d.year, source: d.source, side: d.side, blind: P.blind,
    secs: Math.round((Date.now() - P.started) / 1000), ...P.result });
  if (P.tf === 'D') P.tf = '2';
  drawPlay();
}
function revealHtml() {
  const d = P.day, s = P.result;
  const kp = [['You', rr(s.you_r), s.you_r >= 0 ? 'go' : 'neg'], ['The rules', rr(d.rules_r), d.rules_r >= 0 ? 'go' : 'neg'],
    ['Risk used', `${s.risk_used}R of ${s.budget}R`]];
  if (s.your_real_r != null) kp.splice(2, 0, ['Your real day', rr(s.your_real_r), s.your_real_r >= 0 ? 'go' : 'neg']);
  const flags = s.flags.length ? s.flags.map(f => `<div class="line"><div class="neg">${esc(flagWord(f.type, d.side))}<small>${esc(f.note || '')}</small></div><div class="r">${f.t ? hhmm(f.t) : 'open'}</div></div>`).join('')
    : '<div class="line"><div class="go">Clean: no rule breaks.</div><div class="r"></div></div>';
  const rules = (d.rules || []).length ? d.rules.map((t, i) => `<div class="line"><div>Rule ${i + 1}: ${d.side === 'bounce' ? 'buy' : 'short'} ${hhmm(t.entry_t + 120)} @ ${px(sp(d, t.entry))}<small>${t.kind === 'stop' ? 'stopped' : t.kind === 'trail' ? 'trail exit' : t.kind === 'tripwire' ? 'tripwire cover' : 'held to the close'} ${hhmm(t.exit_t + 120)} @ ${px(sp(d, t.exit))}</small></div><div class="r">${rr(t.r_live)}</div></div>`).join('')
    : '<div class="line"><div class="muted">The rules took no trade.</div><div class="r"></div></div>';
  const yours = (d.fills || []).length ? `<div class="card"><div class="eyebrow">What you really did</div><div class="lines">${d.fills.map(f => `<div class="line"><div>${f.side > 0 ? 'Long' : 'Short'} ${Math.abs(f.size || 0).toLocaleString()} sh @ ${px(sp(d, f.entry))}<small>${f.start ? hhmm(f.start) : ''}–${f.end ? hhmm(f.end) : ''} · out @ ${px(sp(d, f.exit))}</small></div><div class="r ${f.pnl >= 0 ? 'go' : 'neg'}">${rr((f.pnl || 0) / (d.one_r || 3000))}</div></div>`).join('')}</div></div>` : '';
  const grade = d.your_grade ? ` You graded it ${esc(d.your_grade)}.` : '';
  const sys = d.at_open?.grade ? ` The system graded the open ${esc(gradeLabel(d.at_open.grade))}.` : '';
  return `<div class="kpis dense">${kp.map(([k, v, c]) => `<div class="kpi"><div class="eyebrow">${k}</div><b class="num ${c || ''}">${esc(v)}</b></div>`).join('')}</div>
    <div class="card"><div class="eyebrow">Rule breaks</div><div class="lines">${flags}</div></div>
    <div class="card"><div class="eyebrow">What the rules did (R on your sizing: stop floored at ${d.floor_atr} ATR)</div><div class="lines">${rules}</div>
      <div class="muted small-line">${esc(dayLine(d))}.${esc(sys)}${esc(grade)} ${esc(d.setup_class ? `Setup: ${d.setup_class}.` : '')}</div></div>
    ${yours}
    <div class="pr-next"><button class="btn primary" data-pnext>Next day <kbd>↵</kbd></button><button class="btn ghost" data-phome>Practice home</button></div>`;
}

// ── quick reads ──────────────────────────────────────────────────────────
async function renderReads(fresh = false) {
  stopTimer();
  if (!P.card || fresh) {
    $('#reader').innerHTML = '<div class="col"><div class="empty">Dealing cards…</div></div>';
    setTitle('Quick reads', 'Dealing…');
    if (!P.cards.length) await refillCards();
    P.card = P.cards.shift() || null;
    P.answered = null;
    if (!P.card) { $('#reader').innerHTML = '<div class="col"><div class="empty neg">Couldn’t deal cards. Check the connection.</div></div>'; return; }
    P.cardDay = { ...(await loadDay(P.card.key)), revealed: false };
  }
  drawCard();
}
async function refillCards() {
  for (let tries = 0; tries < 3 && P.cards.length < 4; tries++) {
    const keys = [];
    for (let i = 0; i < 3; i++) { const k = await nextKey('reads'); if (k) keys.push(k); }
    const days = await Promise.all(keys.map(k => loadDay(k).catch(() => null)));
    days.forEach((d, i) => {
      if (!d?.ok || !d.cards?.length) return;
      // two cards a day at most, so a session moves across many days
      const pick = [...d.cards].sort(() => Math.random() - 0.5).slice(0, 2);
      pick.forEach(c => P.cards.push({ ...c, key: keys[i] }));
    });
  }
  P.cards.sort(() => Math.random() - 0.5);
  preload('reads');
}
function drawCard() {
  const c = P.card, d = P.cardDay, a = P.answered;
  if (a) { d.revealed = true; setTitle(`${esc(d.ticker)} <span class="muted" style="font-weight:600">${esc(dayLabelLong(d.date))}</span>`, `${SIDE_WORD[d.side]} · ${d.source_word}`); }
  else setTitle('Quick reads', `${SIDE_WORD[d.side]} · ${P.blind ? 'ticker and date hidden · ' : `${d.ticker} ${d.date} · `}card ${P.readRun.n + 1}`);
  const tf = P.cardTf || c.tf;
  const choices = c.type === 'stop'
    ? [...c.choices].sort((x, y) => c.prices[y] - c.prices[x]).map((lab, i) => ({ lab, text: `${'ABCD'[i]} · ${px(sp(d, c.prices[lab]))}` }))
    : c.choices.map(lab => ({ lab, text: lab }));
  P.cardChoices = choices;
  const btns = choices.map((ch, i) => {
    const cls = !a ? 'ghost' : ch.lab === c.answer ? 'primary' : ch.lab === a ? 'danger' : 'ghost';
    return `<button class="btn ${cls} pr-choice" data-pchoice="${i}" ${a ? 'disabled' : ''}><kbd>${i + 1}</kbd> ${esc(ch.text)}</button>`;
  }).join('');
  const verdict = a ? `<div class="card ${a === c.answer ? 'callout' : 'callout warn'}"><div class="eyebrow">${a === c.answer ? 'Right' : `Not quite: ${esc(c.type === 'stop' ? choices.find(x => x.lab === c.answer)?.text : c.answer)}`}</div>
      <div class="small-line">${esc(c.why)}</div>
      <div class="muted small-line">The rest of the day is on the chart: your rules' trades as circles.</div></div>
      <div class="pr-next"><button class="btn primary" data-pnextcard>Next card <kbd>↵</kbd></button><button class="btn ghost" data-phome>Practice home</button></div>` : '';
  $('#reader').innerHTML = `<div class="col pr-col">
    <div class="card chart-card pr-chart-card"><div class="card-head"><span class="eyebrow">${a ? 'The whole day' : `Frozen at ${hhmm(c.cut)}`}</span>
      <div class="seg small">${['D', '5', '2', '1'].map(x => `<button data-pctf="${x}" class="${tf === x ? 'on' : ''}">${x === 'D' ? 'Daily' : `${x}m`}</button>`).join('')}</div></div>
      <div class="chart-wrap"><div class="chart" id="pr-chart"></div></div>${a ? pxNote(d) : ''}</div>
    <div class="card pr-q"><div class="eyebrow">${esc(dayLine(d))}</div><h3>${esc(c.q)}</h3><div class="pr-choices">${btns}</div></div>
    ${verdict}</div>`;
  P.chart?.remove();
  P.chart = new PChart($('#pr-chart'), tf === 'D', P.blind && !d.revealed);
  const upto = c.cut - 60;                  // the last 1-min bar that had closed at the cut
  const full = !!a;
  if (tf === 'D') { P.chart.set(viewBars(d, 'D', upto, full), { markers: full ? [dayMarker(d)] : [] }); return; }
  const vb = viewBars(d, tf, upto, full);
  const m = vb.m * 60, lines = [{ price: sp(d, d.prior_close), color: '#6f7873', title: 'prior close' }];
  const markers = [];
  if (c.mark != null) markers.push({ time: Math.floor(c.mark / m) * m, position: d.side === 'bounce' ? 'belowBar' : 'aboveBar', color: '#a594ff', shape: d.side === 'bounce' ? 'arrowUp' : 'arrowDown', text: '?' });
  if (c.type === 'stop') choices.forEach((ch, i) => lines.push({ price: sp(d, c.prices[ch.lab]), color: a && ch.lab === c.answer ? '#3ddc97' : '#aab2ad', title: 'ABCD'[i], style: 2 }));
  if (c.type === 'manage') lines.push({ price: sp(d, c.entry), color: '#a594ff', title: 'entry', style: 1 }, { price: sp(d, c.stop), color: '#ff6363', title: 'stop', style: 0 });
  if (full) markers.push(...playMarkers(d, null, true).map(x => ({ ...x, time: Math.floor(x.time / m) * m })));
  const span = full ? null : { step: vb.span.step, end: Math.min(vb.span.end, c.cut + 45 * 60) };   // a little room to the right, not the whole day
  P.chart.set(vb.bars, { lines, markers, vwap: vwapDays(vb.bars).flat(), span });
}
function answerCard(i) {
  const c = P.card, ch = P.cardChoices?.[i];
  if (!c || P.answered || !ch) return;
  P.answered = ch.lab;
  P.readRun.n += 1;
  if (ch.lab === c.answer) P.readRun.right += 1;
  postLog({ mode: 'read', key: c.key, card: c.type, chosen: ch.lab, answer: c.answer, correct: ch.lab === c.answer,
    year: P.cardDay.year, source: P.cardDay.source, side: P.cardDay.side, blind: P.blind });
  drawCard();
}
async function nextCard() {
  P.card = null; P.answered = null; P.cardTf = null;
  if (P.cardDay) forget(P.cardDay.key);
  await renderReads(true);
}

// ── input ────────────────────────────────────────────────────────────────
// Returns true when the arena took the key (app.js asks first while the route is practice).
function practiceKey(e) {
  const sub = parseHash().arg;
  if (e.key === 'Escape') { if (sub) { go('practice'); return true; } return false; }
  if (sub === 'play') {
    if (e.shiftKey && /^Key[DFJK]$/.test(e.code)) { e.preventDefault(); hotkey(e.code.slice(3)); return true; }
    if (e.code === 'Space' || e.key === 'ArrowRight') {
      e.preventDefault();
      if (P.phase === 'plan') startDay(); else stepPlay(P.tf === 'D' ? 2 : Number(P.tf));
      return true;
    }
    if (e.key === 'p' || e.key === 'P') { if (!e.shiftKey || P.phase === 'live') togglePlay(); return true; }
    if (e.key === 'Enter') {
      e.preventDefault();
      if (P.phase === 'plan') startDay(); else if (P.phase === 'live') reveal(); else if (P.phase === 'reveal') renderPlay(true);
      return true;
    }
    if (P.phase === 'plan' && /^[0-4]$/.test(e.key)) { P.budget = [0, 1, 3, 6, 10][Number(e.key)]; drawPlay(); return true; }
    return /^[1-9]$/.test(e.key) || e.key === 'ArrowLeft';
  }
  if (sub === 'reads') {
    if (/^[1-4]$/.test(e.key)) { answerCard(Number(e.key) - 1); return true; }
    if ((e.key === 'Enter' || e.code === 'Space') && P.answered) { e.preventDefault(); nextCard(); return true; }
    return /^[5-9]$/.test(e.key) || e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.code === 'Space';
  }
  return false;
}
document.addEventListener('click', e => {
  if (S.route !== 'practice') return;
  const t = e.target.closest('button');
  if (!t) return;
  if (t.dataset.pgo) return go('practice', t.dataset.pgo);
  if (t.dataset.phome != null) return go('practice');
  if (t.dataset.pbudget != null) { P.budget = Number(t.dataset.pbudget); return drawPlay(); }
  if (t.dataset.pstart != null) return startDay();
  if (t.dataset.pkey) return hotkey(t.dataset.pkey);
  if (t.dataset.pstep != null) return stepPlay(P.tf === 'D' ? 2 : Number(P.tf));
  if (t.dataset.pplay != null) return togglePlay();
  if (t.dataset.pspeed) {
    P.speed = Number(t.dataset.pspeed);
    $$('[data-pspeed]').forEach(b => b.classList.toggle('on', b === t));
    if (P.timer) { stopTimer(); togglePlay(); }
    return;
  }
  if (t.dataset.pend != null) return reveal();
  if (t.dataset.pnext != null) return renderPlay(true);
  if (t.dataset.ptf) { P.tf = t.dataset.ptf; return drawPlay(); }
  if (t.dataset.pctf) { P.cardTf = t.dataset.pctf; return drawCard(); }
  if (t.dataset.prealpx != null) {
    const play = parseHash().arg === 'play', d = play ? P.day : P.cardDay;
    if (!d) return;
    d.realPx = !d.realPx;
    return play ? drawPlay() : drawCard();
  }
  if (t.dataset.pchoice != null) return answerCard(Number(t.dataset.pchoice));
  if (t.dataset.pnextcard != null) return nextCard();
});
document.addEventListener('change', e => {
  if (e.target.dataset?.pblind == null) return;
  P.blind = e.target.checked;
  try { localStorage.setItem('method.practice.blind', P.blind ? '1' : '0'); } catch (err) { /* */ }
});
window.renderPractice = renderPractice;
window.practiceKey = practiceKey;
window.practiceLeave = () => { stopTimer(); };
