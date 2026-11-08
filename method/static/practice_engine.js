'use strict';
// Method Practice: the play engine. Pure functions over one day's payload (/api/practice/day),
// so the page and the node tests (tests/js/practice_engine.test.js) run the same code.
//
// Time: the tape reveals 1-min RTH bars one at a time. "Now" is the end of the last revealed
// bar and every fill is at its close, so stepping in 2-min bars fills at the 2-min close the
// rules use. Size: one press = 1R or 3R at the stop, where a try's stop is the low of day
// (the high for a short) and the risk a share is |entry - stop| floored at 0.5 ATR (bounce)
// or 0.25 ATR (reversal): the morning watcher's try sizing. A stop fills at the stop price
// the moment a bar trades through it, before anything else that bar does (as the replay does).
(function (root) {
  const tod = t => { const x = new Date(t * 1000); return x.getUTCHours() * 60 + x.getUTCMinutes(); };
  const RTH_START = 570, RTH_END = 960, NOON30 = 750;     // 9:30, 16:00, 12:30

  function rthBars(candles) { return (candles || []).filter(b => tod(b.time) >= RTH_START && tod(b.time) < RTH_END); }

  function allowedFor(n) {                // Bounce Lessons: R a name by breadth
    if (n == null) return null;
    return n >= 16 ? 6 : n >= 11 ? 1 : 0;
  }

  function newGame(day, budget) {
    const bars = rthBars(day.candles);
    const long = day.side === 'bounce';
    const atr = Number(day.atr) || 0;
    const g = {
      side: day.side, long, atr, floor: (day.floor_atr || (long ? 0.5 : 0.25)) * atr, arm: (day.arm_atr || 1.5) * atr,
      flushPx: bars.length ? bars[0].open - (day.flush_atr || 4) * atr : null, oneR: day.one_r || 3000,
      bars, i: -1, budget, trigEnds: (day.triggers || []).map(t => t + 120),
      pos: null, tries: 0, stops: 0, riskUsed: 0, realized: 0, fills: [], flags: [], done: false,
      lod: Infinity, hod: -Infinity, flushed: false, rulesR: day.rules_r || 0, dayN: day.day_n,
    };
    const allowed = long ? allowedFor(day.day_n) : null;
    if (allowed != null && budget > allowed) flag(g, 'budget_vs_breadth', `Budget ${budget}R on a ${day.day_n}-name day (the rule says ${allowed ? `${allowed}R` : 'pass'})`);
    return g;
  }

  function flag(g, type, note) {
    g.flags.push({ type, t: g.i >= 0 ? g.bars[g.i].time + 60 : null, note });
  }
  const now = g => g.bars[g.i].time + 60;
  const price = g => g.bars[g.i].close;
  const units = pos => pos.lots.reduce((s, l) => s + l.u, 0);
  const avgEntry = pos => pos.lots.reduce((s, l) => s + l.u * l.entry, 0) / units(pos);

  // On the trigger: a trigger bar has closed and this is no later than the next bar's close.
  function onTrigger(g) {
    const t = now(g);
    return g.trigEnds.some(e => e <= t && t - e < 120);
  }

  function closeUnits(g, u, px, kind) {
    const pos = g.pos, sign = g.long ? 1 : -1;
    let left = u, r = 0;
    while (left > 1e-9 && pos.lots.length) {
      const lot = pos.lots[0], take = Math.min(lot.u, left);
      r += take * sign * (px - lot.entry) / lot.risk;
      lot.u -= take; left -= take;
      if (lot.u <= 1e-9) pos.lots.shift();
    }
    g.realized += r;
    g.fills.push({ t: g.bars[g.i].time, px, dir: -sign, u: u - left, kind, r });
    if (!pos.lots.length) g.pos = null;
    return r;
  }

  // One step: reveal the next 1-min bar, take a stop it trades through, track the extremes.
  function stepOne(g) {
    if (g.done || g.i >= g.bars.length - 1) return false;
    g.i += 1;
    const b = g.bars[g.i];
    if (g.pos) {
      const hit = g.long ? b.low < g.pos.stop : b.high > g.pos.stop;
      if (hit) {
        closeUnits(g, units(g.pos), g.pos.stop, 'stop');
        g.stops += 1;
      }
    }
    g.lod = Math.min(g.lod, b.low);
    g.hod = Math.max(g.hod, b.high);
    if (g.pos) {
      g.pos.peak = g.long ? Math.max(g.pos.peak, b.high) : Math.min(g.pos.peak, b.low);
      if (!g.pos.armed && (g.long ? g.pos.peak >= g.pos.first + g.arm : g.pos.peak <= g.pos.first - g.arm)) g.pos.armed = true;
      if (!g.long && g.flushPx != null && b.low <= g.flushPx) g.pos.flushed = true;
    }
    if (g.i === g.bars.length - 1) finish(g);
    return true;
  }

  function step(g, n = 1) {
    let k = 0;
    while (k < n && stepOne(g)) k += 1;
    return k;
  }

  // A hotkey: dir +1 buys, -1 sells; u is 1 or 3 (R at the stop).
  function press(g, dir, u) {
    if (g.done || g.i < 0) return { ignored: g.done ? 'done' : 'before the open' };
    const sign = g.long ? 1 : -1, px = price(g);
    if (dir === sign) {                       // open or add
      if (!g.pos) {
        g.tries += 1;
        if (g.stops >= 2) flag(g, 'third_try', 'A new try after two stop-outs');
        if (!onTrigger(g)) flag(g, 'early', g.long ? 'Bought with no 2-min trigger' : 'Shorted with no 2-min trigger');
        const stop = g.long ? g.lod : g.hod;
        g.pos = { stop, lots: [], peak: px, first: px, armed: false, flushed: false, t0: now(g) };
      } else {
        const avg = avgEntry(g.pos);
        if (g.long ? px < avg : px > avg) flag(g, 'added_to_loser', 'Added below your entry');
        else if (!onTrigger(g)) flag(g, 'add_no_break', 'Added with no fresh 2-min break');
      }
      const risk = Math.max(Math.abs(px - g.pos.stop), g.floor) || Math.abs(px) * 0.0025;
      g.pos.lots.push({ u, entry: px, risk, t: now(g) });
      g.riskUsed += u;
      if (g.riskUsed > g.budget && !g.flags.some(f => f.type === 'over_budget')) {
        flag(g, 'over_budget', `${g.riskUsed}R put on against a ${g.budget}R budget`);
      }
      g.fills.push({ t: g.bars[g.i].time, px, dir, u, kind: 'in', shares: Math.round(u * g.oneR / risk) });
      return { ok: true };
    }
    if (!g.pos) return { ignored: g.long ? 'flat: a sell only closes a long on a bounce day' : 'flat: a buy only covers a short on a reversal day' };
    if (g.long && !g.pos.armed && !g.flags.some(f => f.type === 'early_exit' && f.t0 === g.pos.t0)) {
      g.flags.push({ type: 'early_exit', t: now(g), t0: g.pos.t0, note: 'Sold before +1.5 ATR' });
    }
    if (!g.long && tod(now(g)) < NOON30 && !g.pos.flushed && !g.flags.some(f => f.type === 'early_cover' && f.t0 === g.pos.t0)) {
      g.flags.push({ type: 'early_cover', t: now(g), t0: g.pos.t0, note: 'Covered before 12:30 with no 4-ATR flush' });
    }
    closeUnits(g, Math.min(u, units(g.pos)), px, 'out');
    return { ok: true };
  }

  function finish(g) {
    if (g.done) return g;
    if (g.pos && g.i >= 0) closeUnits(g, units(g.pos), price(g), 'close');
    g.done = true;
    return g;
  }

  function view(g) {                          // what the position line shows
    if (!g.pos) return null;
    const u = units(g.pos), sign = g.long ? 1 : -1, px = price(g);
    const open = g.pos.lots.reduce((s, l) => s + l.u * sign * (px - l.entry) / l.risk, 0);
    const shares = g.pos.lots.reduce((s, l) => s + l.u * g.oneR / l.risk, 0);
    return { units: u, avg: avgEntry(g.pos), stop: g.pos.stop, openR: open, shares: Math.round(shares), armed: g.pos.armed };
  }

  function score(g) {
    return { you_r: g.realized, rules_r: g.rulesR, tries: g.tries, stops: g.stops, risk_used: g.riskUsed,
      budget: g.budget, flags: g.flags.map(({ type, t, note }) => ({ type, t, note })), fills: g.fills };
  }

  const api = { rthBars, newGame, step, press, finish, view, score, allowedFor, tod };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PracticeEngine = api;
})(typeof window !== 'undefined' ? window : globalThis);
