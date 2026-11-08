// Method Practice play engine. Run: node --test tests/js/  (tests/unit/test_method_practice.py runs it too)
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');
const E = require('../../method/static/practice_engine.js');

const days = JSON.parse(fs.readFileSync(path.join(__dirname, '../unit/fixtures/practice_days.json'), 'utf8'));

function stepTo(g, barTime) {               // reveal through the 1-min bar starting at barTime
  while (g.i < 0 || g.bars[g.i].time < barTime) if (!E.step(g, 1)) break;
  assert.strictEqual(g.bars[g.i].time, barTime);
}

// Pressing exactly the rules' entries (and their stop / tripwire / close exits) reproduces
// the rules' R on the live sizing risk, to the cent.
for (const key of Object.keys(days)) {
  test(`rules replayed by hand = rules R: ${key}`, () => {
    const day = days[key];
    const g = E.newGame(day, 10);
    const sign = day.side === 'bounce' ? 1 : -1;
    let want = 0;
    for (const t of day.rules) {
      if (t.kind === 'trail' || t.half) return;        // fills at a level, not a close: not reproducible by a key
      stepTo(g, t.entry_t + 60);                      // the 2-min entry bar's second minute closes it
      E.press(g, sign, 1);
      assert.ok(!g.flags.some(f => f.type === 'early'), 'a rule entry is on the trigger');
      if (t.kind === 'stop') {
        while (g.pos && E.step(g, 1)) { /* until the stop takes it */ }
        assert.strictEqual(g.pos, null);
      } else if (day.side === 'reversal') {
        stepTo(g, t.exit_t + 60);                     // cover at that 2-min bar's close
        E.press(g, -sign, 1);
      } else {
        E.step(g, 10000);                             // bounce 'close': held to the last bar
      }
      want += t.r_live;
    }
    E.finish(g);
    assert.ok(Math.abs(g.realized - want) < 1e-9, `${g.realized} vs ${want}`);
    assert.ok(Math.abs(want - day.rules_r) < 1e-9);
  });
}

test('a stop fills at the stop: a loss under 1R when the risk is floored', () => {
  const day = days['bounce:SOXL:2026-07-28'];
  const t = day.rules[0];
  const g = E.newGame(day, 10);
  stepTo(g, t.entry_t + 60);
  E.press(g, 1, 3);
  while (g.pos && E.step(g, 1)) { /* */ }
  const risk = Math.max(t.entry - t.stop, 0.5 * day.atr);
  assert.ok(Math.abs(g.realized - 3 * (t.stop - t.entry) / risk) < 1e-9);
  assert.strictEqual(g.stops, 1);
  assert.strictEqual(g.fills[g.fills.length - 1].kind, 'stop');
});

test('buying with no trigger, stacking past the budget, and a sell before +1.5 ATR are flagged', () => {
  const day = days['bounce:SOXL:2026-07-28'];
  const g = E.newGame(day, 3);
  E.step(g, 3);                                       // 9:33, no trigger yet
  E.press(g, 1, 3);
  E.press(g, 1, 1);
  E.step(g, 1);
  E.press(g, -1, 1);
  const types = g.flags.map(f => f.type);
  assert.ok(types.includes('early'));
  assert.ok(types.includes('over_budget'));
  assert.ok(types.includes('early_exit'));
  assert.ok(types.includes('budget_vs_breadth'), '3R on a 14-name day: the rule says 1R');
});

test('a third try after two stop-outs is flagged; a sell when flat does nothing', () => {
  const day = days['bounce:SOXL:2026-07-28'];
  const g = E.newGame(day, 10);
  assert.ok(E.press(g, 1, 1).ignored, 'nothing before the open');
  E.step(g, 1);
  assert.ok(E.press(g, -1, 1).ignored);
  let stops = 0;
  while (stops < 2 && !g.done) {                       // buy each bar until stopped twice
    if (!g.pos) E.press(g, 1, 1);
    E.step(g, 1);
    stops = g.stops;
  }
  assert.strictEqual(g.stops, 2);
  if (g.pos) { while (g.pos && E.step(g, 1)) { /* flat first */ } }
  E.step(g, 1);
  E.press(g, 1, 1);
  assert.ok(g.flags.some(f => f.type === 'third_try'));
});

test('reversal: K shorts, D covers, a cover before 12:30 without a flush is early', () => {
  const day = days['reversal:AI:2023-06-16'];
  const t = day.rules[0];
  const g = E.newGame(day, 3);
  stepTo(g, t.entry_t + 60);
  E.press(g, -1, 1);
  assert.ok(g.pos && g.pos.stop >= t.entry);
  E.step(g, 10);
  E.press(g, 1, 1);
  assert.strictEqual(g.pos, null);
  assert.ok(g.flags.some(f => f.type === 'early_cover'));
});

test('the end of the day closes what is open at the last close', () => {
  const day = days['reversal:AMC:2021-01-27'];
  const g = E.newGame(day, 1);
  E.step(g, 3);
  E.press(g, -1, 1);
  E.step(g, 100000);
  assert.ok(g.done);
  assert.strictEqual(g.pos, null);
  const last = g.fills[g.fills.length - 1];
  assert.ok(last.kind === 'close' || last.kind === 'stop');
});
