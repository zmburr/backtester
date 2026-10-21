"""Offline integration checks, runnable with production Python (no pytest needed)."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import Mock, patch


def validate(project, stage=None):
    project = Path(project).resolve()
    sys.path.insert(0, str(project))
    source = Path(stage).resolve() if stage else project
    tracked = ['data/signal_outcomes.csv', 'data/signal_analysis_state.json']
    before = {p: hashlib.sha256((project/p).read_bytes()).hexdigest() for p in tracked}
    for name in ['scripts.signal_decision_evidence', 'scripts.signal_scorecard', 'scripts.priority_report']:
        path = Path(*name.split('.')).with_suffix('.py')
        module = types.ModuleType(name)
        module.__file__, module.__package__ = str(project/path), 'scripts'
        sys.modules[name] = module
        exec(compile((source/path).read_text(encoding='utf-8'), str(project/path), 'exec'), module.__dict__)
        setattr(__import__('scripts'), name.split('.')[-1], module)
    from scripts import priority_report as pr, signal_scorecard as sc, signal_decision_evidence as evidence
    from analyzers.bounce_scorer import BouncePretrade
    checks = []
    metrics = dict(pct_from_200mav=-.1, selloff_total_pct=-.35, pct_off_30d_high=-.60,
                   gap_pct=-.05, prior_day_range_atr=1.5, pct_change_3=-.25, pct_off_52wk_high=None)
    result = BouncePretrade().validate('TEST', metrics, cap='Medium')
    item = dict(ticker='TEST', bucket='bounce', cap='Medium', rec=result.recommendation,
                score_str=f'{result.score}/{result.max_score}', score_result=result, metrics=metrics)
    e = evidence.build_decision_evidence(item)
    assert e['setup_profile'] == 'GapFade_weakstock' and e['score'] == 5
    assert len(e['rule_files']) == 1
    assert e['criteria'][-1]['actual'] is None and not e['criteria'][-1]['input_present']
    assert all(k in e['inputs'] for k in metrics)
    checks.append('actual profile, exact criteria, missing inputs, rule source version')
    item['decision_evidence'] = e
    with tempfile.TemporaryDirectory(prefix='signal-evidence-check-') as tmp:
        tmp = Path(tmp)
        pr._save_signals_to_json([item], 1, 0, out_dir=tmp/'surfaced', enrich=False)
        path = next((tmp/'surfaced').glob('*.json'))
        payload = json.loads(path.read_text())
        assert payload['signals'][0]['decision_evidence'] == e
        payload['date'], payload['session'] = '2026-10-01', 'morning'
        path.write_text(json.dumps(payload))
        with contextlib.ExitStack() as stack:
            for name, folder in [('SIGNAL_DIR', 'surfaced'), ('CONTROL_DIR', 'controls'), ('BREADTH_DIR', 'breadth')]:
                if hasattr(sc, name): stack.enter_context(patch.object(sc, name, tmp/folder))
            sig = next(iter(sc.collect_signals().values()))
        assert json.loads(sig['decision_evidence']) == e
        checks.append('report JSON to collected scorecard evidence')
        rows = sc.load_outcomes()
        with patch.object(sc, 'OUTCOMES_FILE', tmp/'roundtrip.csv'):
            sc.save_outcomes(rows)
            reread = sc.load_outcomes()
        for key, row in rows.items():
            assert all(reread[key].get(k) == v for k, v in row.items())
            assert all(reread[key][k] == '' for k in evidence.DECISION_COLUMNS if k not in row)
        checks.append(f'{len(rows)} historical rows preserved through extended CSV schema')
        with contextlib.ExitStack() as stack:
            previous_dir = Path.cwd()
            os.chdir(tmp)
            stack.callback(os.chdir, previous_dir)
            for obj, name, value in [
                (pr.ss, 'watchlist', ['TEST']), (pr.ss, 'get_all_stocks_data', lambda _: {'TEST': {}}),
                (pr.gr, 'ReportCache', Mock()), (pr.gr, 'get_pretrade_metrics', lambda *a: {}),
                (pr.gr, 'route_playbook', lambda *a, **k: ('reversal', '', False)),
                (pr.gr, 'get_ticker_cap', lambda *a: 'Medium'),
                (pr.gr, 'score_pretrade_setup', lambda *a, **k: dict(score=0, max_score=5, recommendation='NO-GO', criteria=[])),
                (pr, 'log_signals', Mock()), (pr, '_send_report', Mock()),
                (pr, '_SIGNAL_DIR', tmp/'quiet_surfaced'), (pr, '_CONTROL_DIR', tmp/'quiet_controls')]:
                stack.enter_context(patch.object(obj, name, value))
            pr.generate_priority_report()
        quiet = json.loads(next((tmp/'quiet_controls').glob('*.json')).read_text())
        assert quiet['signals'][0]['decision_evidence']['recommendation'] == 'NO-GO'
        assert json.loads(next((tmp/'quiet_surfaced').glob('*.json')).read_text())['signals'] == []
        checks.append('quiet report records NO-GO evidence and empty surfaced file; sender stubbed')
    after = {p: hashlib.sha256((project/p).read_bytes()).hexdigest() for p in tracked}
    assert before == after
    return dict(checks=checks, production_data_unchanged=True, hashes=after)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--staged-root')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = validate(args.project_root, args.staged_root)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
