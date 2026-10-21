"""Serialize the actual report decision without re-running or changing scoring."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISION_COLUMNS = ['decision_id', 'setup_profile', 'rule_version', 'decision_evidence']


def _plain(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, 'item'):
        return _plain(value.item())
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return str(value)


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


@lru_cache(maxsize=4)
def _rule_files(bucket):
    names = (['analyzers/bounce_scorer.py'] if bucket == 'bounce' else
             ['scripts/generate_report.py', 'analyzers/reversal_scorer.py', 'data/reversal_data.csv'])
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in names if (ROOT / name).exists()}


def build_decision_evidence(item, *, premarket_rvol_threshold=None, evaluated_at=None):
    """Keep missing inputs distinct from numerical zero and failed conditions.

    The time is captured immediately after this ticker is scored. For other
    callers, omitted time means recorded now; the original scorer's naive
    timestamp is retained separately when available.
    """
    bucket = item.get('bucket')
    if bucket not in ('bounce', 'reversal'):
        return None
    metrics = _plain(item.get('metrics') or {})
    scored = item.get('score_result')
    criteria, gates, classification = [], {}, {}
    setup = None
    if bucket == 'bounce' and scored is not None:
        setup = scored.setup_type
        classification = _plain(scored.classification_details)
        for c in scored.items:
            value = metrics.get(c.name)
            criteria.append({'key': c.name, 'description': c.description,
                'operator': '>=' if c.name == 'prior_day_range_atr' else '<=',
                'threshold': c.threshold, 'actual': value,
                'input_present': value is not None, 'passed': bool(c.passed)})
        score, maximum = scored.score, scored.max_score
    elif bucket == 'reversal' and isinstance(scored, dict):
        # This report uses generic thresholds and post-score gates. Its display
        # setup matcher is a different model, never the scoring profile.
        setup = 'generic_reversal_pretrade'
        score, maximum = scored.get('score'), scored.get('max_score')
        for c in scored.get('criteria', []):
            entry = dict(c)
            if c['key'] == 'vol_signal':
                values = [metrics.get('prior_day_rvol'), metrics.get('premarket_rvol')]
                entry.update(operator='any', input_present=any(v is not None for v in values),
                    alternatives=[{'key': 'prior_day_rvol', 'operator': '>=',
                                   'threshold': c['threshold'], 'actual': values[0]},
                                  {'key': 'premarket_rvol', 'operator': '>=',
                                   'threshold': premarket_rvol_threshold, 'actual': values[1]}])
            else:
                entry.update(operator='>=', input_present=metrics.get(c['key']) is not None)
            criteria.append(entry)
        gates = {k: v for k, v in scored.items() if k in (
            'range_gate_passed', 'momentum_pctile_passed', 'momentum_pctile_3',
            'archetype_passed', 'archetype_detail')}
    else:
        score = maximum = None
    files = _rule_files(bucket)
    # The actual thresholds and gate outcomes remain recorded even if rules or
    # reference data are later edited; the hash identifies their source version.
    version = hashlib.sha256(_canonical(files).encode()).hexdigest()
    evidence = _plain({'schema_version': 1, 'producer': 'priority_report',
        'evaluated_at': evaluated_at or datetime.now().astimezone().isoformat(),
        'scorer_timestamp': getattr(scored, 'timestamp', None),
        'ticker': item.get('ticker'), 'bucket': bucket, 'cap': item.get('cap'),
        'setup_profile': setup, 'rule_version': version, 'rule_files': files,
        'status': 'scored' if scored is not None else 'not_scored',
        'score': score, 'max_score': maximum, 'recommendation': item.get('rec'),
        'inputs': metrics, 'classification': classification, 'criteria': criteria, 'gates': gates})
    evidence['decision_id'] = hashlib.sha256(_canonical(evidence).encode()).hexdigest()
    return evidence


def scorecard_decision_fields(signal):
    """Absent historical evidence stays absent; never invent it from today's rules."""
    evidence = signal.get('decision_evidence')
    if not isinstance(evidence, dict) or evidence.get('schema_version') != 1:
        return {name: '' for name in DECISION_COLUMNS}
    return {'decision_id': evidence.get('decision_id', ''),
            'setup_profile': evidence.get('setup_profile') or '',
            'rule_version': evidence.get('rule_version', ''),
            'decision_evidence': _canonical(evidence)}
