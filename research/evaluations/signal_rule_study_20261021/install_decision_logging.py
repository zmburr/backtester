"""Install the hash-checked logging patch on the active Mac; leave outcomes alone."""
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/Users/zacharyburr/.openclaw/workspace/backtester')
WORK = ROOT/'research/evaluations/signal_rule_study_20261021'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def run():
    manifest = json.loads((WORK/'logging_manifest.json').read_text())
    checked = json.loads((WORK/'pre_install_validation.json').read_text())
    assert checked['production_data_unchanged']
    for name, digest in checked['hashes'].items():
        assert sha(ROOT/name) == digest, f'Data changed since validation: {name}'
    active = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True)
    assert not any(('python' in line.lower()) and any(s in line for s in
        ['scripts/priority_report.py', '-m scripts.priority_report', 'scripts/signal_scorecard.py', '-m scripts.signal_scorecard'])
        for line in active.splitlines()), 'Wait for active report/scorecard writer'
    for name, hashes in manifest.items():
        assert sha(ROOT/name) == hashes['before'], f'Live code changed: {name}'
        assert sha(WORK/'staged'/name) == hashes['after'], f'Staged code changed: {name}'
    backup = WORK/'backup'
    for name, hashes in manifest.items():
        if hashes['before'] is not None:
            path = backup/name
            path.parent.mkdir(parents=True, exist_ok=True)
            assert not path.exists(), f'Backup already exists: {path}'
            shutil.copy2(ROOT/name, path)
    installed = []
    try:
        # The helper must exist before either consumer is replaced.
        order = ['scripts/signal_decision_evidence.py', 'scripts/signal_scorecard.py', 'scripts/priority_report.py']
        for name in order:
            path = ROOT/name
            temp = path.with_name(path.name+'.evidence-install-tmp')
            shutil.copy2(WORK/'staged'/name, temp)
            os.replace(temp, path)
            installed.append(name)
        for name, hashes in manifest.items():
            assert sha(ROOT/name) == hashes['after']
    except BaseException:
        for name in reversed(installed):
            if manifest[name]['before'] is None:
                (ROOT/name).unlink()
            else:
                shutil.copy2(backup/name, ROOT/name)
        raise
    for name, digest in checked['hashes'].items():
        assert sha(ROOT/name) == digest
    result = dict(installed_at=datetime.now(timezone.utc).isoformat(), files=manifest,
                  production_data_unchanged=True, backup=str(backup))
    (WORK/'installed.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
