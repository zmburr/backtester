"""Install the tested Mac repair after checking exact source and backup hashes."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT=Path('/Users/zacharyburr/.openclaw/workspace/backtester')
DEST=ROOT/'research/evaluations/scorecard_repair_20261021'


def sha(data):return hashlib.sha256(data).hexdigest()


def atomic(path,data,mode):
    temp=path.with_name(path.name+'.repair_20261021.tmp')
    with temp.open('wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    temp.chmod(mode)
    os.replace(temp,path)


def run():
    code=json.loads((DEST/'code_manifest.json').read_text())
    data=json.loads((DEST/'data_repair.json').read_text())
    verification=json.loads((DEST/'analysis_verification.json').read_text())
    assert verification['state_unchanged'] and verification['email_guard_passed']
    # Do not replace the outcome store while one of its writers is active.
    processes=subprocess.run(['ps','-ax','-o','pid=,args='],capture_output=True,text=True,check=True).stdout
    for line in processes.splitlines():
        try:
            pid,args=line.strip().split(None,1);parts=shlex.split(args)
        except ValueError:continue
        if not parts or not Path(parts[0]).name.startswith('python'):continue
        jobs={'signal_scorecard.py','signal_analysis.py','backfill_entry_r.py',
              'scripts.signal_scorecard','scripts.signal_analysis','scripts.backfill_entry_r'}
        assert not any(Path(x).name in jobs or x in jobs for x in parts[1:4]),f'Writer active: {pid}'
    planned=[]
    for name,hashes in code.items():
        target=ROOT/'scripts'/name;replacement=(DEST/'staged/scripts'/name).read_bytes()
        assert sha(target.read_bytes())==hashes['before_sha256'],name
        assert sha(replacement)==hashes['after_sha256'],name
        compile(replacement,str(target),'exec')
        planned.append((target,replacement))
    target=ROOT/'data/signal_outcomes.csv';replacement=(DEST/'proposed_signal_outcomes.csv').read_bytes()
    assert sha(target.read_bytes())==data['source_sha256']
    assert sha(replacement)==data['proposed_sha256']
    rows=list(csv.DictReader(io.StringIO(replacement.decode())))
    assert len(rows)==data['rows_after']==data['rows_before']
    assert not any(r['complete'].lower()=='true' and 0<int(r['days_available'] or 0)<4 for r in rows)
    planned.append((target,replacement))
    state=ROOT/'data/signal_analysis_state.json';state_hash=sha(state.read_bytes())
    originals={}
    for target,replacement in planned:
        original=target.read_bytes();mode=target.stat().st_mode
        backup=DEST/'backup'/target.relative_to(ROOT);backup.parent.mkdir(parents=True,exist_ok=True)
        if backup.exists():assert backup.read_bytes()==original
        else:backup.write_bytes(original)
        originals[target]=(original,mode)
    installed=[]
    try:
        for target,replacement in planned:
            assert target.read_bytes()==originals[target][0],f'Concurrent modification: {target}'
            atomic(target,replacement,originals[target][1]);installed.append(target)
        assert sha(state.read_bytes())==state_hash
    except Exception:
        for target in reversed(installed):atomic(target,*originals[target])
        raise
    result={'files':[{'path':str(p),'before_sha256':sha(originals[p][0]),'after_sha256':sha(p.read_bytes())} for p in installed],
            'state_unchanged':True,'outcome_rows':len(rows),'records_repaired':len(data['repairs']),
            'incomplete_history':[r for r in data['repairs'] if not r['complete']]}
    (DEST/'installed.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
