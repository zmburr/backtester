"""Prepare a backed-up, targeted data repair on the Mac; do not install it."""
import csv
import hashlib
import importlib.util
import io
import json
import logging
from pathlib import Path
import sys

ROOT=Path('/Users/zacharyburr/.openclaw/workspace/backtester')
DEST=ROOT/'research/evaluations/scorecard_repair_20261021'
KEY=('target_date','ticker','bucket','cohort')
sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):return tuple(row[k] for k in KEY)


def load(name):
    spec=importlib.util.spec_from_file_location('repair_'+name,DEST/'staged/scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.PROJECT_ROOT=ROOT
    for attr,rel in [('OUTCOMES_FILE','signal_outcomes.csv'),('SIGNAL_DIR','priority_signals'),
                     ('CONTROL_DIR','control_signals'),('BREADTH_DIR','breadth_signals')]:
        setattr(module,attr,ROOT/'data'/rel)
    return module


def run():
    sc=load('signal_scorecard')
    sc.logger.setLevel(logging.ERROR)
    manifest=json.loads((DEST/'code_manifest.json').read_text())
    for name,hashes in manifest.items():
        assert digest(ROOT/'scripts'/name)==hashes['before_sha256'],name
        assert digest(DEST/'staged/scripts'/name)==hashes['after_sha256'],name
    for rel in ['data/signal_outcomes.csv','data/signal_analysis_state.json',
                'scripts/signal_scorecard.py','scripts/signal_analysis.py']:
        source=ROOT/rel;backup=DEST/'backup'/rel
        backup.parent.mkdir(parents=True,exist_ok=True)
        if backup.exists():assert backup.read_bytes()==source.read_bytes(),rel
        else:backup.write_bytes(source.read_bytes())
    source=ROOT/'data/signal_outcomes.csv';before_hash=digest(source)
    with source.open(newline='') as f:
        reader=csv.DictReader(f);rows=list(reader);header=reader.fieldnames
    before={key(r):dict(r) for r in rows}
    repairs=[];rekeyed=[];client=sc._polygon_client()
    from scripts import backfill_entry_r as entry
    from bounce_entry_study.detectors import build_2min
    for index,old in enumerate(rows):
        target=sc.on_or_next_trading_day(old['signal_date']) if old['session']=='morning' else sc.next_trading_day(old['signal_date'])
        wrong_date=target!=old['target_date']
        partial=old['complete'].lower()=='true' and 0<int(old['days_available'] or 0)<sc.HORIZON_DAYS
        if not (partial or wrong_date):continue
        sig={k:old[k] for k in ['signal_date','session','ticker','bucket','cohort','cap','recommendation','score']}
        sig['target_date']=target
        sig['atr_pct']=float(old['atr_pct']) if old['atr_pct'] else None
        sig['metrics']={k:float(old[k]) for k in sc.METRIC_COLUMNS if old.get(k)}
        window=sc.trading_days_from(target,sc.HORIZON_DAYS)
        assert len(window)==sc.HORIZON_DAYS
        # Use the client directly: exceptions abort this proposal, rather than
        # turning an API failure into a supposedly observed empty window.
        aggs=client.get_aggs(old['ticker'],1,'day',from_=window[0],to=window[-1],limit=50)
        import datetime
        from zoneinfo import ZoneInfo
        bars={datetime.datetime.fromtimestamp(b.timestamp/1000,ZoneInfo('America/New_York')).date().isoformat():
              {'open':b.open,'high':b.high,'low':b.low,'close':b.close} for b in aggs}
        computed=sc.compute_outcome(sig,bars)
        if not wrong_date:sc.preserve_entry_replay(computed,old)
        replay_status='not_applicable'
        if old['bucket']=='bounce' and computed.get('entry_open'):
            minute=entry.fetch_day(old['ticker'],target)
            if minute is not None and not minute.empty:
                regular=minute.between_time('09:30','15:59:59')
                if not regular.empty and abs(float(regular.open.iloc[0])/computed['entry_open']-1)<=.03:
                    computed.update(entry.replay_entry_day(build_2min(minute),sig['atr_pct']*computed['entry_open'] if sig['atr_pct'] else None))
                    replay_status='recomputed'
                else:replay_status='bar_basis_mismatch'
            else:replay_status='missing_minute_bars'
        new={c:computed.get(c,'') for c in header}
        # Episode fields are re-derived below. Every other unselected row stays
        # byte-for-byte equivalent at the CSV field level.
        rows[index]=new
        if wrong_date:rekeyed.append({'old':list(key(old)),'new':list(key(new))})
        repairs.append({'old_key':list(key(old)),'new_key':list(key(new)),
            'old_days':int(old['days_available']),'new_days':computed['days_available'],
            'complete':computed['complete'],'missing_days':[d for d in window if d not in bars],
            'old_open':old['entry_open'],'new_open':computed.get('entry_open'),
            'entry_replay':replay_status})
        print(json.dumps(repairs[-1]),flush=True)
    outcomes={key(r):r for r in rows}
    assert len(outcomes)==len(rows),'Repair would collide with an existing key'
    sc.assign_episodes(outcomes);sc.assign_clusters(outcomes)
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=header)
    writer.writeheader();writer.writerows(rows)
    proposed=DEST/'proposed_signal_outcomes.csv'
    proposed.write_bytes(out.getvalue().encode('utf-8'))
    normalized=list(csv.DictReader(io.StringIO(out.getvalue())))
    repaired_old={tuple(r['old_key']) for r in repairs}
    episode_changes=[]
    for row in normalized:
        k=key(row)
        if k in repaired_old or k not in before:continue
        differences=[c for c in header if row[c]!=before[k][c]]
        assert set(differences)<=set(['episode_id','episode_signal_num','cluster_id','cluster_size']),(k,differences)
        if differences:episode_changes.append({'key':list(k),'columns':differences})
    assert digest(source)==before_hash,'Source changed during proposal'
    report={'source_sha256':before_hash,'proposed_sha256':digest(proposed),
            'rows_before':len(before),'rows_after':len(rows),'repairs':repairs,
            'rekeyed':rekeyed,'episode_changes':episode_changes}
    (DEST/'data_repair.json').write_text(json.dumps(report,indent=2))
    print('PROPOSAL_READY',len(repairs),'records;',sum(r['complete'] for r in repairs),'full windows;',len(episode_changes),'other episode changes')


if __name__=='__main__':run()
