"""Post-inference per-document attribution and transparent practical audit."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,html,json,re
from pii_demo.domain import Decision,Action
from audit_results import applied

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'real_documents'
ARMS=['baseline','atomic','cognition','multipass']

def j(p):return json.loads(p.read_text(encoding='utf-8'))
def duration(s):return f'{int(s)//60}m {s%60:04.1f}s'

def gather(selected_arms=None,live_snapshot=None):
    arms=ARMS if selected_arms is None else selected_arms
    labels=j(BASE/'practical_targets.json')
    statuses=j(BASE/'status.json')
    assert len(statuses)==4 and all(s['status']=='complete' or (s['arm']=='multipass' and live_snapshot is not None)
                                  for s in statuses if s['arm'] in arms)
    paths={s['arm']:BASE/s.get('result_path',f"runs/{s['arm']}-101/results.json") for s in statuses}
    runs={arm:(live_snapshot['result'] if arm=='multipass' and live_snapshot else j(paths[arm])) for arm in arms}
    docs=runs[arms[0]]['documents']
    for n,t in docs.items():assert hashlib.sha256(t.encode()).hexdigest()==labels['corpus_sha256'][n]
    audit={'rubric':'practical_targets.json','documents':{n:{} for n in docs},'targets':{},'noise_candidates':{},'totals':{}}
    for arm,r in runs.items():
        ds=applied([Decision.model_validate(x) for x in r['decisions']])
        rows=(live_snapshot['requests'] if arm=='multipass' and live_snapshot else
              [json.loads(x) for x in (paths[arm].parent/'requests.jsonl').read_text(encoding='utf-8').splitlines()])
        timing=defaultdict(lambda:Counter(completed_calls=0,attempted_calls=0,request_wall_seconds=0,server_seconds=0,cutoff_calls=0))
        names=list(docs)
        for i,row in enumerate(rows):
            if arm=='multipass':name=row['doc']
            elif arm in ('baseline','atomic') or i<4:
                name=names[i]
            else:
                prompt='\n'.join(m['content'] for m in row['request']['messages'])
                candidates=[n for n,t in docs.items() if '[document: '+t.splitlines()[0]+']' in prompt]
                assert len(candidates)==1,(arm,i,candidates)
                name=candidates[0]
            assert name in docs
            timing[name].update(completed_calls=int('response' in row),attempted_calls=1,
                request_wall_seconds=row.get('original_wall_seconds',row['wall_seconds']),
                server_seconds=row.get('response',{}).get('total_duration',0)/1e9,
                cutoff_calls=int(bool(row.get('budget_cutoff'))))
        details=[]
        for g in labels['targets']:
            overlaps=[d for d in ds if d.doc==g['doc'] and d.start<g['end'] and g['start']<d.end]
            relevant=[d for d in overlaps if d.action!=Action.KEEP]
            required={i for i in range(g['start'],g['end']) if docs[g['doc']][i].isalnum()}
            covered={i for d in relevant for i in range(d.start,d.end)}
            marked=required.issubset(covered)
            # Report obvious identifier reintroduction, rather than crediting a nominal action.
            replacement=' '.join(d.replacement or '' for d in relevant)
            pattern={'patient_surname':r'\bBiden\b','presidential_role':r'\bPresident\b|\bPOTUS\b',
                     'institutional_context':r'\bWhite House\b|\bWHMU\b'}.get(g['kind'])
            if pattern and re.search(pattern,replacement,re.I):marked=False
            if g['kind']=='explicit_patient_date' and any(d.action==Action.GENERALIZE and d.replacement and
                re.search(r'February|\b16\b',d.replacement,re.I) for d in relevant):marked=False
            details.append(dict(g,marked=marked,partially_marked=bool(relevant) and not marked,
                actions=[d.model_dump(mode='json') for d in relevant]))
        audit['targets'][arm]=details
        for d in ds:
            if d.action==Action.KEEP:continue
            if d.action!=Action.ESCALATE and (d.replacement or '[REDACTED]').strip()==d.text.strip():continue
            related=[g for g in labels['targets'] if g['doc']==d.doc and d.start<g['end'] and g['start']<d.end]
            if related:continue
            payload=d.model_dump(mode='json')
            key=hashlib.sha256(json.dumps(dict(arm=arm,**payload),sort_keys=True).encode()).hexdigest()[:16]
            audit['noise_candidates'][key]=dict(arm=arm,**payload)
        for n in docs:
            x=dict(timing[n]);x['duration_display']=duration(x['request_wall_seconds'])
            for cat in ('obvious','reasonable'):
                own=[g for g in details if g['doc']==n and g['category']==cat]
                x[cat+'_marked']=sum(g['marked'] for g in own)
                x[cat+'_missed']=sum(not g['marked'] for g in own)
            x['review_flags']=sum(d.doc==n and d.action==Action.ESCALATE for d in ds)
            audit['documents'][n][arm]=x
        audit['totals'][arm]={'model_calls':sum(v['attempted_calls'] for v in timing.values()),
            'completed_responses':sum(v['completed_calls'] for v in timing.values()),
            'token_limited_responses':sum(x.get('response',{}).get('done_reason')=='length' for x in rows),
            'token_limited_with_no_answer':sum(x.get('response',{}).get('done_reason')=='length' and not x['response']['message'].get('content') for x in rows),
            'request_wall_seconds':sum(v['request_wall_seconds'] for v in timing.values()),
            'active_wall_seconds':r['wall_seconds'],
            'orchestration_seconds':r['wall_seconds']-sum(v['request_wall_seconds'] for v in timing.values())}
    return audit

def main():
    audit=gather()
    (BASE/'practical_audit_draft.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    judgments_path=BASE/'noise_judgments.json'
    if not judgments_path.exists():
        print(json.dumps(audit['noise_candidates'],indent=2,ensure_ascii=False));return
    judgments=j(judgments_path)
    assert set(audit['noise_candidates'])==set(judgments['decisions']), 'Every candidate requires an explicit judgment'
    for n,arms in audit['documents'].items():
        for arm,x in arms.items():
            own=[judgments['decisions'][key] for key,d in audit['noise_candidates'].items() if d['doc']==n and d['arm']==arm]
            x['clear_noise']=sum(y['category']=='clear_noise' for y in own)
            x['low_risk_noise']=sum(y['category']=='low_risk_noise' for y in own)
            x['context_dependent_flags']=sum(y['category']=='context_dependent' for y in own)
            x['noise_count']=x['clear_noise']+x['low_risk_noise']
    audit['noise_judgments']=judgments
    audit['summary']=judgments['summary']
    audit['interpretation']=judgments['interpretation']
    partial_path=BASE/'user_requested_partial.json'
    if partial_path.exists():
        partial=j(partial_path)
        keys=['obvious_marked','reasonable_marked','noise_count']
        audit['checkpoint_comparison']={
            'source':'user_requested_partial.json',
            'earlier_elapsed_seconds':partial['snapshot']['elapsed_seconds'],
            'earlier':{k:sum(v['multipass'][k] for v in partial['documents'].values()) for k in keys},
            'final':{k:sum(v['multipass'][k] for v in audit['documents'].values()) for k in keys},
            'selection_note':'Earlier checkpoint was captured at the user’s request, not chosen by its score. Final answers still use the last accepted state at the fixed deadline.'}
    (BASE/'practical_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(audit['documents'],indent=2))

if __name__=='__main__':main()
