"""Frozen-workflow transfer test on four downloaded, authentic public letters.

All source, annotation and harness hashes are registered before inference.
No annotation is supplied to a model. Scoring is performed after each arm ends.
Upstream source is not edited. The timed workflow is imported unchanged.
"""
from pathlib import Path
import hashlib,json,math,time,datetime,sys,traceback
from dataclasses import asdict
from collections import Counter,defaultdict
from pydantic_ai.models.function import FunctionModel
from pii_demo import baseline,pipeline,score
from pii_demo.domain import Decision,Action,Subject
from evaluate_gemma import OllamaTransport,contextual_baseline,api,save
from audit_results import coverage_audit,applied
import run_budget_gemma as timed

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'real_documents'
CORPUS=BASE/'corpus'
RUNS=BASE/'runs'
MODEL='gemma4:e4b-it-qat'
SEED=101

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def real_gold(path,docs):
    annotations=json.loads((BASE/'annotations.json').read_text(encoding='utf-8'))
    out=[]
    for g in annotations['targets']:
        assert docs[g['doc']][g['start']:g['end']]==g['text']
        out.append((g['doc'],g['start'],g['end'],g))
    return out

# Route only gold-file loading to the new occurrence-level annotations. All
# upstream scoring and inference functions retain their original behavior.
score.load_gold=real_gold

def supplementary(result):
    """Audit actual renderer-applied decisions; do not count partial overlap as safe."""
    ds=applied([Decision.model_validate(d) for d in result['decisions']])
    targets=json.loads((BASE/'annotations.json').read_text(encoding='utf-8'))['targets']
    groups=defaultdict(lambda:Counter(total=0,strict_correct=0,fully_withheld_or_correctly_generalized=0,kept_intact=0))
    details=[]
    hits=result['hits']
    for g,h in zip(targets,hits):
        assert g['doc']==h['doc'] and g['start']==h['start']
        overlap=[d for d in ds if d.doc==g['doc'] and d.start<g['end'] and g['start']<d.end]
        positions={i for i in range(g['start'],g['end']) if result['documents'][g['doc']][i].isalnum()}
        withheld={i for d in overlap if d.action in (Action.REDACT,Action.ESCALATE) or
                    (d.action==Action.GENERALIZE and not d.replacement)
                  for i in range(d.start,d.end)}
        intact=all(d.action==Action.KEEP for d in overlap)
        generalized=any(d.action==Action.GENERALIZE and d.start<=g['start'] and d.end>=g['end']
                        and (d.replacement or '').strip()==g.get('replacement') for d in overlap) if g.get('replacement') else False
        protected=positions.issubset(withheld) or generalized
        groups[g['category']].update(total=1,strict_correct=int(h['outcome']=='correct'),
                     fully_withheld_or_correctly_generalized=int(protected),kept_intact=int(intact))
        details.append({'doc':g['doc'],'text':g['text'],'start':g['start'],'category':g['category'],
                        'expected':g['action'],'strict_outcome':h['outcome'],'kept_intact':intact,
                        'fully_withheld_or_correctly_generalized':protected,
                        'applied':[d.model_dump(mode='json') for d in overlap]})
    result['target_audit']={'categories':dict(groups),'targets':details,
       'scope':'Target-level coverage only, not full-document privacy. Replacement reintroduction and unique contextual re-identification require review.'}
    result['coverage_audit']=coverage_audit(result)
    return result

def one_arm(arm):
    out=RUNS/f'{arm}-101'
    out.mkdir(parents=True,exist_ok=False)
    docs=pipeline.read_packet(CORPUS)
    transport=OllamaTransport(MODEL,SEED,.2,out/'requests.jsonl')
    llm=FunctionModel(transport,model_name=MODEL)
    meta={'model':MODEL,'arm':arm,'seed_base':SEED,'source_protocol':'../../protocol.json',
          'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'corpus_sha256':{n:hashlib.sha256(t.encode()).hexdigest() for n,t in docs.items()}}
    save(out/'metadata.json',meta)
    started=time.perf_counter()
    if arm=='cognition':
        packet=pipeline.run(CORPUS,llm,policy='safe-harbor',trials=3,timeout=300,min_confidence=.6,ref_year=2026)
        decisions=packet.decisions
        extra={'mentions':[m.model_dump(mode='json') for m in packet.mentions],
               'subjects':[s.model_dump(mode='json') for s in packet.world.entities if isinstance(s,Subject)],
               'considered':packet.considered,'notes':packet.notes,'timings':packet.timings}
    elif arm=='baseline':
        decisions,timings=baseline.run_baseline(docs,llm,ref_year=2026,timeout=300)
        extra={'timings':timings}
    elif arm=='atomic':
        decisions,extra=contextual_baseline(docs,llm,atomic=True)
    elapsed=time.perf_counter()-started
    # First annotation content read for this arm happens after inference finishes.
    hits,extras=score.score(decisions,real_gold(None,docs))
    rows=[json.loads(line) for line in (out/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    result={'metadata':meta,'documents':docs,'decisions':[d.model_dump(mode='json') for d in decisions],
         'upstream_score':score.summarize(hits,extras),'hits':[dict(asdict(h),outcome=h.outcome) for h in hits],
         'extras':[d.model_dump(mode='json') for d in extras],'wall_seconds':elapsed,
         'ollama_server_seconds':sum(r.get('response',{}).get('total_duration',0)/1e9 for r in rows),
         'model_calls':transport.calls,'input_tokens':transport.input_tokens,'output_tokens':transport.output_tokens,
         **extra}
    supplementary(result)
    save(out/'results.json',result)
    print(f'COMPLETED {arm}: calls={transport.calls} wall_seconds={elapsed:.1f}',flush=True)
    return result

def unload():
    api('/api/generate',{'model':MODEL,'keep_alive':0})

def main():
    if (BASE/'protocol.json').exists():raise FileExistsError('Protocol already registered; do not overwrite or silently rerun.')
    RUNS.mkdir(exist_ok=True)
    protocol={'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'objective':'Transfer the previously tested workflows unchanged to authentic, downloaded public medical letters.',
      'arms':['baseline','atomic','cognition','multipass'],
      'selection':'Four complete short physician letters about one publicly identified patient; selected for authentic medical content, public provenance and feasible local inference. No selection using model outcomes. This is a convenience sample, not a random clinical sample.',
      'model':MODEL,'seed_base':SEED,'temperature':.2,'num_ctx':8192,'num_predict':8192,'thinking':'default enabled',
      'cognition':{'trials':3,'confidence_threshold':.6,'ref_year':2026,'policy':'safe-harbor','source_changes':'none'},
      'timed_gemma_budget_rule':'max(900 seconds, ceil(completed Polycog wall-clock seconds)). Thus Gemma gets at least as much time and at least the earlier 15 minutes. Budget is set from time only, never from scores.',
      'multipass':'Import the earlier frozen extraction/batched-decision/review workflow unchanged; change only paths, annotation loader and predeclared time budget. Final selection remains latest accepted state, not best score.',
      'gold':'55 targeted occurrence-level expectations fixed before inference. Codex annotations, not independent expert gold. Source files and model prompts contain no gold labels.',
      'controls':'Unload model between arms; same local structured JSON transport. Temporary idle-sleep prevention; no cloud inference.',
      'file_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'evaluate_gemma.py',ROOT/'run_budget_gemma.py',BASE/'annotations.json',BASE/'source_manifest.json',*sorted(CORPUS.glob('*.txt'))]},
      'ollama_version':api('/api/version'),'ollama_model':api('/api/show',{'model':MODEL})}
    save(BASE/'protocol.json',protocol)
    statuses=[]
    poly=None
    for arm in ['baseline','atomic','cognition']:
        unload()
        print('STARTING '+arm,flush=True)
        try:
            result=one_arm(arm)
            if arm=='cognition':poly=result
            statuses.append({'arm':arm,'status':'complete'})
        except Exception as exc:
            error={'arm':arm,'status':'failed','error':repr(exc),'traceback':traceback.format_exc()}
            statuses.append(error);save(RUNS/f'{arm}-101'/'failure.json',error)
            print('FAILED '+arm+': '+repr(exc),flush=True)
        save(BASE/'status.json',statuses)
    if poly is None:
        raise RuntimeError('Polycog did not complete, so the time-matching rule cannot be applied.')
    budget=max(900,math.ceil(poly['wall_seconds']))
    save(BASE/'timed_budget.json',{'budget_seconds':budget,'polycog_wall_seconds':poly['wall_seconds'],
                                   'rule':protocol['timed_gemma_budget_rule']})
    unload()
    timed.ROOT=BASE
    timed.OUT=RUNS/'multipass-101'
    timed.BUDGET=float(budget)
    print(f'STARTING multipass: {budget}s budget',flush=True)
    timed.main()
    r=json.loads((timed.OUT/'results.json').read_text(encoding='utf-8'))
    supplementary(r)
    save(timed.OUT/'results.json',r)
    statuses.append({'arm':'multipass','status':'complete'})
    save(BASE/'status.json',statuses)
    unload()
    print('ALL REAL-DOCUMENT RUNS COMPLETE',flush=True)

if __name__=='__main__':main()
