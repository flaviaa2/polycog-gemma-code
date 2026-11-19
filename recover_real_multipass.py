"""Continue the time-matched arm after a truncated extraction, charging all work.

Post-protocol amendment: up to three attempts per extraction/decision call for
token-limit/schema errors. Prompts, settings, scheduling and review policy are
unchanged. The initial failed run and its full cost remain in the evidence.
"""
import datetime,json,time,traceback
from pathlib import Path
import run_real_comparison as test
from resume_real_comparison import status,read

BASE=test.BASE
SOURCE=test.RUNS/'multipass-101'
OUT=test.RUNS/'multipass-101-recovered'
OLD=read(SOURCE/'results.json')
SAVED=[json.loads(x) for x in (SOURCE/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
EVENTS=[]
OriginalRunner=test.timed.Runner


class RecoveringRunner(OriginalRunner):
    def __init__(self):
        super().__init__()
        self.started-=OLD['wall_seconds']
        self.deadline=self.started+test.timed.BUDGET

    def saved_call(self,stage,doc,prompt,schema_type):
        row=SAVED[self.calls]
        schema=schema_type.model_json_schema()
        expected={'model':test.timed.MODEL,'stream':True,'keep_alive':'30m',
            'messages':[{'role':'user','content':prompt},
                        {'role':'user','content':'Return only JSON matching this schema:\n'+json.dumps(schema)}],
            'format':schema,'options':{'seed':test.timed.SEED+self.calls+1,'temperature':.2,'num_ctx':8192,'num_predict':8192}}
        assert expected==row['request'] and stage==row['stage'] and doc==row['doc'], 'Saved request differs; refusing replay'
        self.calls+=1
        raw=row['response']
        self.completed+=1
        self.server_seconds+=raw.get('total_duration',0)/1e9
        self.input_tokens+=raw.get('prompt_eval_count',0)
        self.output_tokens+=raw.get('eval_count',0)
        record=dict(row,replayed_from=str(SOURCE/'requests.jsonl'),original_wall_seconds=row['wall_seconds'],wall_seconds=0.0)
        with (OUT/'requests.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        print(f'[replay] {stage} {doc}: original call {self.calls}; charged original {row["wall_seconds"]:.1f}s',flush=True)
        if raw.get('done_reason')=='length':
            raise ValueError('Model output reached token limit; revision not accepted')
        return schema_type.model_validate_json(raw['message']['content'])

    def call(self,stage,doc,prompt,schema_type):
        attempts=0
        while True:
            attempts+=1
            try:
                if self.calls<len(SAVED):return self.saved_call(stage,doc,prompt,schema_type)
                return super().call(stage,doc,prompt,schema_type)
            except ValueError as exc:
                if stage not in ('extract','decide') or attempts>=3:raise
                event=dict(stage=stage,doc=doc,failed_call=self.calls,attempt=attempts,
                    elapsed_seconds=time.perf_counter()-self.started,error=repr(exc),
                    action='Retry identical prompt/schema/settings with the next normal call seed; all cost charged to same budget.')
                EVENTS.append(event)
                test.save(OUT/'retry_events.json',EVENTS)
                print(f'RETRY {stage} {doc}: attempt {attempts} failed, {self.remaining():.1f}s remaining',flush=True)


def main():
    assert len(SAVED)==4 and all(x.get('accepted') for x in SAVED[:3])
    assert SAVED[3]['response']['done_reason']=='length'
    assert OLD['final_checkpoint'] is None
    assert not OUT.exists()
    frozen=read(BASE/'protocol.json')
    for rel,want in frozen['file_sha256'].items():assert test.sha(test.ROOT/rel)==want,rel
    budget=read(BASE/'timed_budget.json')['budget_seconds']
    amendment=dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reason='The original Gemma-only workflow stopped before decisions because its fourth extraction reached the token limit.',
        change='Bounded retries only: at most three attempts per extraction or decision request after token-limit/schema errors. Reviews keep their existing failure behavior.',
        unchanged='All prompts, schemas, model settings, next-call seed progression, extraction/decision/review schedule and latest-valid-output selection.',
        total_budget_seconds=budget,prior_active_seconds_charged=OLD['wall_seconds'],
        remaining_budget_at_restart=budget-OLD['wall_seconds'],
        saved_responses='Replay three accepted extractions and the original rejected fourth response; retry the fourth with the next seed. No additional budget.',
        original_failure='runs/multipass-101/results.json',
        original_requests_sha256=test.sha(SOURCE/'requests.jsonl'),
        recovery_wrapper_sha256=test.sha(Path(__file__)))
    test.save(BASE/'multipass_recovery_protocol.json',amendment)
    events=read(BASE/'execution_events.json')
    events.append(dict(recorded_utc=amendment['registered_utc'],event=amendment['reason'],
        action=amendment['change'],budget='Every initial/retried request, including the failed response, remains charged to the same 2743-second allowance. Setup pause excluded.'))
    test.save(BASE/'execution_events.json',events)
    status('multipass','running',result_path='runs/multipass-101-recovered/results.json',
           initial_attempt='runs/multipass-101/results.json',amendment='multipass_recovery_protocol.json')
    test.unload()
    test.timed.ROOT=BASE
    test.timed.OUT=OUT
    test.timed.BUDGET=float(budget)
    test.timed.Runner=RecoveringRunner
    print(f'RECOVERING multipass with {budget-OLD["wall_seconds"]:.1f}s remaining in original {budget}s allowance',flush=True)
    test.timed.main()
    result=read(OUT/'results.json')
    test.supplementary(result)
    result['recovery']=amendment
    result['retry_events']=EVENTS
    result['reused_model_calls']=len(SAVED)
    result['fresh_model_calls']=result['attempted_calls']-len(SAVED)
    test.save(OUT/'results.json',result)
    assert result['final_checkpoint'] is not None, 'Recovery failed before completing an initial decision pass'
    assert result['wall_seconds']>=budget-2, 'Recovery exited before its fixed budget was consumed'
    status('multipass','complete',result_path='runs/multipass-101-recovered/results.json',
           initial_attempt='runs/multipass-101/results.json',amendment='multipass_recovery_protocol.json')
    print('RECOVERED COMPARISON COMPLETE; original failure retained and charged',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        test.save(BASE/'multipass_recovery_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:test.unload()
