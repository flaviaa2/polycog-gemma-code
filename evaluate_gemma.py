"""Local Gemma evaluation; upstream prompts, corpus, rules and scoring are unmodified.

The transport adapter uses Ollama's schema-constrained JSON API for every arm.
FunctionModel delegates every answer to the actual model; it supplies no answers.
No cloud endpoint, API key or gold labels enter a model request.
"""
import argparse
import hashlib
import json
import re
import subprocess
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from pii_demo import baseline, pipeline, score
from pii_demo.domain import Action, Decision, Subject
from pii_demo.reference import RESTRICTED_ZIP3

ROOT = Path(__file__).resolve().parent


def api(path, data=None, timeout=300):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request('http://127.0.0.1:11434'+path, data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


class OllamaTransport:
    def __init__(self, model, seed, temperature, log, resume_from=None):
        self.model, self.seed, self.temperature, self.log = model, seed, temperature, log
        self.calls = 0
        self.input_tokens = self.output_tokens = 0
        self.reused_calls = 0
        self.resume_from = resume_from
        self.saved = {}
        if resume_from:
            for line in resume_from.read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                if 'response' in row and row['response'].get('done_reason')!='length':
                    key=json.dumps(row['request'],sort_keys=True)
                    self.saved[key]=row

    def __call__(self, messages, info):
        self.calls += 1
        wire = []
        if info.instructions:
            wire.append({'role':'system', 'content':info.instructions})
        for message in messages:
            for part in message.parts:
                kind = part.part_kind
                if kind in ('system-prompt', 'user-prompt'):
                    if not isinstance(part.content, str):
                        raise TypeError('This text-only adapter requires string prompts')
                    wire.append({'role':'system' if kind == 'system-prompt' else 'user', 'content':part.content})
                elif kind == 'retry-prompt':
                    wire.append({'role':'user', 'content':str(part.content)})
                elif kind == 'text':
                    wire.append({'role':'assistant', 'content':part.content})
                elif kind == 'tool-call':
                    wire.append({'role':'assistant', 'content':part.args_as_json_str()})
                else:
                    raise ValueError(f'Unsupported message part: {kind}')
        if info.function_tools or len(info.output_tools) > 1:
            raise ValueError('Only one structured output schema is supported')
        output_tool = info.output_tools[0] if info.output_tools else None
        body = {'model':self.model, 'messages':wire, 'stream':False, 'keep_alive':'30m',
                'options':{'seed':self.seed + self.calls, 'temperature':self.temperature,
                           'num_ctx':8192, 'num_predict':8192}}
        if output_tool:
            schema = output_tool.parameters_json_schema
            body['format'] = schema
            wire.append({'role':'user', 'content':'Return only JSON matching this schema:\n'+json.dumps(schema)})
        t0 = time.perf_counter()
        cached=self.saved.get(json.dumps(body,sort_keys=True))
        try:
            if cached:
                raw=cached['response']
                self.reused_calls+=1
            else:
                raw = api('/api/chat', body, timeout=300)
        except Exception as exc:
            with self.log.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'request':body, 'error':repr(exc), 'wall_seconds':time.perf_counter()-t0})+'\n')
            raise
        with self.log.open('a', encoding='utf-8') as f:
            record={'request':body,'response':raw,'wall_seconds':time.perf_counter()-t0}
            if cached:
                record.update(replayed_from=str(self.resume_from),original_wall_seconds=cached['wall_seconds'])
            f.write(json.dumps(record)+'\n')
        self.input_tokens += raw.get('prompt_eval_count',0)
        self.output_tokens += raw.get('eval_count',0)
        content = raw['message']['content']
        if raw.get('done_reason') == 'length':
            raise RuntimeError('Output was truncated')
        part = ToolCallPart(output_tool.name, content) if output_tool else TextPart(content)
        return ModelResponse(parts=[part], usage=RequestUsage(input_tokens=raw.get('prompt_eval_count',0),
                                                             output_tokens=raw.get('eval_count',0)))


def locate_findings(doc, text, findings):
    decisions, unmatched = [], []
    for f in findings:
        pattern = re.compile(r'(?<![\w-])'+re.escape(f.text.strip())+r'(?![\w]|-\w)')
        matches = list(pattern.finditer(text))
        if not matches:
            unmatched.append(f.model_dump(mode='json'))
        for m in matches:
            decisions.append(Decision(doc=doc,text=m.group(),start=m.start(),end=m.end(),action=f.action,
                replacement=f.replacement if f.action is Action.GENERALIZE else
                    ('[REVIEW]' if f.action is Action.ESCALATE else '[REDACTED]' if f.action is Action.REDACT else None),
                rule='baseline-llm',citation="(model's own reading)",rationale=f.reason))
    return decisions, unmatched


def contextual_baseline(docs, llm, atomic=False):
    """Same task/schema, all other documents supplied as context for each target."""
    agent = Agent(llm, output_type=baseline.BaselineReport, model_settings={'timeout':300})
    decisions, unmatched, timings = [], {}, {}
    for name, text in docs.items():
        context = '\n\n'.join(f'== Supporting document: {n} ==\n{t}' for n,t in docs.items() if n != name)
        prompt = ('The following supporting documents are from the SAME packet. Use them to determine '
                  'whether identifiers in the target document are derived from personal information. '
                  'Return findings for the TARGET document only.\n\n'+context+'\n\n== TARGET TASK ==\n'+
                  baseline.PROMPT.format(ref_year=2026,zip3=', '.join(sorted(RESTRICTED_ZIP3)),document=text))
        if atomic:
            prompt += ('\n\nOutput requirements: return ONE ATOMIC ITEM per finding, not whole sentences or '
                       'rows. Split compounds (for example Austin, TX 78701 into Austin, TX, and 78701). '
                       'Consider every name, date, age, location, contact detail, identifier, code and '
                       'distinctive personal fact. Include allowed items as KEEP. For every GENERALIZE '
                       'action you MUST include replacement containing ONLY the allowed coarser value '
                       '(year, ZIP3 followed by **, or 90+). Do not omit replacement. '
                       'Before answering, check all target-document lines for missed items, and check '
                       'each code against names, birth dates and record numbers in the supporting documents.')
        t0=time.perf_counter()
        report = agent.run_sync(prompt).output
        ds, missing = locate_findings(name,text,report.findings)
        decisions.extend(ds)
        unmatched[name]=missing
        timings[name]=time.perf_counter()-t0
        print(f'[context] {name}: {len(report.findings)} findings ({timings[name]:.1f}s)',flush=True)
    return decisions, {'timings':timings,'unmatched':unmatched}


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['baseline','context','atomic','cognition'],required=True)
    p.add_argument('--model',default='gemma3:4b')
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--temperature',type=float,default=0.2)
    p.add_argument('--corpus',type=Path,default=ROOT/'corpus')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume-from',type=Path,help='Reuse only successful, byte-equivalent saved model requests')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():
        raise FileExistsError('Refusing to overwrite a completed result')
    transport=OllamaTransport(args.model,args.seed,args.temperature,args.out/'requests.jsonl',args.resume_from)
    llm=FunctionModel(transport,model_name=args.model)
    docs=pipeline.read_packet(args.corpus)
    metadata={'model':args.model,'arm':args.arm,'seed_base':args.seed,'temperature':args.temperature,
              'num_ctx':8192,'trials':3,'ref_year':2026,'min_confidence':0.6,'policy':'safe-harbor',
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'corpus_sha256':{n:hashlib.sha256(t.encode()).hexdigest() for n,t in docs.items()},
              'ollama_version':api('/api/version'),'ollama_model':api('/api/show',{'model':args.model}),
              'transport':'Ollama /api/chat JSON schema via FunctionModel, shared by all arms',
              'resume_from':str(args.resume_from) if args.resume_from else None,
              'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    save(args.out/'metadata.json',metadata)
    t0=time.perf_counter()
    if args.arm=='cognition':
        packet=pipeline.run(args.corpus,llm,policy='safe-harbor',trials=3,timeout=300,min_confidence=.6,ref_year=2026)
        decisions=packet.decisions
        extra={'mentions':[m.model_dump(mode='json') for m in packet.mentions],
               'subjects':[s.model_dump(mode='json') for s in packet.world.entities if isinstance(s,Subject)],
               'considered':packet.considered,'notes':packet.notes,'timings':packet.timings}
    elif args.arm=='baseline':
        decisions,timings=baseline.run_baseline(docs,llm,ref_year=2026,timeout=300)
        extra={'timings':timings}
    else:
        decisions,extra=contextual_baseline(docs,llm,atomic=args.arm=='atomic')
    wall=time.perf_counter()-t0
    gold=score.load_gold(args.corpus/'gold.json',docs)
    hits,extras=score.score(decisions,gold)
    result={'metadata':metadata,'documents':docs,'decisions':[d.model_dump(mode='json') for d in decisions],
            'upstream_score':score.summarize(hits,extras),'hits':[dict(asdict(h),outcome=h.outcome) for h in hits],
            'extras':[d.model_dump(mode='json') for d in extras],'wall_seconds':wall,
            'model_calls':transport.calls,'input_tokens':transport.input_tokens,
            'output_tokens':transport.output_tokens,'reused_model_calls':transport.reused_calls,
            'fresh_model_calls':transport.calls-transport.reused_calls,**extra}
    save(args.out/'results.json',result)
    print(json.dumps({k:result[k] for k in ['upstream_score','wall_seconds','model_calls','input_tokens','output_tokens']},default=str),flush=True)


if __name__=='__main__':
    main()
