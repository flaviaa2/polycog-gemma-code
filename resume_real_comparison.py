"""Resume saved real-document inference without changing the frozen workflows."""
import datetime
import hashlib
import json
import math
import time
import traceback
from dataclasses import asdict

import evaluate_gemma as eg
import run_real_comparison as test
from pydantic_ai.models.function import FunctionModel
from pii_demo import pipeline, score
from pii_demo.domain import Subject

BASE = test.BASE
OUT = test.RUNS / 'cognition-101-resumed'
SOURCE = test.RUNS / 'cognition-101' / 'requests.jsonl'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def status(arm, state, **extra):
    rows = read(BASE / 'status.json')
    for row in rows:
        if row['arm'] == arm:
            row.clear()
            row.update(arm=arm, status=state, **extra)
            break
    else:
        rows.append(dict(arm=arm, status=state, **extra))
    test.save(BASE / 'status.json', rows)


def main():
    protocol = read(BASE / 'protocol.json')
    for rel, expected in protocol['file_sha256'].items():
        assert test.sha(test.ROOT / rel) == expected, f'Frozen input changed: {rel}'
    assert not OUT.exists(), 'Never overwrite a resumption or completed result'
    assert not (test.RUNS / 'multipass-101').exists()
    for arm in ('baseline', 'atomic'):
        assert (test.RUNS / f'{arm}-101' / 'results.json').exists()
    OUT.mkdir()
    events = read(BASE / 'execution_events.json')
    events.append({
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'event': 'User explicitly requested continuation, then resume. Saved successful requests are replayed exactly; missing requests resume locally.',
        'timing': 'Recovered active duration is resumed elapsed time plus original elapsed time of replayed successful requests. User pause, pre-pause orchestration overhead, and any interrupted unsaved request are excluded. This is an approximate time match, not an uninterrupted timing measurement.'})
    test.save(BASE / 'execution_events.json', events)
    transport = eg.OllamaTransport(test.MODEL, test.SEED, .2, OUT / 'requests.jsonl', resume_from=SOURCE)
    expected_replays = len(transport.saved)
    live_api = eg.api

    def checked_api(path, data=None, timeout=300):
        if path == '/api/chat':
            assert transport.reused_calls == expected_replays, 'Cached request sequence diverged; refusing fresh inference'
        return live_api(path, data, timeout)

    eg.api = checked_api
    docs = pipeline.read_packet(test.CORPUS)
    meta = dict(model=test.MODEL, arm='cognition', seed_base=test.SEED,
                source_protocol='../../protocol.json', resumed_from=str(SOURCE),
                source_requests_sha256=test.sha(SOURCE), expected_replays=expected_replays,
                resumption_wrapper_sha256=test.sha(test.ROOT / 'resume_real_comparison.py'),
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                corpus_sha256={n: hashlib.sha256(t.encode()).hexdigest() for n, t in docs.items()})
    test.save(OUT / 'metadata.json', meta)
    status('cognition', 'running', result_path='runs/cognition-101-resumed/results.json')
    print(f'RESUMING cognition: {expected_replays} exact saved responses; frozen hashes verified', flush=True)
    test.unload()
    started = time.perf_counter()
    packet = pipeline.run(test.CORPUS, FunctionModel(transport, model_name=test.MODEL),
                          policy='safe-harbor', trials=3, timeout=300, min_confidence=.6, ref_year=2026)
    fresh_wall = time.perf_counter() - started
    rows = [json.loads(line) for line in (OUT / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    assert transport.reused_calls == expected_replays
    prior_wall = sum(r.get('original_wall_seconds', 0) for r in rows)
    active_wall = fresh_wall + prior_wall
    hits, extras = score.score(packet.decisions, test.real_gold(None, docs))
    result = dict(metadata=meta, documents=docs,
                  decisions=[d.model_dump(mode='json') for d in packet.decisions],
                  upstream_score=score.summarize(hits, extras),
                  hits=[dict(asdict(h), outcome=h.outcome) for h in hits],
                  extras=[d.model_dump(mode='json') for d in extras],
                  wall_seconds=active_wall, resumed_wall_seconds=fresh_wall,
                  replayed_original_wall_seconds=prior_wall,
                  timing_basis=events[-1]['timing'],
                  ollama_server_seconds=sum(r.get('response', {}).get('total_duration', 0) / 1e9 for r in rows),
                  model_calls=transport.calls, reused_model_calls=transport.reused_calls,
                  fresh_model_calls=transport.calls-transport.reused_calls,
                  input_tokens=transport.input_tokens, output_tokens=transport.output_tokens,
                  mentions=[m.model_dump(mode='json') for m in packet.mentions],
                  subjects=[s.model_dump(mode='json') for s in packet.world.entities if isinstance(s, Subject)],
                  considered=packet.considered, notes=packet.notes, timings=packet.timings,
                  timings_scope='Stage timers cover the resumed execution; replayed calls are near-instant. Use recovered wall_seconds or full successful server time for total cost.')
    test.supplementary(result)
    test.save(OUT / 'results.json', result)
    status('cognition', 'complete', result_path='runs/cognition-101-resumed/results.json')
    print(f'COMPLETED cognition: {transport.calls} calls; recovered active time {active_wall:.1f}s', flush=True)
    eg.api = live_api
    budget = max(900, math.ceil(active_wall))
    test.save(BASE / 'timed_budget.json', dict(budget_seconds=budget,
              polycog_wall_seconds=active_wall, rule=protocol['timed_gemma_budget_rule'],
              timing_basis=result['timing_basis'], cognition_results='runs/cognition-101-resumed/results.json'))
    test.unload()
    test.timed.ROOT = BASE
    test.timed.OUT = test.RUNS / 'multipass-101'
    test.timed.BUDGET = float(budget)
    status('multipass', 'running')
    print(f'STARTING multipass: {budget}s fixed budget', flush=True)
    test.timed.main()
    multipass = read(test.timed.OUT / 'results.json')
    test.supplementary(multipass)
    test.save(test.timed.OUT / 'results.json', multipass)
    status('multipass', 'complete')
    print('ALL REAL-DOCUMENT RUNS COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        test.save(BASE / 'resumption_failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        test.unload()
