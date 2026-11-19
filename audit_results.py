"""Coverage helpers used by the comparison runners.

Publication adaptation: retain the two scoring functions used by the runners;
remove standalone diagnostic demonstrations, rule probes, and report generation.
The retained function bodies are unchanged from the experiment source.
"""
import re
from pathlib import Path
from pii_demo import score
from pii_demo.domain import Action, Decision

ROOT = Path(__file__).resolve().parent


def applied(decisions):
    """Exactly the overlap-resolution order in upstream report._released."""
    out=[]
    for doc in sorted({d.doc for d in decisions}):
        pos=0
        for d in sorted((d for d in decisions if d.doc==doc),key=lambda d:(d.start,-(d.end-d.start))):
            if d.start < pos:
                continue
            out.append(d)
            pos=d.end
    return out


def coverage_audit(result):
    docs=result['documents']
    ds=[Decision.model_validate(d) for d in result['decisions']]
    actual=applied(ds)
    gold=score.load_gold(ROOT/'corpus'/'gold.json',docs)
    hits,extras=score.score(ds,gold)
    checked_correct=0
    audit=[]
    residual=[]
    for h in hits:
        overlap=[d for d in actual if d.doc==h.doc and d.start<h.end and h.start<d.end]
        positions={i for i in range(h.start,h.end) if docs[h.doc][i].isalnum()}
        covered={i for d in overlap if d.action==h.expected for i in range(d.start,d.end)}
        if h.expected==Action.KEEP:
            passed=all(d.action==Action.KEEP for d in overlap)
        else:
            passed=positions.issubset(covered)
            if h.expected==Action.GENERALIZE:
                passed=passed and any(d.action==h.expected and d.start<=h.start and d.end>=h.end
                    and (d.replacement or '').strip()==h.replacement_expected for d in overlap)
        if h.outcome=='correct' and passed:
            checked_correct+=1
        if h.outcome=='correct' and not passed:
            audit.append({'doc':h.doc,'text':h.text,'expected':h.expected.value,
                          'actually_applied':[d.model_dump(mode='json') for d in overlap]})
        if h.expected in (Action.REDACT,Action.ESCALATE):
            removed={i for d in overlap if d.action!=Action.KEEP for i in range(d.start,d.end)}
            exposed=sorted(positions-removed)
            if exposed:
                residual.append({'doc':h.doc,'text':h.text,'expected':h.expected.value,
                                 'unchanged_alphanumeric_characters':len(exposed)})
    # Lenient comparison uses the specific equivalences acknowledged in upstream summary.
    lenient=0
    for h in hits:
        ok=h.outcome=='correct'
        if h.expected==Action.GENERALIZE and h.got==Action.GENERALIZE:
            got=(h.replacement_got or '').strip()
            want=h.replacement_expected
            if want in ('000**','049**'):
                ok=got in (want,want[:3])
            elif want and re.fullmatch(r'\d{4}',want):
                # Allow prose around the year, but never retained month/day/digits.
                ok=re.findall(r'\d+',got)==[want]
            elif want=='90+':
                ok=got.lower() in ('90+','90 or older','90 or over')
        if h.text=='1934-08-22' and h.got==Action.GENERALIZE:
            ok=(h.replacement_got or '').strip().lower() in ('90+','90 or older','90 or over')
        lenient+=bool(ok)
    return {'coverage_checked_strict_correct':checked_correct,
            'upstream_lenient_correct':lenient,
            'upstream_correct_but_failed_coverage':audit,
            'identifier_or_review_spans_with_unchanged_characters':residual,
            'actually_applied_decisions':len(actual),
            'escalated_decisions':sum(d.action==Action.ESCALATE for d in actual)}
