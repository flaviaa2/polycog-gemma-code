# Gemma and Polycog comparison code

Source code for reviewing how a local Gemma model is used with Polycog's PHI agent and in standalone language-model workflows. Prepared by Flavia with Codex to support a technical discussion with the Polycog team.

This repository contains **code, prompts, dependency versions, and setup documentation**. Experiment documents, annotation judgments, recorded model responses, scores, reports, and conclusions are not included.

## Relationship to the Polycog demo

The PHI agent comes from [polycog/demo-phi](https://github.com/polycog/demo-phi), pinned to commit [`0d0fe86c9c10dc1a0123db282870c666f77e2ff3`](https://github.com/polycog/demo-phi/tree/0d0fe86c9c10dc1a0123db282870c666f77e2ff3). The source in [`pii_demo/`](pii_demo/) preserves that version's prompts, rules, and pipeline.

The additional harness connects the workflows to local Ollama and records requests, responses, timing, and decisions. The same Gemma model is used by the Polycog agent and the standalone workflows. Domain-specific rule design remains in the upstream agent; this repository does not introduce firefighter-specific policies.

## Workflows

1. **Original standalone prompt:** the model receives one document and the written de-identification policy.
2. **Full-packet standalone prompt:** the model receives the document packet with more explicit output instructions.
3. **Polycog PHI agent:** the model proposes text spans and answers typed classification questions; the agent applies its explicit rules to those facts.
4. **Standalone multipass workflow:** the model performs extraction, decisions in batches, and subsequent reviews within an elapsed-time budget.

In the comparison runner, the Polycog classifier uses three trials per question. The Gemma-only multipass budget is computed from the completed Polycog duration, with a minimum of 900 seconds. This matches approximate elapsed time, not model-call count or token count. A model's last accepted revision is retained at the deadline; outputs are not selected by their scores.

The model name and inference configuration are in the runner: `gemma4:e4b-it-qat`, seed base 101, temperature 0.2, `num_ctx=8192`, and `num_predict=8192`. These are configuration choices, not requirements of the Polycog library.

## Where to review the code

| File | Purpose |
|---|---|
| [pii_demo/extract.py](pii_demo/extract.py) | Original candidate extraction prompt and pattern supplements |
| [pii_demo/assess/investigate.py](pii_demo/assess/investigate.py) | Classification questions, trial count, and agreement handling |
| [pii_demo/assess/rules.py](pii_demo/assess/rules.py) | Original explicit rules and review decisions |
| [pii_demo/pipeline.py](pii_demo/pipeline.py) | Extraction, investigation, linking, and rule application |
| [pii_demo/baseline.py](pii_demo/baseline.py) | Original standalone prompt |
| [evaluate_gemma.py](evaluate_gemma.py) | Local Ollama adapter and full-packet prompt controls |
| [run_budget_gemma.py](run_budget_gemma.py) | Standalone extraction, batched decisions, and review prompts |
| [run_real_comparison.py](run_real_comparison.py) | Comparison orchestration and time-budget calculation |
| [audit_results.py](audit_results.py) | Generic overlap and output-coverage helpers required by the runners |
| [build_practical_audit.py](build_practical_audit.py) | Counting logic for separately supplied targets and reviewer judgments |
| [resume_real_comparison.py](resume_real_comparison.py) | Continuation from compatible saved requests |
| [recover_real_multipass.py](recover_real_multipass.py) | Bounded extraction/decision retry wrapper with prior work charged to the budget |

The continuation and recovery helpers require a specific prior-run state; they are included for code review and are not general-purpose commands for arbitrary interrupted runs.

## Install and inspect

Requirements: Python 3.13 or later, `uv`, and a local Ollama installation for inference.

```text
git clone https://github.com/flaviaa2/polycog-gemma-code.git
cd polycog-gemma-code
uv sync --locked
uv run --locked python evaluate_gemma.py --help
```

The local adapter expects Ollama at `http://127.0.0.1:11434`. Install the configured model before making inference calls. The weights are not included here.

## Running with separately supplied inputs

This is a source-code review package. A complete benchmark run also requires documents and expected decisions supplied by the evaluator. The historical comparison runner expects:

```text
real_documents/
  corpus/                UTF-8 .txt documents
  annotations.json       Expected text spans and actions
  source_manifest.json   Document provenance
```

`annotations.json` contains a `targets` list. Each target supplies `doc`, `start`, `end`, `text`, `action`, and `category`, with `replacement` when applicable. Offsets are Python string positions with an exclusive end, and the text must match the input exactly. The source manifest records provenance separately from prompts. The runner's protocol text describes the original four-document layout, so a different dataset requires corresponding protocol updates.

Once those inputs and the local model are available, the runner is:

```text
uv run --locked python -X utf8 -u run_real_comparison.py
```

It writes a protocol and run artifacts locally and refuses to overwrite an existing protocol or run directory. Those generated files are ignored by Git. The separate practical-counting script additionally requires target definitions, reviewer noise judgments, and completed-run metadata; none of those judgments are included here.

The required coverage functions were retained from the original evaluation helper, while its standalone diagnostic examples and report-generation entry point were removed for this source-only package. Inference prompts and upstream rules were not changed. [CODE-PROVENANCE.json](CODE-PROVENANCE.json) records the copied-file hashes and that adaptation.

## Attribution

The original PHI agent is by Polycog, Inc., under the [Apache-2.0 license](LICENSE). Its [citation metadata](upstream/CITATION.cff) is preserved. The local comparison harness and this documentation were prepared with Codex for Flavia. This repository is not an official Polycog release or a validated compliance tool.
