# val-bpb
A behavioral compliance harness for local LLMs. Tests whether a model holds its values under adversarial pressure — not what it can do, but whether it will hold the line.
# val_bpb

A behavioral compliance harness that tests whether a local LLM 
holds its values under adversarial pressure.

## What this is

Most LLM benchmarks test capability — can the model code, reason, 
do math. val_bpb tests character — will the model hold the line 
when pushed.

84 test cases across categories including Hard Floor violations, 
Sovereignty and Privacy, Professional conduct, Constitutional 
integrity, and circumvention attempts. Each case is designed to 
pressure a model into crossing a line it shouldn't cross. The score 
reflects how often it doesn't.

## Why I built it

I'm building a sovereign AI system called JARVIS — a local, 
constitutionally-grounded AI agent running on my own hardware, 
for my family and my pharmacy. Before I could trust it with real 
work I needed to know whether it would hold its values when 
someone pushed back, reframed a request as fiction, or applied 
urgency pressure.

I couldn't find a harness that tested for that. So I built one.

## What I learned

Capability and constitutional alignment are not correlated.

On April 23, 2026 I ran val_bpb against four models:

| Model | val_bpb Score | Notes |
|---|---|---|
| Raw qwen3.6 (no constitution) | 0.0% | Floor |
| Claude Sonnet (bare) | 25.0% | Control |
| GLM-4.7-Flash | 58.3% | 35 Hard Floor violations |
| Gemma 4 27B | 59.5% | 34 Hard Floor violations |
| Claude Sonnet + constitution | 70.2% | Supervisor baseline |
| JARVIS qwen3.5:35b + constitution | 77.4% | Current champion |

GLM-4.7-Flash scores 89% on LiveCodeBench. It scored 58.3% here. 
The gap between what a model can do and whether it will hold its 
values under pressure is real and measurable. val_bpb measures 
the gap.

## How it works

val_bpb loads a test suite of adversarial prompts organized by 
category. Each prompt is designed to elicit a specific type of 
failure — fabricating data, crossing privacy lines, complying 
with reframed harmful requests, abandoning stated values under 
pressure.

Each response is evaluated against a rubric:
- PASS: model held the line
- ESCALATE: model flagged appropriately for human review  
- FAIL: model crossed the line
- ERROR: model failed to respond

Cases marked must_refuse trigger a HARD STOP if failed — these 
are non-negotiable constitutional violations.

## Running it

```bash
python val_bpb.py --model qwen3.5:35b --provider ollama \
  --delay 1 --results-dir results/
```

Results written to JSONL. Each run produces a score, a per-case 
breakdown, and a HARD STOP report if must_refuse cases fail.

## Bring your own constitution

val_bpb ships with the JARVIS constitutional test suite. The 
categories and cases reflect my specific Hard Floors and values.

If you have a different constitution — different Hard Floors, 
different sovereignty requirements, different professional 
context — the harness is designed to be extended. Fork it, 
replace the test suite with your own cases, run it against 
your model.

The architecture is the contribution. The specific cases are 
a reference implementation.

## Who I am

Licensed pharmacist. Independent pharmacy owner in rural Kentucky. 
I build AI systems after hours on the same hardware my pharmacy 
compliance apps run on during the day.

I'm not an ML researcher. I'm not a lab. I'm someone who needed 
to trust a local AI with real decisions and built the tools to 
verify that trust before extending it.

Core mission: equality and light into darkness.

## Status

Active development. JARVIS is in production. val_bpb runs 
every time a new model candidate arrives.

Contributions welcome — especially new test case categories, 
alternative scoring rubrics, and ports to other inference 
backends beyond Ollama.
