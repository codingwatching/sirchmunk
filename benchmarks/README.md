# Benchmarks ResearchOps Usage

This module provides the experimental infrastructure used to evaluate Sirchmunk under research-grade conditions. It is designed for experiments where the central claim is not simply that one question-answering system is always more accurate than another, but that a system can operate directly over dynamic raw data while preserving traceability, setup-cost transparency, and competitive answer quality.

The default reference workflow is HotpotQA fullwiki, but the framework is benchmark-agnostic. The same lifecycle also supports mechanism-oriented experiments such as setup cost, freshness, storage overhead, source fidelity, and warm reuse.

## Design Philosophy

The benchmark stack follows a ResearchOps philosophy: every reported number should be reproducible, attributable, and falsifiable from structured artifacts rather than reconstructed from logs or narrative claims.

The framework therefore separates four concerns:

- Execution: run benchmark adapters through a unified runner with checkpointing, retry, cache policy, and budget guards.
- Evaluation: compare systems on the same GoldenSet and re-score predictions with a shared benchmark judge.
- Governance: distinguish exploration runs from frozen evaluation runs and prevent test-set tuning from entering publication claims.
- Reporting: generate metric-first tables and reports only from machine-readable artifacts, with validator gates for missing provenance or unfair comparisons.

For paper experiments, use official task metrics as the primary evidence. For HotpotQA, official exact match and token F1 are the main answer-quality metrics. LLM-based semantic judging is allowed only as an explicitly marked auxiliary analysis and must not replace the official metrics in the main result table.

## Directory Structure

```text
benchmarks/
  framework/              # ResearchOps execution, protocol, artifact, queue, and analysis layer
  evaluation/             # GoldenSet, baseline evaluation, statistics, tables, reports, validation
  baselines/              # BaselineAdapter implementations and imported-prediction adapters
  hotpotqa/               # HotpotQA adapter, judge, evidence, metrics, env examples
  setup_cost/             # Mechanism benchmark: setup cost
  freshness/              # Mechanism benchmark: dynamic data freshness
  storage_overhead/       # Mechanism benchmark: storage overhead
  source_fidelity/        # Mechanism benchmark: raw-source traceability
  warm_reuse/             # Mechanism benchmark: warm cache / reuse behavior
  run_queue.py            # P3 queue and unattended execution CLI
  run_evaluation.py       # Baseline comparison and paper table CLI
  run_report.py           # Metric-first report generation CLI
  run_research_loop.py    # Exploration and research-loop CLI
```

Supported benchmark names are resolved through `framework/registry.py`. Current names include `hotpotqa`, `setup_cost`, `freshness`, `storage_overhead`, `source_fidelity`, and `warm_reuse`.

## End-to-End Workflow

The recommended workflow has four stages. The stages are intentionally separated to prevent exploratory optimization from contaminating frozen paper evaluation.

### Stage 0: Prepare Environment And Data

Create a private runtime env from the provided example. Do not commit secrets.

```bash
cp benchmarks/hotpotqa/env.hotpotqa.exploration.example benchmarks/hotpotqa/.env.hotpotqa.exploration
cp benchmarks/hotpotqa/env.hotpotqa.frozen.example benchmarks/hotpotqa/.env.hotpotqa.frozen
export LLM_API_KEY="..."
```

For HotpotQA fullwiki, the dataset directory should contain both the parquet split and the Wikipedia corpus directory referenced by the env file.

```text
HOTPOT_DATASET_DIR/
  fullwiki/
    validation-*.parquet
  enwiki-20171001-pages-meta-current-withlinks-abstracts/
    ... raw wiki files ...
```

Use the exploration env for smoke tests and development subsets. Use the frozen env for paper-grade evaluation only.

### Stage 1: Exploration Or Smoke Runs

Exploration is for debugging, integration checks, and candidate configuration search. It should use a limited sample size and must not be used as a final paper claim.

```bash
python benchmarks/run_queue.py add-matrix \
  --queue-path benchmarks/hotpotqa/output/exploration_queue.json \
  --registry-path benchmarks/hotpotqa/output/exploration_registry.jsonl \
  --add-bm hotpotqa=benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --systems sirchmunk \
  --seeds 42 \
  --cache-modes warm \
  --stage exploration \
  --limit 100 \
  --replace

python benchmarks/run_queue.py run \
  --queue-path benchmarks/hotpotqa/output/exploration_queue.json \
  --registry-path benchmarks/hotpotqa/output/exploration_registry.jsonl \
  --max-concurrent 1 \
  --max-tasks 1
```

Exploration may use warm cache and auxiliary diagnostics, but it should keep eval feedback disabled unless the experiment is explicitly designed to study adaptive behavior. Exploration artifacts are useful for debugging and badcase analysis, not for final claims.

### Stage 2: Frozen Sirchmunk Evaluation

Frozen evaluation produces the main Sirchmunk result. The runner and protocol validator enforce stricter constraints in this stage.

```bash
python benchmarks/run_queue.py add-matrix \
  --queue-path benchmarks/hotpotqa/output/frozen_queue.json \
  --registry-path benchmarks/hotpotqa/output/frozen_registry.jsonl \
  --add-bm hotpotqa=benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --systems sirchmunk \
  --seeds 42 \
  --cache-modes cold \
  --stage frozen \
  --limit 0 \
  --replace \
  --config-json '{"sample_timeout_seconds":300,"benchmark_timeout_seconds":172800}'

python benchmarks/run_queue.py run \
  --queue-path benchmarks/hotpotqa/output/frozen_queue.json \
  --registry-path benchmarks/hotpotqa/output/frozen_registry.jsonl \
  --max-concurrent 1
```

Frozen evaluation requirements:

- `stage` must be `frozen`.
- Cache mode must be `cold` or `compiled`; `warm`, `none`, and dry-run cache are rejected.
- Eval feedback must be disabled.
- Adaptive memory updates must be disabled. If memory is enabled, it must be versioned and read-only.
- LLM judge must be disabled for the main table unless explicitly marked as auxiliary.
- The run artifact must record sample count, sample ID checksum, config hash, git snapshot, system specs, dataset manifest, cache report, predictions, and per-sample evaluation.

A successful frozen run writes artifacts under the benchmark output directory, for example:

```text
benchmarks/hotpotqa/output/frozen/runs/<run_id>/
  protocol.yaml
  manifest.json
  config_snapshot.json
  git_snapshot.json
  system_specs.json
  dataset_manifest.json
  cache_report.json
  results/
    metrics.json
    predictions.jsonl
    per_sample_eval.jsonl
  checkpoints/
    samples.jsonl
```

### Stage 3: Baseline Comparison

Use `run_evaluation.py` to compare Sirchmunk with baselines on the same GoldenSet. The framework enforces sample ID consistency and records setup cost for fair comparison.

For local baselines:

```bash
python benchmarks/run_evaluation.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sirchmunk-results benchmarks/hotpotqa/output/frozen/results_YYYYMMDD_HHMMSS.jsonl \
  --baselines bm25,naive_rag \
  --golden-n 0 \
  --golden-seed 42 \
  --baseline-sample-timeout 300 \
  --baseline-max-runtime 172800 \
  --output-dir benchmarks/hotpotqa/output/paper_table
```

For imported baselines such as LightRAG or GraphRAG, provide both predictions and setup metrics:

```bash
python benchmarks/run_evaluation.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sirchmunk-results benchmarks/hotpotqa/output/frozen/results_YYYYMMDD_HHMMSS.jsonl \
  --import-baseline "LightRAG v1=outputs/lightrag_hotpotqa_predictions.jsonl" \
  --import-baseline-setup "LightRAG v1=outputs/lightrag_hotpotqa_setup_metrics.json" \
  --golden-n 0 \
  --golden-seed 42 \
  --output-dir benchmarks/hotpotqa/output/paper_table
```

The imported prediction JSONL should contain one prediction per sample ID:

```json
{"sample_id":"example-id","prediction":"answer text","elapsed":3.2,"tokens_used":1024}
```

The setup metrics JSON should report preprocessing and indexing cost:

```json
{
  "setup_seconds": 3600.0,
  "preprocessing_seconds": 1200.0,
  "index_build_seconds": 2400.0,
  "storage_bytes": 123456789,
  "indexed_documents": 5233329
}
```

Publication-grade baseline comparison requires:

- identical sample ID sets for all non-published systems;
- setup metrics for every baseline;
- imported baseline prediction coverage of at least 95%;
- classified baseline failures below the validator threshold;
- paired statistical tests where raw per-sample predictions are available.

### Stage 4: Report Generation And Validation

Generate a metric-first report from the frozen run artifact and paper table JSON:

```bash
python benchmarks/run_report.py \
  --run-dir benchmarks/hotpotqa/output/frozen/runs/<run_id> \
  --table-json benchmarks/hotpotqa/output/paper_table/paper_table.json \
  --output-dir benchmarks/hotpotqa/output/paper_table/report \
  --title "HotpotQA Fullwiki ResearchOps Report"
```

The report generator produces:

```text
report.md
report.tex
validation.json
figures/
  accuracy_latency.svg
  setup_cost.svg
  storage_overhead.svg
```

The validator returns `passed=true` only when the package is suitable as a publication-ready evidence bundle. A `BLOCKED` report should not be used for paper claims until all error-level issues are resolved.

## Quality Gates

The validator checks the following classes of evidence:

- Artifact completeness: protocol, manifest, config snapshot, git snapshot, system specs, dataset manifest, metrics, predictions, and per-sample evaluation.
- Frozen-stage integrity: stage is frozen, cache policy is deterministic, eval feedback is off, memory is not adaptive, and LLM judge is auxiliary if enabled.
- Sample pairing: all non-published systems share the same sample ID checksum.
- Baseline fairness: setup metrics are present and baseline failures are classified.
- Imported baseline coverage: prediction coverage is reported and must meet the minimum threshold.
- Failure governance: timeout, budget-exceeded, prediction, judge, and import-missing failures are surfaced rather than silently folded into answer quality.

## Research Loop Usage

`run_research_loop.py` is for exploration, badcase analysis, and configuration search. It is not the recommended entry point for final frozen claims.

```bash
python benchmarks/run_research_loop.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --limit 50 \
  --max-iter 5 \
  --dry-run
```

For multi-benchmark exploration:

```bash
python benchmarks/run_research_loop.py \
  --multi \
  --add-bm hotpotqa=benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --add-bm setup_cost=benchmarks/setup_cost/.env.setup_cost \
  --limit 30 \
  --shadow-fraction 0.10 \
  --dry-run
```

Use outputs from research-loop runs to understand failure modes and candidate configurations. Promote a configuration to frozen evaluation only after the protocol, seed, sample policy, cache policy, and metric hierarchy are fixed.

## Mechanism Benchmarks

The mechanism benchmarks are intended to test claims that are not captured by QA accuracy alone:

- `setup_cost`: startup and preprocessing cost.
- `freshness`: ability to reflect dynamic corpus updates.
- `storage_overhead`: extra artifacts required by each method.
- `source_fidelity`: traceability to raw source files.
- `warm_reuse`: behavior under cache reuse.

Example smoke run:

```bash
python benchmarks/run_queue.py add-matrix \
  --add-bm setup_cost=benchmarks/setup_cost/.env.setup_cost \
  --systems sirchmunk \
  --seeds 42 \
  --cache-modes cold \
  --stage frozen \
  --limit 1 \
  --replace

python benchmarks/run_queue.py run --max-concurrent 1 --max-tasks 1
```

These mechanism benchmarks should be reported alongside QA metrics when the paper claim concerns dynamic raw-data retrieval, setup cost, freshness, or source fidelity.

## Interpretation Guidelines

Do not interpret a single aggregate accuracy number as sufficient evidence. For research use, inspect at least:

- official EM and token F1;
- evidence recall and source grounding;
- latency distribution and token usage;
- setup time, preprocessing time, index build time, and storage bytes;
- failure counts by category;
- sample ID checksum and GoldenSet configuration;
- validator status and artifact provenance.

This framing is intentionally conservative. It treats an experiment as an auditable evidence package rather than a one-off script execution.
