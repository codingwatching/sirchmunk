# Benchmarks ResearchOps Usage

This module provides the experimental infrastructure used to evaluate Sirchmunk under research-grade conditions. It is designed for experiments where the central claim is not simply that one question-answering system is always more accurate than another, but that a system can operate directly over dynamic raw data while preserving traceability, setup-cost transparency, and competitive answer quality.

The default reference workflow is HotpotQA fullwiki, but the framework is benchmark-agnostic. The same lifecycle also supports mechanism-oriented experiments such as setup cost, freshness, storage overhead, source fidelity, and warm reuse.

## Quickstart: Run The Full Pipeline

Use this path first. It validates the complete local pipeline — config loading, HotpotQA data loading, retrieval, real LLM calls, metrics, artifacts, and report generation.

### Step 0: Install Dependencies

```bash
pip install -r requirements/core.txt -r requirements/benchmarks.txt
```

`requirements/benchmarks.txt` contains benchmark-only extras such as `pyarrow`, which HotpotQA needs to read fullwiki parquet files. Normal Sirchmunk usage does not require these extras.

### Step 1: Create Private Env Files

```bash
cp benchmarks/.env.global.example benchmarks/.env.global
cp benchmarks/hotpotqa/env.hotpotqa.base.example benchmarks/hotpotqa/.env.hotpotqa.base
cp benchmarks/hotpotqa/env.hotpotqa.exploration.example benchmarks/hotpotqa/.env.hotpotqa.exploration
# Then edit benchmarks/.env.global and set LLM_API_KEY=<your-api-key>.
```

The default local setup uses DashScope-compatible Qwen:

```text
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen3.7-plus
```

`LLM_API_KEY` should live in the private ignored `benchmarks/.env.global` file. Shell environment variables still work as temporary highest-priority overrides, but the recommended persistent setup is the private env file.

### Step 2: Run One Command

```bash
python benchmarks/run_quickstart.py
```

This command runs the sample count configured by the active profile (`HOTPOT_LIMIT`; `--limit` explicitly overrides it) with the configured real LLM provider, automatically skips the interactive improvement step, locates the generated run artifact, and generates a report. By default it uses `--context-corpus-mode sample`, which materializes each sample's parquet context as temporary raw-text files and filters to context-answerable smoke samples so the quickstart has a closed retrieval corpus. Use `--context-corpus-mode wiki` when you want to exercise the configured raw fullwiki corpus instead.

### Step 3: Read The Outputs

The command prints the latest `run_dir`, `metrics.json`, and report paths. Typical outputs are:

```text
benchmarks/hotpotqa/output/exploration/runs/<run_id>/results/metrics.json
benchmarks/hotpotqa/output/exploration/runs/<run_id>/results/predictions.jsonl
benchmarks/hotpotqa/output/exploration/runs/<run_id>/reports/report.md
benchmarks/hotpotqa/output/exploration/runs/<run_id>/reports/validation.json
```

Quickstart is an exploration smoke test. `quickstart_ok=True` means the end-to-end run completed without system failures and retrieved supporting evidence. `paper_ready=False` is expected for exploration runs because publication-grade reports require frozen stage settings and baseline comparison artifacts.

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
  run_quickstart.py       # One-command HotpotQA smoke + report CLI
  run_queue.py            # P3 queue and unattended execution CLI
  run_evaluation.py       # Baseline comparison and paper table CLI
  run_lifecycle_eval.py   # Full-corpus baseline build/index feasibility CLI
  run_scaling_study.py    # Multi-scale feasibility and amortized-cost CLI
  run_report.py           # Metric-first report generation CLI
  run_research_loop.py    # Exploration and research-loop CLI
```

Supported benchmark names are resolved through `framework/registry.py`. Current names include `hotpotqa`, `setup_cost`, `freshness`, `storage_overhead`, `source_fidelity`, and `warm_reuse`.

## Advanced Paper Workflow

Use this workflow after the Quickstart passes. These stages are intentionally separated to prevent exploratory optimization from contaminating frozen paper evaluation.

### Stage 0: Prepare Environment And Data

Create private runtime env files from the provided examples. Do not commit secrets.

```bash
cp benchmarks/.env.global.example benchmarks/.env.global
cp benchmarks/hotpotqa/env.hotpotqa.base.example benchmarks/hotpotqa/.env.hotpotqa.base
cp benchmarks/hotpotqa/env.hotpotqa.exploration.example benchmarks/hotpotqa/.env.hotpotqa.exploration
cp benchmarks/hotpotqa/env.hotpotqa.frozen.example benchmarks/hotpotqa/.env.hotpotqa.frozen
# Then edit benchmarks/.env.global and set LLM_API_KEY=<your-api-key>.
```

HotpotQA uses layered configuration:

```text
benchmarks/.env.global
  global LLM/provider defaults
benchmarks/hotpotqa/.env.hotpotqa.base
  shared HotpotQA dataset, corpus, search, and guard defaults
--env benchmarks/hotpotqa/.env.hotpotqa.exploration
  exploration-only differences
--env benchmarks/hotpotqa/.env.hotpotqa.frozen
  frozen-evaluation-only differences
os.environ
  highest-priority runtime overrides, such as LLM_API_KEY
```

Loading priority is:

```text
.env.global < .env.hotpotqa.base < profile env < os.environ
```

For HotpotQA fullwiki, configure the dataset and corpus paths in `.env.hotpotqa.base`. `HOTPOT_DATASET_DIR` can point either to the dataset root or directly to the `fullwiki/` parquet directory. When it points directly to `fullwiki/`, set `HOTPOT_WIKI_CORPUS_DIR` to the raw Wikipedia corpus directory explicitly.

```text
# Option A: dataset root
HOTPOT_DATASET_DIR=/path/to/hotpotqa_dataset
HOTPOT_WIKI_CORPUS_DIRNAME=enwiki-20171001-pages-meta-current-withlinks-abstracts

# Option B: direct fullwiki parquet directory
HOTPOT_DATASET_DIR=/path/to/hotpotqa_dataset/fullwiki
HOTPOT_WIKI_CORPUS_DIR=/path/to/hotpotqa_dataset/enwiki-20171001-pages-meta-current-withlinks-abstracts
```

Expected dataset layout:

```text
/path/to/hotpotqa_dataset/
  fullwiki/
    validation-*.parquet
    test-*.parquet
    train-*.parquet
  enwiki-20171001-pages-meta-current-withlinks-abstracts/
    ... raw wiki files ...
```

Use the exploration profile for smoke tests and development subsets. Use the frozen profile for paper-grade evaluation only. The base env should contain shared settings; profile env files should only contain stage-specific overrides.

For real LLM runs, set the real `LLM_API_KEY` in the private ignored `benchmarks/.env.global` file. Shell environment variables still work as temporary highest-priority overrides, but the recommended persistent setup is the private env file. Never commit real secrets.

### Optional: Manual Smoke Test

`run_quickstart.py` is the recommended entry point. If you need to debug the underlying commands manually, the equivalent smoke run is:

```bash
printf 'skip\n' | python benchmarks/run_research_loop.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --max-iter 1 \
  --dry-run \
  --log-level INFO
```

Omit `--limit` to use the active profile's `HOTPOT_LIMIT`; pass `--limit <N>` only when you want a one-off override.

Then generate the report manually:

```bash
python benchmarks/run_report.py \
  --run-dir benchmarks/hotpotqa/output/exploration/runs/<run_id> \
  --output-dir benchmarks/hotpotqa/output/exploration/runs/<run_id>/reports \
  --title "HotpotQA Quickstart Smoke Test Report"
```

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

Exploration may use warm cache and auxiliary diagnostics, but it should keep eval feedback disabled unless the experiment is explicitly designed to study adaptive behavior. Exploration artifacts are useful for debugging and badcase analysis, not for final claims. Shared dataset and search defaults should remain in `.env.hotpotqa.base`; profile files should only override the stage-dependent keys.

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

### Stage 3.5: Full-Corpus Feasibility Evaluation

For HotpotQA fullwiki and other large corpora, index-heavy baselines such as LightRAG, GraphRAG, or RAPTOR must be evaluated as lifecycle systems. A baseline that cannot finish preprocessing under the declared resource budget should remain in the feasibility table with a structured failure reason instead of being silently removed.

```bash
python benchmarks/run_lifecycle_eval.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --baselines bm25,naive_rag \
  --corpus-scale fullwiki \
  --build-timeout 86400 \
  --max-disk-bytes 500000000000
```

The command writes lifecycle artifacts under `lifecycle_eval/`, including:

```text
baseline_lifecycle.jsonl
<baseline>_latest.json
tables/feasibility_table.{md,tex,json}
lifecycle_summary.json
```

Use `module:factory` baseline specs to plug in real index-heavy competitors:

```bash
--baselines bm25,my_lightrag_adapter:create_lightrag_v1
```

The factory must return a `BaselineAdapter`, usually an `IndexingSdkBaseline`, whose `prepare_fn` builds the external index and whose `validate_fn` verifies full-corpus readiness.

### Stage 3.6: Multi-Scale Scaling Study And Amortized Cost

To diagnose whether a competitor fails because of corpus scale rather than implementation error, run the same lifecycle protocol over increasing corpus sizes.

```bash
python benchmarks/run_scaling_study.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --baselines bm25,naive_rag \
  --scales 10k:10000,100k:100000,fullwiki:0 \
  --materialize symlink \
  --build-timeout 86400 \
  --q-values 1,10,100,1000
```

Outputs are written under `scaling_study/`:

```text
corpus_subsets/                 # deterministic subset manifests / symlink dirs
<scale>/lifecycle/              # per-scale lifecycle records
scaling_lifecycle_records.jsonl
scaling_metrics.json            # seconds_per_doc / bytes_per_doc / failure reason
amortized_cost_curves.json      # C_avg(Q) curves
scaling_study_summary.json
```

The amortized lifecycle cost is defined as:

$$C_{\text{avg}}(Q) = \frac{C_{\text{build}} + C_{\text{update}}}{Q} + C_{\text{query}}$$

Report full-corpus feasibility and warm-query quality separately. Warm-query metrics should only be reported for systems whose full-corpus index is `READY`.

### Stage 3.7: Dynamic Update And Ablation Planning

P2 utilities expose the raw building blocks for dynamic-document experiments and LENS mechanism ablations:

- `framework.dynamic_update.DynamicUpdateManager`: creates mutated corpus versions and records update cost / rebuild-required outcomes.
- `framework.ablation_matrix.default_lens_ablation_spec()`: generates conservative one-axis-at-a-time LENS ablation variants for search mode, knowledge reuse, position prior, intent modulation, and loop budget.

These utilities are intended to feed future frozen runs and paper ablations. Dynamic update experiments should report update time, whether a full rebuild is required, freshness accuracy, and post-update query quality.

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
