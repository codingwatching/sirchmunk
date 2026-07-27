# Benchmarks ResearchOps Usage

The `benchmarks/` module is the ResearchOps layer for evaluating Sirchmunk under reproducible, paper-grade conditions. The main workflow is now the unified control entry point:

```bash
python benchmarks/run_benchmark.py <block> [options]
```

Use this README from top to bottom. The direct low-level scripts still exist, but they are now advanced/debug tools and are documented after the main workflow.

## Main Workflow

The recommended paper workflow is:

```text
smoke-tune
→ assets
→ main
→ ablation
→ report/status
```

The purpose is to keep exploration, baseline asset construction, frozen evaluation, ablation, reporting, and status inspection in separate, auditable stages while exposing one user-facing command surface.

| Block | Command | Role | Paper claim? |
|---|---|---|---|
| `smoke-tune` | `run_benchmark.py smoke-tune` | Small smoke run, integration check, tuning | No |
| `assets` | `run_benchmark.py assets` | Baseline preprocessing/index/graph/embedding lifecycle | Setup evidence only |
| `main` | `run_benchmark.py main` | Frozen main experiment and paper table/report | Yes, if gates pass |
| `ablation` | `run_benchmark.py ablation` | Frozen LENS/Sirchmunk mechanism variants | Yes, as ablation |
| `report` | `run_benchmark.py report` | Regenerate report/table validation from artifacts | Depends on gates |
| `status` | `run_benchmark.py status` | Inspect summaries and asset registries | No |
| `queue` | `run_benchmark.py queue` | Advanced queue operations | Operational |

## Install Benchmark Dependencies

```bash
pip install -r requirements/core.txt -r requirements/benchmarks.txt
```

`requirements/benchmarks.txt` contains benchmark-only extras such as `pyarrow` for HotpotQA fullwiki parquet files. Normal Sirchmunk usage does not require these extras.

## Prepare Private Environments

Create private env files from examples and never commit secrets:

```bash
cp benchmarks/.env.global.example benchmarks/.env.global
cp benchmarks/hotpotqa/env.hotpotqa.base.example benchmarks/hotpotqa/.env.hotpotqa.base
cp benchmarks/hotpotqa/env.hotpotqa.exploration.example benchmarks/hotpotqa/.env.hotpotqa.exploration
cp benchmarks/hotpotqa/env.hotpotqa.frozen.example benchmarks/hotpotqa/.env.hotpotqa.frozen
```

Set the real provider credentials in `benchmarks/.env.global`:

```text
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen3.7-plus
LLM_API_KEY=<your-api-key>
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
  highest-priority runtime overrides
```

Loading priority is:

```text
.env.global < .env.hotpotqa.base < profile env < os.environ
```

For HotpotQA fullwiki, configure dataset and raw corpus paths in `.env.hotpotqa.base`:

```text
HOTPOT_DATASET_DIR=/path/to/hotpotqa_dataset
HOTPOT_WIKI_CORPUS_DIRNAME=enwiki-20171001-pages-meta-current-withlinks-abstracts
```

or, if `HOTPOT_DATASET_DIR` points directly to the `fullwiki/` parquet directory:

```text
HOTPOT_DATASET_DIR=/path/to/hotpotqa_dataset/fullwiki
HOTPOT_WIKI_CORPUS_DIR=/path/to/hotpotqa_dataset/enwiki-20171001-pages-meta-current-withlinks-abstracts
```

## Step 1: Smoke And Tune

Run the unified smoke path first. This validates env loading, data loading, retrieval, judging, artifact writing, and report generation. It is exploration-only and must not be used as a final paper claim.

```bash
python benchmarks/run_benchmark.py smoke-tune \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --limit 20 \
  --context-corpus-mode sample
```

Optional baseline smoke after the same run:

```bash
python benchmarks/run_benchmark.py smoke-tune \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --limit 20 \
  --run-evaluation \
  --baselines bm25,naive_rag
```

Expected exploration outputs:

```text
benchmarks/hotpotqa/output/exploration/runs/<run_id>/results/metrics.json
benchmarks/hotpotqa/output/exploration/runs/<run_id>/results/predictions.jsonl
benchmarks/hotpotqa/output/exploration/runs/<run_id>/reports/report.md
benchmarks/hotpotqa/output/exploration/runs/<run_id>/reports/validation.json
```

`quickstart_ok=True` means the local pipeline is healthy. `paper_ready=False` is expected because exploration artifacts are intentionally blocked from paper claims.

## Step 2: Build Baseline Assets

Baseline and competitor preprocessing belongs to the `assets` block. This includes indexing, embedding, graph construction, lifecycle feasibility, setup cost, storage cost, and structured failure reasons.

```bash
python benchmarks/run_benchmark.py assets \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --methods bm25,naive_rag \
  --corpus-scale fullwiki \
  --build-timeout 86400 \
  --max-disk-bytes 500000000000 \
  --stage frozen \
  --strict
```

Outputs:

```text
benchmarks/hotpotqa/output/assets/
  asset_registry.jsonl
  asset_summary.json
  lifecycle/
    baseline_lifecycle.jsonl
    <method>_latest.json
  tables/
    feasibility_table.json
    feasibility_table.md
    feasibility_table.tex
```

Inspect the registry:

```bash
python benchmarks/run_benchmark.py status \
  --asset-registry benchmarks/hotpotqa/output/assets/asset_registry.jsonl \
  --benchmark hotpotqa
```

The registry is append-only. Failed baselines stay visible with structured reasons such as `timeout`, `oom`, `disk_exceeded`, `api_budget_exceeded`, `dependency_missing`, `partial_index`, or `index_validation_failed`.

## Step 3: Run The Frozen Main Experiment

The main block owns the paper-facing path:

```text
sampling / fixed IDs
→ optional frozen Sirchmunk run
→ baseline comparison
→ paper table
→ report
→ Gate 0-5 validation
→ main_summary.json
```

If Sirchmunk results already exist, pass them directly:

```bash
python benchmarks/run_benchmark.py main \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sirchmunk-results benchmarks/hotpotqa/output/main/runs/<run_id>/results/predictions.jsonl \
  --run-artifact-dir benchmarks/hotpotqa/output/main/runs/<run_id> \
  --baselines bm25,naive_rag \
  --asset-registry benchmarks/hotpotqa/output/assets/asset_registry.jsonl \
  --sampling-method stratified \
  --golden-n 2000 \
  --strata type,supporting_fact_bucket \
  --cache-mode cold \
  --generate-report \
  --strict
```

To let the control layer run Sirchmunk first:

```bash
python benchmarks/run_benchmark.py main \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --run-sirchmunk \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sample_ids.json \
  --baselines bm25,naive_rag \
  --asset-registry benchmarks/hotpotqa/output/assets/asset_registry.jsonl \
  --cache-mode cold \
  --generate-report \
  --strict
```

Main outputs:

```text
benchmarks/hotpotqa/output/main/
  sampling/
    sampling_*_protocol.json
    sampling_*_manifest.json
    sampling_*_sample_ids.json
  runs/<run_id>/
  evaluation/
    paper_table.json
    paper_table.md
    paper_table.tex
  report/
    report.md
    report.tex
    report_fragment.tex
    validation.json
  main_summary.json
```

Use `main_summary.json` as the single status artifact for the main run. `paper_ready=true` requires all blocking gates to pass.

## Step 4: Run Ablations

The ablation block creates frozen LENS/Sirchmunk mechanism variants and queues them through the P3 queue infrastructure.

```bash
python benchmarks/run_benchmark.py ablation \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sample_ids.json \
  --cache-mode cold \
  --max-combinations 16 \
  --replace
```

To execute queued variants immediately:

```bash
python benchmarks/run_benchmark.py ablation \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sample_ids.json \
  --cache-mode cold \
  --max-combinations 16 \
  --run \
  --max-concurrent 1 \
  --replace
```

Outputs:

```text
benchmarks/hotpotqa/output/ablation/
  ablation_spec.json
  variants.json
  ablation_summary.json
benchmarks/hotpotqa/output/queue/
  ablation_queue.json
  ablation_registry.jsonl
```

The default ablation matrix varies one mechanism at a time around the frozen baseline: search mode, knowledge reuse, position prior, intent modulation, and loop budget.

## Step 5: Report And Status

Regenerate a report from existing artifacts:

```bash
python benchmarks/run_benchmark.py report \
  --run-dir benchmarks/hotpotqa/output/main/runs/<run_id> \
  --table-json benchmarks/hotpotqa/output/main/evaluation/paper_table.json \
  --output-dir benchmarks/hotpotqa/output/main/report \
  --title "HotpotQA Fullwiki ResearchOps Report" \
  --strict
```

Inspect a main summary:

```bash
python benchmarks/run_benchmark.py status \
  --summary benchmarks/hotpotqa/output/main/main_summary.json
```

Inspect queue state:

```bash
python benchmarks/run_benchmark.py queue \
  --queue-path benchmarks/hotpotqa/output/queue/ablation_queue.json \
  status
```

## Quality Gates

The control layer evaluates Gate 0-5:

| Gate | Scope | Blocking evidence |
|---|---|---|
| Gate 0 | Parameters | benchmark, stage, cache mode, sampling args, asset args |
| Gate 1 | Assets | registry readability, query-ready baseline assets, structured failures |
| Gate 2 | Sampling | fixed sample IDs, GoldenSet, sampling protocol, checksum |
| Gate 3 | Frozen run | `stage=frozen`, deterministic cache, protocol validity |
| Gate 4 | Evaluation | sample count, systems, baseline comparison completeness |
| Gate 5 | Report | academic validator, table/sample pairing, provenance |

Frozen paper runs must satisfy:

```text
stage=frozen
cache_mode in {cold, compiled}
eval feedback disabled
memory updates disabled
sample_id_checksum recorded
all non-published systems use the same sample IDs
validator has no error-level issue
```

## Output Layout

The unified control layer uses this canonical layout:

```text
benchmarks/{benchmark}/output/
  assets/
    asset_registry.jsonl
    asset_summary.json
    lifecycle/
    tables/
    update_readiness/
  exploration/
    runs/
    candidates/
    reports/
  main/
    sampling/
    runs/
    evaluation/
    report/
    main_summary.json
  ablation/
    ablation_spec.json
    variants.json
    ablation_summary.json
  scaling/
    scaling_study/
  queue/
    ablation_queue.json
    ablation_registry.jsonl
```

## Scaling And Update Readiness

Scaling is now reachable through the assets block:

```bash
python benchmarks/run_benchmark.py assets scaling \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --methods bm25,naive_rag \
  --scales 10k:10000,100k:100000,fullwiki:0 \
  --materialize symlink \
  --build-timeout 86400 \
  --q-values 1,10,100,1000
```

Update readiness is also part of assets lifecycle governance:

```bash
python benchmarks/run_benchmark.py assets update-readiness \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --methods bm25,naive_rag \
  --base-corpus-dir /path/to/raw/wiki \
  --operation mixed \
  --delta-docs-dir /path/to/delta/docs \
  --doc-ids doc_a.txt,doc_b.txt
```

Report scaling and update cost separately from warm-query accuracy. A system whose full-corpus index is not `READY` should not appear as a warm-query quality baseline without an explicit feasibility caveat.

## Advanced Direct Scripts

The direct scripts remain available for debugging and backward compatibility. Prefer `run_benchmark.py` for normal workflows.

| Direct script | Preferred block | When to use directly |
|---|---|---|
| `run_quickstart.py` | `smoke-tune` | Debug quickstart internals |
| `run_sampling.py` | `main` | Manually inspect sampling distributions |
| `run_lifecycle_eval.py` | `assets` | Legacy lifecycle experiments |
| `run_scaling_study.py` | `assets scaling` | Legacy scaling runs |
| `run_evaluation.py` | `main` | Manual table assembly |
| `run_report.py` | `report` | Manual report regeneration |
| `run_queue.py` | `queue` | Low-level queue debugging |
| `run_research_loop.py` | `smoke-tune` | Exploration and badcase tuning |

Manual sampling example:

```bash
python benchmarks/run_sampling.py create \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --method stratified \
  --target-n 2000 \
  --seed 42 \
  --strata type,supporting_fact_bucket \
  --allocation proportional \
  --output-dir benchmarks/hotpotqa/output/main/sampling
```

Manual research loop example:

```bash
python benchmarks/run_research_loop.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --limit 50 \
  --max-iter 5 \
  --dry-run
```

Use direct-script outputs as inputs to the unified blocks only when their artifacts preserve stage, sample IDs, checksum, config hash, and manifest provenance.

## Supported Benchmarks

Supported benchmark names are resolved through `framework/registry.py`:

```text
hotpotqa
setup_cost
freshness
storage_overhead
source_fidelity
warm_reuse
```

Mechanism benchmarks test claims not captured by QA accuracy alone:

| Benchmark | Claim |
|---|---|
| `setup_cost` | Startup and preprocessing cost |
| `freshness` | Dynamic corpus freshness |
| `storage_overhead` | Extra storage artifacts |
| `source_fidelity` | Traceability to raw sources |
| `warm_reuse` | Cache and reuse behavior |

Example mechanism smoke:

```bash
python benchmarks/run_benchmark.py smoke-tune \
  --benchmark setup_cost \
  --env benchmarks/setup_cost/.env.setup_cost \
  --limit 1 \
  --skip-report
```

## Interpreting Sampled Evaluation

For HotpotQA fullwiki, the validation split has 7,405 examples. If full evaluation is too expensive, use a frozen stratified subset instead of an ad hoc `--limit`.

Recommended main sampled evaluation:

```text
n = 2000
strata = type + supporting_fact_bucket
answer_type = monitored distribution, not primary strata
```

Recommended wording:

```text
We evaluate on a fixed stratified subset of the HotpotQA fullwiki validation split under an end-to-end raw-corpus protocol. All systems use the same sample IDs and sample-ID checksum, enabling paired uncertainty estimates.
```

Do not write sampled results as full benchmark results. Published full-benchmark numbers can be included as separate reference rows, but they are not paired comparisons unless raw predictions on the same sample IDs are available.

## Troubleshooting

- `paper_ready=false`: inspect `main_summary.json` and `report/validation.json` for failed gates.
- `Gate 1` fails: build or validate assets with `run_benchmark.py assets` and check `asset_registry.jsonl`.
- `Gate 2` fails: provide `--sample-ids-file`, `--sampling-protocol`, or a GoldenSet generated from a frozen protocol.
- `Gate 3` fails: verify `stage=frozen`, `cache-mode cold|compiled`, and disabled eval feedback/memory updates.
- `Gate 5` fails: regenerate the report with `run_benchmark.py report` and inspect validator errors.
- Env file missing: create the private profile from examples and keep secrets in ignored files.
- Imported baseline coverage low: ensure the JSONL contains one prediction for every frozen sample ID.
