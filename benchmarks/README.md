<!-- markdownlint-disable MD033 -->

# Benchmarks ResearchOps Guide

This README is the end-to-end operating guide for the `benchmarks/` module. Read it as a user journey: start with a mock/smoke experiment, freeze the evaluation sample IDs, build baseline lifecycle evidence, construct dynamic `G_n/D_n` artifacts, run the frozen main experiment, and regenerate the final report.

The normal command surface is:

```bash
python benchmarks/run_benchmark.py <task> [options]
python benchmarks/run_benchmark.py --task <task> [options]
```

Direct scripts still exist for debugging and special workflows. They are listed near the end instead of interrupting the main story.

## Storyline At A Glance

```text
mock/smoke exploration
→ frozen sample IDs
→ baseline assets
→ dynamic G_n/D_n artifacts
→ frozen main experiment
→ report/status
→ optional ablation appendix
```

| Stage | Command | Purpose | Paper claim? |
|---|---|---|---|
| Mock/smoke | `run_benchmark.py smoke-tune` | Small exploration run, env check, report smoke, optional baseline comparison | No |
| Freeze samples | `run_sampling.py create` | Create fixed stratified sample IDs and checksum | Sampling evidence only |
| Assets | `run_benchmark.py assets` | Build/validate baseline preprocessing, indexing, storage, and lifecycle evidence | Setup/lifecycle evidence |
| Dynamic G/D | `run_benchmark.py dynamic` | Build nested `G_n/D_n` sample/corpus bindings, optional dynamic baselines, and the optional stale-index arm | Yes, if stage checks pass |
| Frozen main | `run_benchmark.py main` | Run or assemble Sirchmunk results, run main baselines, generate tables/reports | Yes, if gates pass |
| Report/status | `run_benchmark.py report/status` | Regenerate reports and inspect gate state | Yes, if tied to frozen artifacts |
| Optional appendix | `run_benchmark.py ablation` | Mechanism ablation | Appendix/ablation |

The core discipline is simple: exploration can be small and iterative, but paper-facing results must use frozen sample IDs, deterministic frozen settings, and the same sample IDs across systems.

## Baseline Names And Scope

Use exactly one canonical command value per implementation.

| Scope | Command value | Role |
|---|---|---|
| Quickstart lexical smoke | `bm25` | Local lexical baseline for fast regression checks |
| Quickstart local RAG smoke | `naive_rag` | Small local RAG smoke baseline |
| Paper sparse RAG | `bm25_rag` | Fixed-chunk BM25 RAG row for main comparisons |
| Paper hybrid RAG | `hybrid_rag` | BM25 + dense reciprocal-rank fusion RAG row |
| Paper tool-use agent | `react` | Ordinary ReAct/search-agent baseline |
| LightRAG SDK lifecycle | `lightrag_v136`, `lightrag_v136_<mode>` | Index-heavy related-work/lifecycle baseline |
| Imported LightRAG v1 | `lightrag_v1` | Precomputed prediction/setup import |
| Imported GraphRAG | `graphrag` | Precomputed prediction/setup import |
| LENS ablation | `lens_full`, `lens_no_prior`, `lens_no_seq`, `lens_no_reuse` | Mechanism ablation |
| Custom adapter | `module:factory` | Advanced custom `BaselineAdapter` factory |

Recommended lists:

```text
smoke comparison: bm25,naive_rag,bm25_rag,hybrid_rag,react
paper main:       bm25_rag,hybrid_rag,react
asset build:      bm25,bm25_rag,naive_rag,react
LightRAG modes:   lightrag_v136_naive,lightrag_v136_local,lightrag_v136_global,lightrag_v136_hybrid,lightrag_v136_mix
```

`run_evaluation.py` supports the full paper-main list, including `hybrid_rag`. `run_baseline_assets.py` currently owns asset/lifecycle records for its locally supported asset methods; attach `--asset-registry` to `main` only when the registry contains ready records for every method you ask the frozen gate to enforce.

## Prepare The Environment

Install benchmark dependencies:

```bash
pip install -r requirements/core.txt -r requirements/benchmarks.txt
```

Create private env files from examples:

```bash
cp benchmarks/.env.global.example benchmarks/.env.global
cp benchmarks/hotpotqa/env.hotpotqa.base.example benchmarks/hotpotqa/.env.hotpotqa.base
cp benchmarks/hotpotqa/env.hotpotqa.exploration.example benchmarks/hotpotqa/.env.hotpotqa.exploration
cp benchmarks/hotpotqa/env.hotpotqa.frozen.example benchmarks/hotpotqa/.env.hotpotqa.frozen
```

Put provider credentials only in ignored private files or shell environment variables:

```text
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen3.7-plus
LLM_API_KEY=<your-api-key>
```

Configure HotpotQA data in `benchmarks/hotpotqa/.env.hotpotqa.base`:

```text
HOTPOT_DATASET_DIR=/path/to/hotpotqa_dataset
HOTPOT_WIKI_CORPUS_DIRNAME=enwiki-20171001-pages-meta-current-withlinks-abstracts
```

If `HOTPOT_DATASET_DIR` points directly to the `fullwiki/` parquet directory, also set the raw wiki corpus explicitly:

```text
HOTPOT_DATASET_DIR=/path/to/hotpotqa_dataset/fullwiki
HOTPOT_WIKI_CORPUS_DIR=/path/to/hotpotqa_dataset/enwiki-20171001-pages-meta-current-withlinks-abstracts
```

Configuration loading order is:

```text
benchmarks/.env.global < benchmarks/hotpotqa/.env.hotpotqa.base < profile env < os.environ
```

Use the exploration profile for mock/smoke work. Use the frozen profile only after sample IDs and run settings are fixed.

## Step 1: Mock/Smoke Exploration

This stage answers the first user question: “Can my local environment, data path, retrieval, judging, artifact writing, and report generation work at all?” It is intentionally not paper evidence.

Run the minimal smoke:

```bash
python benchmarks/run_benchmark.py smoke-tune \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --limit 20 \
  --seed 42 \
  --max-iter 1 \
  --context-corpus-mode sample
```

Run a complete smoke comparison across the current local/paper baseline families:

```bash
python benchmarks/run_benchmark.py smoke-tune \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --limit 20 \
  --seed 42 \
  --max-iter 1 \
  --context-corpus-mode sample \
  --run-evaluation \
  --baselines bm25,naive_rag,bm25_rag,hybrid_rag,react \
  --baseline-sample-timeout 0 \
  --baseline-max-runtime 0 \
  --generate-evaluation-report
```

Expected smoke outputs:

```text
benchmarks/hotpotqa/output/exploration/runs/<run_id>/results/metrics.json
benchmarks/hotpotqa/output/exploration/runs/<run_id>/results/predictions.jsonl
benchmarks/hotpotqa/output/exploration/runs/<run_id>/reports/report.md
benchmarks/hotpotqa/output/exploration/runs/<run_id>/reports/validation.json
benchmarks/hotpotqa/output/exploration/quickstart_eval/paper_table.md
```

`quickstart_ok=True` means the local pipeline is healthy. `paper_ready=False` is expected because exploration artifacts are deliberately blocked from paper claims.

### Corpus Modes And Smoke Boundaries

`smoke-tune` defaults to `--context-corpus-mode sample` for fast HotpotQA health checks. In this mode, each sampled HotpotQA parquet `context` is materialized into local `.txt` files, and sample mode forces `HOTPOT_REQUIRE_CONTEXT_ANSWERABLE=true`. This makes the smoke set intentionally closed and answerable.

Treat sample-context scores as pipeline health metrics only. BM25, Naive RAG, BM25-RAG, and Hybrid-RAG build indexes over the evaluation-set sample contexts during baseline preparation, so their smoke rows carry `evaluation_set_context_index` risk. ReAct does not build the same global index, but it still searches gold-adjacent per-sample context and is therefore not raw-corpus evidence.

Use these corpus modes deliberately:

| Mode | Intended use | Paper-facing? |
|---|---|---|
| `sample` | Fast smoke/debug over answerable HotpotQA sample contexts | No; validator emits `sample_context_corpus` and `evaluation_set_context_index` errors |
| `wiki` | Raw HotpotQA wiki corpus checks | Yes, if frozen samples, pairing, and gates pass |
| `hybrid` | Diagnostic sample+wiki comparison | Not directly comparable to raw-corpus claims; validator emits a hybrid warning |

Evaluation tables now record `corpus_provenance`, `corpus_risk`, and per-baseline `baseline_index_scope`. A sample-context Markdown table also prints an explicit warning. For paper-facing runs, prefer raw wiki or dynamic `G_n/D_n` snapshots and require the academic validator to have no error-level corpus issue.

## Step 2: Freeze The Sample IDs

After smoke passes, freeze the evaluation set. For HotpotQA fullwiki, the recommended main sampled protocol is stratified `n=500` over `type` and `supporting_fact_bucket`, with the default validation population size of 7,405. `n=500` is the recommended maximum: it already reproduces the population strata proportions to within 0.11 percentage points while keeping query cost tractable for every baseline.

Sampling is gated on raw-corpus synchronization. The parquet split defines the questions and their `supporting_facts` titles, but the raw enwiki dump must actually contain those articles, and the parquet files do not record which dump they belong to. `create` therefore resolves every referenced article against the dump before freezing anything, and refuses to write sample IDs when a supporting-fact article is missing. Check it independently at any time:

```bash
python benchmarks/run_sampling.py check-corpus-sync \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen
```

On the official `enwiki-20171001-pages-meta-current-withlinks-abstracts` dump the validation split is fully closed:

```text
shard_count                   = 15517
evidence_title_closure        = 100.0   (13781/13781, blocking)
context_title_closure         = 100.0   (58293/58293, informational)
question_closure              = 100.0   (7405/7405 resolvable)
passed                        = True
```

Evidence closure is blocking because a question whose supporting-fact article is absent cannot be answered from a snapshot at all. Context-distractor closure is reported but non-blocking, since a missing distractor only makes a snapshot slightly easier. Use `--allow-corpus-desync` to freeze anyway when you accept that the run is not main-table eligible.

The first check scans the dump once (about 30s) and caches a title index keyed by the dump fingerprint under `benchmarks/hotpotqa/.work/.cache/corpus_index/`; later checks and stage builds reuse it in about a second. `--rebuild-corpus-index` forces a rescan. The dump identity is recorded as `wiki_corpus_fingerprint` in the dataset manifest and as `wiki_corpus_fingerprint` in the sample-ids metadata, so replacing, truncating, or extending the dump changes the recorded fingerprint instead of silently passing.

```bash
python benchmarks/run_sampling.py create \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --method stratified \
  --target-n 500 \
  --seed 42 \
  --strata type,supporting_fact_bucket \
  --allocation proportional \
  --min-per-stratum 1 \
  --expected-population-size 7405 \
  --output-dir benchmarks/hotpotqa/output/main/sampling
```

Use the generated sample ID file in all frozen runs:

```text
benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_500_sample_ids.json
```

Validate the manifest before running expensive systems:

```bash
python benchmarks/run_sampling.py validate \
  --manifest benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_500_manifest.json
```

Never replace this with an ad hoc `--limit` for paper-facing claims. A sampled result is valid only as a paired comparison when every system uses the same sample IDs and checksum.

## Step 3: Build Baseline Assets

The assets stage records lifecycle evidence: preprocessing, indexing, storage, feasibility, structured failures, and Sirchmunk’s no-index row. It is a frozen preflight for lifecycle evidence and is separate from Step 4 warm-query accuracy.

Run this frozen asset preflight before the final main experiment:

```bash
python benchmarks/run_benchmark.py assets \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --output-dir benchmarks/hotpotqa/output \
  --methods bm25,bm25_rag,naive_rag,react \
  --limit 20 \
  --seed 42 \
  --corpus-scale fullwiki \
  --build-timeout 86400 \
  --max-ram-bytes 0 \
  --max-disk-bytes 500000000000 \
  --max-llm-tokens 0 \
  --max-api-cost-usd 0 \
  --retry-count 0 \
  --bm25-max-files 20000 \
  --naive-rag-max-files 5000 \
  --stage frozen \
  --strict
```

Use this command as the default Step 3 for the final experiment. `--limit 20` loads a small `golden_like` sample set only for baseline path inference and `baseline.prepare()` preflight; it is not the final evaluation size. The final Step 4 sample size is controlled by the frozen sample ID file from Step 2, normally `n=500` for the HotpotQA sampled main protocol.

Parameter notes for the final path:

| Parameter | Final command value | Why it is set this way |
|---|---|---|
| `--output-dir` | `benchmarks/hotpotqa/output` | Keeps assets, main runs, evaluation tables, and reports under the same benchmark output root |
| `--methods` | `bm25,bm25_rag,naive_rag,react` | Builds currently supported lifecycle asset rows plus Sirchmunk’s no-index row |
| `--limit` | `20` | Fast frozen asset preflight; do not change it to `500` just to match Step 4 |
| `--corpus-scale` | `fullwiki` | Records the asset evidence as HotpotQA fullwiki-facing |
| `--build-timeout` | `86400` | Enforces a 24h wall-clock timeout per baseline asset build |
| `--max-disk-bytes` | `500000000000` | Records a 500GB disk budget for fullwiki asset feasibility |
| `--max-ram-bytes`, `--max-llm-tokens`, `--max-api-cost-usd` | `0` | No explicit ceiling in this command; set positive values only for constrained runs |
| `--retry-count` | `0` | Keeps failure classification reproducible |
| `--stage frozen`, `--strict` | enabled | Required for paper-facing asset evidence; strict exits non-zero on blocked/failed assets |

In raw wiki mode, the small `--limit` usually resolves to the same fullwiki corpus directory, so index scale is mainly controlled by `--corpus-scale` and the file caps. In sample or hybrid corpus modes, `--limit` changes the number of materialized sample-context paths and therefore changes the asset input scale.

Do not pass the Step 3 registry to Step 4 by default. The final Step 4 command below uses `bm25_rag,hybrid_rag,react`, while this assets facade currently builds registry rows for `bm25,bm25_rag,naive_rag,react`. Attach `--asset-registry` to Step 4 only after you have a registry with ready records for every exact method in Step 4.

Inspect the registry:

```bash
python benchmarks/run_benchmark.py status \
  --asset-registry benchmarks/hotpotqa/output/assets/asset_registry.jsonl \
  --benchmark hotpotqa \
  --methods bm25,bm25_rag,naive_rag,react \
  --stage frozen \
  --reusable-only
```

Expected asset outputs:

```text
benchmarks/hotpotqa/output/assets/asset_registry.jsonl
benchmarks/hotpotqa/output/assets/asset_summary.json
benchmarks/hotpotqa/output/assets/lifecycle/baseline_lifecycle.jsonl
benchmarks/hotpotqa/output/assets/tables/feasibility_table.md
```

Failed baseline assets remain visible with structured reasons such as `timeout`, `oom`, `disk_exceeded`, `api_budget_exceeded`, `dependency_missing`, `partial_index`, or `index_validation_failed`.

## Step 4: Run The Frozen Main Experiment

The final frozen main command uses fixed sample IDs, cold cache, strict gates, and the core paper-main online baseline set: `bm25_rag,hybrid_rag,react`.

Run Sirchmunk and the final core paper-main baselines:

```bash
python benchmarks/run_benchmark.py main \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --output-dir benchmarks/hotpotqa/output \
  --run-sirchmunk \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_500_sample_ids.json \
  --expected-population-size 7405 \
  --baselines bm25_rag,hybrid_rag,react \
  --cache-mode cold \
  --baseline-sample-timeout 0 \
  --baseline-max-runtime 0 \
  --baseline-max-total-tokens 0 \
  --baseline-max-api-cost-usd 0 \
  --baseline-max-disk-bytes 0 \
  --baseline-min-free-disk-bytes 0 \
  --generate-report \
  --title "HotpotQA Fullwiki ResearchOps Report" \
  --strict
```

If Sirchmunk predictions already exist, assemble the same frozen main table without rerunning Sirchmunk:

```bash
python benchmarks/run_benchmark.py main \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --output-dir benchmarks/hotpotqa/output \
  --sirchmunk-results benchmarks/hotpotqa/output/main/runs/<run_id>/results/predictions.jsonl \
  --run-artifact-dir benchmarks/hotpotqa/output/main/runs/<run_id> \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_500_sample_ids.json \
  --expected-population-size 7405 \
  --baselines bm25_rag,hybrid_rag,react \
  --cache-mode cold \
  --baseline-sample-timeout 0 \
  --baseline-max-runtime 0 \
  --baseline-max-total-tokens 0 \
  --baseline-max-api-cost-usd 0 \
  --baseline-max-disk-bytes 0 \
  --baseline-min-free-disk-bytes 0 \
  --generate-report \
  --title "HotpotQA Fullwiki ResearchOps Report" \
  --strict
```

Related-work imports such as `lightrag_v1` or `graphrag` are not part of this directly executable core command because they require precomputed prediction/setup files. Add them in a separate assembly run only after those files exist, or use direct `run_evaluation.py` for non-default SDK LightRAG options.

Main outputs:

```text
benchmarks/hotpotqa/output/main/runs/<run_id>/results/predictions.jsonl
benchmarks/hotpotqa/output/main/evaluation/paper_table.json
benchmarks/hotpotqa/output/main/evaluation/paper_table.md
benchmarks/hotpotqa/output/main/evaluation/paper_table.tex
benchmarks/hotpotqa/output/main/report/report.md
benchmarks/hotpotqa/output/main/report/validation.json
benchmarks/hotpotqa/output/main/main_summary.json
```

Use `main_summary.json` as the single run-status artifact. `paper_ready=true` requires all blocking gates to pass.

## Step 5: Regenerate Report And Inspect Status

Regenerate the report from existing frozen artifacts:

```bash
python benchmarks/run_benchmark.py report \
  --run-dir benchmarks/hotpotqa/output/main/runs/<run_id> \
  --table-json benchmarks/hotpotqa/output/main/evaluation/paper_table.json \
  --output-dir benchmarks/hotpotqa/output/main/report \
  --title "HotpotQA Fullwiki ResearchOps Report" \
  --stage frozen \
  --strict
```

Inspect the frozen run summary:

```bash
python benchmarks/run_benchmark.py status \
  --summary benchmarks/hotpotqa/output/main/main_summary.json
```

A terminal `Baseline Final Report` is printed whenever `--baselines` triggers real baseline execution. It summarizes `Baseline`, `N`, `Acc`, `EM`, `F1`, `Cov`, `Evd`, `Avg`, `P95`, `Tok/Q`, `Fail`, and `Notes` without requiring you to open the JSON table.

## Quality Gates

Frozen paper runs are checked by Gate 0-5:

| Gate | Scope | Blocking evidence |
|---|---|---|
| Gate 0 | Parameters | benchmark, stage, cache mode, sampling args, asset args |
| Gate 1 | Assets | registry readability and reusable baseline assets when asset reuse is requested |
| Gate 2 | Sampling | fixed sample IDs, GoldenSet, sampling protocol, checksum |
| Gate 3 | Frozen run | `stage=frozen`, deterministic cache, valid protocol |
| Gate 4 | Evaluation | sample count, systems, baseline comparison completeness |
| Gate 5 | Report | academic validator, table/sample pairing, corpus provenance/risk |

Frozen paper runs must satisfy:

```text
stage=frozen
cache_mode in {cold, compiled}
eval feedback disabled
memory updates disabled
sample_id_checksum recorded
all non-published systems use the same sample IDs
corpus_provenance is not sample for paper-facing tables
baseline_index_scope is not evaluation_set_sample_context
validator has no error-level issue
```

The academic validator treats corpus boundary violations as blocking issues. `sample_context_corpus` means the table used answerable HotpotQA sample contexts rather than the raw corpus. `evaluation_set_context_index` means at least one baseline indexed the evaluation-set sample context. `hybrid_context_corpus` is a warning that sample+wiki results need separate claim wording and should not be compared directly with raw-corpus rows.

<details>
<summary>Optional: Ablation</summary>

Create frozen LENS/Sirchmunk mechanism variants:

```bash
python benchmarks/run_benchmark.py ablation \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_500_sample_ids.json \
  --cache-mode cold \
  --max-combinations 16 \
  --replace
```

Run queued variants immediately:

```bash
python benchmarks/run_benchmark.py ablation \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_500_sample_ids.json \
  --cache-mode cold \
  --max-combinations 16 \
  --run \
  --max-concurrent 1 \
  --replace
```

Core variants are `lens_full`, `lens_no_prior`, and `lens_no_seq`. Keep `lens_no_reuse` appendix-only unless the paper explicitly studies warm-start amortization.

</details>

<details>
<summary>Optional: Scaling, Update Readiness, And LightRAG Lifecycle</summary>

Run scaling through the assets task:

```bash
python benchmarks/run_benchmark.py assets scaling \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --methods bm25,bm25_rag,naive_rag,react \
  --scales 10k:10000,100k:100000,fullwiki:0 \
  --limit 20 \
  --seed 42 \
  --materialize symlink \
  --build-timeout 86400 \
  --max-ram-bytes 0 \
  --max-disk-bytes 500000000000 \
  --max-llm-tokens 0 \
  --max-api-cost-usd 0 \
  --q-values 1,10,100,1000
```

Run update-readiness governance:

```bash
python benchmarks/run_benchmark.py assets update-readiness \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --methods bm25,bm25_rag,naive_rag,react \
  --base-corpus-dir /path/to/raw/wiki \
  --operation mixed \
  --delta-docs-dir /path/to/delta/docs \
  --doc-ids doc_a.txt,doc_b.txt \
  --mutation-ratio 0.0 \
  --materialize symlink \
  --limit 20 \
  --seed 42 \
  --bm25-max-files 20000 \
  --naive-rag-max-files 5000
```

Install LightRAG v1.3.6 only for SDK-backed related-work lifecycle rows:

```bash
pip install git+https://github.com/HKUDS/LightRAG.git@v1.3.6
```

Run the default LightRAG lifecycle mode through the dynamic task:

```bash
python benchmarks/run_benchmark.py dynamic \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --golden-n 500 \
  --seed 42 \
  --stages 125,250,500 \
  --strata type,supporting_fact_bucket \
  --materialize symlink \
  --background-ratio 3.0 \
  --background-seed 42 \
  --run-baselines \
  --baselines lightrag_v136 \
  --lightrag-query-mode hybrid \
  --lightrag-max-files 0 \
  --lightrag-max-file-chars 300000
```

Run all LightRAG query modes for appendix sensitivity:

```bash
python benchmarks/run_benchmark.py dynamic \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --golden-n 500 \
  --seed 42 \
  --stages 125,250,500 \
  --strata type,supporting_fact_bucket \
  --materialize symlink \
  --background-ratio 3.0 \
  --background-seed 42 \
  --run-baselines \
  --baselines lightrag_v136_naive,lightrag_v136_local,lightrag_v136_global,lightrag_v136_hybrid,lightrag_v136_mix \
  --lightrag-max-files 0 \
  --lightrag-max-file-chars 300000
```

Report scaling and update cost separately from warm-query accuracy. A full-corpus index that is not `READY` should not appear as a warm-query baseline without a feasibility caveat.

Dynamic task outputs include the default paper-facing baselines `bm25_rag,hybrid_rag,react` unless `--baselines` is explicitly set:

```text
benchmarks/hotpotqa/output/dynamic_eval/tables/dynamic_main_results.*
benchmarks/hotpotqa/output/dynamic_eval/tables/lifecycle_main.*
benchmarks/hotpotqa/output/dynamic_eval/tables/budget_quality.*
benchmarks/hotpotqa/output/dynamic_eval/tables/update_readiness.*
benchmarks/hotpotqa/output/dynamic_eval/tables/snapshot_audit.*
```

### Stale-index arm

`update_readiness` records whether a system *declares* that it needs a rebuild. It does not show what that requirement costs in answer quality. Add `--stale-index-arm` to measure the cost directly:

```bash
python benchmarks/run_benchmark.py dynamic \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --golden-n 500 \
  --seed 42 \
  --stages 125,250,500 \
  --strata type,supporting_fact_bucket \
  --materialize symlink \
  --run-baselines \
  --baselines bm25_rag,hybrid_rag,react \
  --stale-index-arm \
  --staleness-max-samples 200
```

For each transition `D_{n-1} -> D_n` the arm compares two runs over exactly the newly added questions (`delta = G_n \ G_{n-1}`), whose supporting evidence articles only exist in `D_n`:

| Arm | Index built on | Corpus queried | Meaning |
|---|---|---|---|
| fresh | `D_n` | `D_n` | rebuild already paid |
| stale | `D_{n-1}` | `D_n` | corpus grew, index not rebuilt |

The stale arm reuses the previous stage's system instance and skips its preparation step, so index-heavy systems answer from an index that cannot reach the new evidence, while index-free systems read the current corpus at query time. Expected reading:

| System class | Members | Expected `Ev.Rec Gap` |
|---|---|---|
| `index_dependent` | `bm25_rag`, `hybrid_rag`, `lightrag_v136*` | positive |
| `index_free` | `react`, `lens_full`, `lens_no_prior`, `lens_no_seq` | near zero |

The near-zero row is a measured control result, not an assumption, which is why the arm runs for every requested system including LENS itself. Use `--staleness-max-samples` to cap delta questions per transition when query budget is limited.

Additional outputs:

```text
benchmarks/hotpotqa/output/dynamic_eval/tables/stale_index_quality.*
benchmarks/hotpotqa/output/dynamic_eval/runs/<stage>/staleness/baseline_<name>.jsonl
benchmarks/hotpotqa/output/dynamic_eval/runs/<stage>/stage_records/<name>_staleness_record.json
```

Each staleness record pins `from_corpus_checksum`, `to_corpus_checksum`, `delta_sample_id_checksum`, and `stale_index_prepared_on`, so a reported gap is traceable to one index state and one delta question set. `dynamic_eval_manifest.json` carries `stale_index_arm` and a `staleness_summary` aggregated by system class.

### Nested stage sampling fidelity

A stratified parent set only guarantees proportional strata at its own size. Cutting prefixes out of a randomly shuffled parent order turns each smaller stage into a simple random subsample, which drifts from the population and can drop rare strata entirely. Nested stages are therefore derived with a stratum-balanced order: each stratum's members are spread evenly across the sequence, so every stage stays proportional while remaining a strict subset of the next one.

Measured on the HotpotQA fullwiki validation population (7,405 questions, 8 strata from `type` x `supporting_fact_bucket`, rarest stratum 0.32%):

| Stage | Prefix of shuffled parent | Stratum-balanced order |
|---|---|---|
| `G_125` | 7.30pp drift, 2 strata empty | 0.70pp drift, 0 empty |
| `G_250` | 2.90pp drift, 1 stratum empty | 0.30pp drift, 0 empty |
| `G_500` | 0.11pp drift, 0 empty | 0.11pp drift, 0 empty |

Every stage records `strata_distribution`, `proportion_delta_by_stratum`, `max_abs_proportion_delta`, and `empty_strata` in `nested_sample_manifest.json` and in its own sample-ids file, with `reference_scope` naming what the stage was compared against (`population` when the parent manifest recorded the population distribution). The academic validator warns above 5pp drift, so these values are directly checkable rather than assumed. Pass `balance_strata=False` to `derive_nested_sample_sets` only when an older parent-order artifact must be reproduced exactly.

### Raw-corpus synchronization gate

The dynamic task re-checks corpus synchronization against the frozen parent set before building any snapshot, and records the report as `corpus_sync` in `dynamic_eval_manifest.json`. A missing supporting-fact article aborts the run with an explicit sync error instead of failing later inside a snapshot build; `--allow-corpus-desync` downgrades it to a non-blocking warning and marks the run as not main-table eligible. `--rebuild-corpus-index` forces a dump rescan. Snapshot title resolution reuses the same cached index, so evidence articles are read from the one shard that holds them rather than by streaming the dump.

</details>

<details>
<summary>Advanced Direct Scripts</summary>

Prefer `run_benchmark.py` for normal workflows. Use direct scripts when debugging or when a workflow has no facade task.

| Direct script | Preferred role | When to use directly |
|---|---|---|
| `run_quickstart.py` | `smoke-tune` | Debug quickstart internals |
| `run_sampling.py` | Freeze samples | Create/validate explicit sampling artifacts |
| `run_baseline_assets.py` | `assets` | Debug asset registry and lifecycle records |
| `run_evaluation.py` | `main` | Manual table assembly or non-default evaluation flags |
| `run_dynamic_evaluation.py` | `dynamic` | Debug `G_n/D_n` snapshots, dynamic baselines, and the stale-index arm |
| `run_report.py` | `report` | Manual report regeneration |
| `run_queue.py` | `queue` | Low-level queue debugging |
| `run_research_loop.py` | Exploration | Badcase tuning and dry-run analysis |

Manual evaluation with imported predictions:

```bash
python benchmarks/run_evaluation.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sirchmunk-results benchmarks/hotpotqa/output/main/runs/<run_id>/results/predictions.jsonl \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_500_sample_ids.json \
  --baselines bm25_rag,hybrid_rag,react \
  --import-baseline "External System=output/external_predictions.jsonl" \
  --import-baseline-setup "External System=output/external_setup_metrics.json" \
  --import-published "Reported System:acc=45.0,cov=80.0,lat=5.2,tok=0" \
  --lightrag-query-mode hybrid \
  --lightrag-max-files 0 \
  --lightrag-max-file-chars 300000 \
  --bm25-max-files 20000 \
  --naive-rag-max-files 5000 \
  --hybrid-max-files 5000 \
  --hybrid-bm25-top-k 20 \
  --hybrid-dense-top-k 20 \
  --hybrid-final-top-k 5 \
  --hybrid-dense-backend hash \
  --hybrid-dense-dim 256 \
  --baseline-sample-timeout 0 \
  --baseline-max-runtime 0 \
  --baseline-max-total-tokens 0 \
  --baseline-max-api-cost-usd 0 \
  --baseline-max-disk-bytes 0 \
  --baseline-min-free-disk-bytes 0 \
  --context-corpus-provenance wiki \
  --context-corpus-risk raw_wiki \
  --generate-report
```

Use direct-script outputs as inputs to unified tasks only when they preserve stage, sample IDs, checksum, config hash, manifest provenance, and corpus provenance/risk.

</details>

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

Mechanism smoke example:

```bash
python benchmarks/run_benchmark.py smoke-tune \
  --benchmark setup_cost \
  --env benchmarks/setup_cost/.env.setup_cost \
  --limit 1 \
  --seed 42 \
  --skip-report
```

## Troubleshooting

- `paper_ready=false`: inspect `benchmarks/hotpotqa/output/main/main_summary.json` and `benchmarks/hotpotqa/output/main/report/validation.json`.
- `Gate 1` fails: attach `--asset-registry` only when it contains ready reusable records for every frozen `--baselines` method, or rebuild matching assets first.
- `Gate 2` fails: pass the frozen `sampling_stratified_42_500_sample_ids.json`, a sampling protocol, or a valid GoldenSet manifest.
- `Gate 3` fails: verify `stage=frozen`, `cache-mode cold|compiled`, and disabled eval feedback/memory updates.
- `Gate 5` fails: regenerate the report with `run_benchmark.py report` and inspect validator errors.
- Env file missing: create the private profile from examples and keep secrets in ignored files.
- LightRAG v1.3.6 skipped: install the `v1.3.6` Git ref and rerun with `--baselines lightrag_v136`.
- Imported baseline coverage low: ensure every frozen sample ID has exactly one prediction row.
