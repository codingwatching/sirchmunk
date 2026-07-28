# Benchmarks ResearchOps 使用指南

`benchmarks/` 模块是 Sirchmunk 在可复现、论文级条件下进行评估的 ResearchOps 层。当前主流程统一收敛到总控入口：

```bash
python benchmarks/run_benchmark.py <block> [options]
```

请按本文档从上到下使用。底层 direct scripts 仍然保留，但主要用于高级调试和向后兼容，相关说明放在主流程之后。

## Main Workflow

推荐的论文实验流程为：

```text
smoke-tune
→ assets
→ main
→ ablation
→ report/status
```

该流程的目标是将探索、baseline 资产构建、冻结评估、消融、报告和状态检查拆分为互相隔离、可审计的阶段，同时对用户暴露统一命令入口。

| Block | Command | 作用 | 可作为论文结论？ |
|---|---|---|---|
| `smoke-tune` | `run_benchmark.py smoke-tune` | 小样本 smoke run、集成检查、调参 | 否 |
| `assets` | `run_benchmark.py assets` | baseline 预处理、索引、图、embedding 生命周期 | 仅作为 setup 证据 |
| `main` | `run_benchmark.py main` | 冻结主实验和论文表格/报告 | 若 gate 通过则可以 |
| `ablation` | `run_benchmark.py ablation` | 冻结的 LENS/Sirchmunk 机制变体 | 可作为消融 |
| `report` | `run_benchmark.py report` | 从 artifacts 重新生成报告/表格验证 | 取决于 gate |
| `status` | `run_benchmark.py status` | 检查 summary 和 asset registry | 否 |
| `queue` | `run_benchmark.py queue` | 高级队列操作 | 运维用途 |

## Baseline Scope (Phase 0)

Phase 0 只校准 baseline 语义和缓存复用安全，不新增新的检索家族。`bm25` / `bm25_local` 与 `naive_rag` / `naive_rag_local` 只用于 quickstart/local smoke baseline，可用于回归检查，但不代表论文主表中的 BM25-RAG。

面向论文主实验时，使用 `bm25_rag` 表示 fixed-chunk sparse RAG，使用 `hybrid_rag` 表示 BM25+dense reciprocal-rank fusion RAG，使用 `react` / `react_search` 表示普通 tool-use agent baseline。已有 baseline JSONL 只有在缓存中的 `baseline_name`、`citation_name`、adapter class、schema version 和 config hash 与当前 adapter 完全匹配时才允许复用，避免用新表格名称包装旧预测结果。

剩余计划保持克制：可选暴露 `dense_rag` 到 appendix/sensitivity；LightRAG v1.3.6 保持在 lifecycle/related-work 表。Long-context baseline 明确排除在当前计划之外。

`hybrid_rag` 默认使用 deterministic hashed dense backend，保证 smoke 和 lifecycle 检查不依赖模型下载。若要运行更强的 embedding-backed 设置，可传入 `--hybrid-dense-backend sirchmunk_embedding` 并配置 `EMBEDDING_MODEL_ID`；backend 选择会写入 baseline metadata 和 cache identity。

## Install Benchmark Dependencies

```bash
pip install -r requirements/core.txt -r requirements/benchmarks.txt
```

`requirements/benchmarks.txt` 包含 benchmark 专用的额外依赖，例如 HotpotQA fullwiki parquet 文件读取所需的 `pyarrow`。普通 Sirchmunk 使用不需要安装这些额外依赖。

LightRAG v1.3.6 是可选的 related-work baseline。只有需要复现 LightRAG 生命周期表时才安装：

```bash
pip install git+https://github.com/HKUDS/LightRAG.git@v1.3.6
```

LightRAG adapter 会通过 LightRAG SDK 调用 OpenAI 兼容的 LLM 和 embedding 接口。provider 凭证应放在 `benchmarks/.env.global` 或 shell 环境变量中，不要提交到仓库。

## Prepare Private Environments

从示例文件创建私有 env 文件，并且不要提交密钥：

```bash
cp benchmarks/.env.global.example benchmarks/.env.global
cp benchmarks/hotpotqa/env.hotpotqa.base.example benchmarks/hotpotqa/.env.hotpotqa.base
cp benchmarks/hotpotqa/env.hotpotqa.exploration.example benchmarks/hotpotqa/.env.hotpotqa.exploration
cp benchmarks/hotpotqa/env.hotpotqa.frozen.example benchmarks/hotpotqa/.env.hotpotqa.frozen
```

在 `benchmarks/.env.global` 中设置真实 provider 凭证：

```text
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen3.7-plus
LLM_API_KEY=<your-api-key>
```

HotpotQA 使用分层配置：

```text
benchmarks/.env.global
  全局 LLM/provider 默认配置
benchmarks/hotpotqa/.env.hotpotqa.base
  HotpotQA 共享的数据、语料、搜索和 guard 默认配置
--env benchmarks/hotpotqa/.env.hotpotqa.exploration
  exploration 阶段专属差异
--env benchmarks/hotpotqa/.env.hotpotqa.frozen
  frozen-evaluation 阶段专属差异
os.environ
  最高优先级的运行时覆盖
```

加载优先级为：

```text
.env.global < .env.hotpotqa.base < profile env < os.environ
```

对于 HotpotQA fullwiki，请在 `.env.hotpotqa.base` 中配置数据集与 raw corpus 路径：

```text
HOTPOT_DATASET_DIR=/path/to/hotpotqa_dataset
HOTPOT_WIKI_CORPUS_DIRNAME=enwiki-20171001-pages-meta-current-withlinks-abstracts
```

如果 `HOTPOT_DATASET_DIR` 直接指向 `fullwiki/` parquet 目录：

```text
HOTPOT_DATASET_DIR=/path/to/hotpotqa_dataset/fullwiki
HOTPOT_WIKI_CORPUS_DIR=/path/to/hotpotqa_dataset/enwiki-20171001-pages-meta-current-withlinks-abstracts
```

## Step 1: Smoke And Tune

先运行统一 smoke 路径。它会验证 env 加载、数据加载、检索、评估、artifact 写入和报告生成。该阶段只用于 exploration，不能作为最终论文结论。

```bash
python benchmarks/run_benchmark.py smoke-tune \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --limit 20 \
  --context-corpus-mode sample
```

如果需要在同一次 run 后做可选 baseline smoke：

```bash
python benchmarks/run_benchmark.py smoke-tune \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --limit 20 \
  --run-evaluation \
  --baselines bm25,naive_rag
```

smoke run 中的 `bm25` 和 `naive_rag` 是 quickstart-local baselines，与 frozen main experiment 中 paper-facing 的 `bm25_rag` 行严格区分。

当显式传入 `--baselines` 且评估不是 `--table-only` 时，evaluation 步骤会在论文表格路径之后额外打印终端版 `Baseline Final Report`。该 ASCII 汇总表包含 `Baseline`、`N`、`Acc`、`EM`、`F1`、`Cov`、`Evd`、`Avg`、`P95`、`Tok/Q`、`Fail` 和 `Notes`，因此不打开生成的 JSON 表格也能快速看到 smoke baseline 是否退化。

典型 exploration 输出：

```text
benchmarks/hotpotqa/output/exploration/runs/<run_id>/results/metrics.json
benchmarks/hotpotqa/output/exploration/runs/<run_id>/results/predictions.jsonl
benchmarks/hotpotqa/output/exploration/runs/<run_id>/reports/report.md
benchmarks/hotpotqa/output/exploration/runs/<run_id>/reports/validation.json
```

`quickstart_ok=True` 表示本地链路健康。`paper_ready=False` 是预期结果，因为 exploration artifacts 会被有意阻止进入论文结论。

## Step 2: Build Baseline Assets

Baseline 和竞品预处理属于 `assets` block，包括索引、embedding、图构建、生命周期可行性、setup cost、storage cost 和结构化失败原因。

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

输出：

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

查看 registry：

```bash
python benchmarks/run_benchmark.py status \
  --asset-registry benchmarks/hotpotqa/output/assets/asset_registry.jsonl \
  --benchmark hotpotqa
```

registry 是 append-only。失败 baseline 也会保留，并带有结构化原因，例如 `timeout`、`oom`、`disk_exceeded`、`api_budget_exceeded`、`dependency_missing`、`partial_index` 或 `index_validation_failed`。

## Step 3: Run The Frozen Main Experiment

`main` block 负责论文面对的路径：

```text
sampling / fixed IDs
→ optional frozen Sirchmunk run
→ baseline comparison
→ paper table
→ report
→ Gate 0-5 validation
→ main_summary.json
```

如果 Sirchmunk 结果已经存在，可直接传入：

```bash
python benchmarks/run_benchmark.py main \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sirchmunk-results benchmarks/hotpotqa/output/main/runs/<run_id>/results/predictions.jsonl \
  --run-artifact-dir benchmarks/hotpotqa/output/main/runs/<run_id> \
  --baselines bm25_rag,hybrid_rag,react \
  --asset-registry benchmarks/hotpotqa/output/assets/asset_registry.jsonl \
  --sampling-method stratified \
  --golden-n 2000 \
  --strata type,supporting_fact_bucket \
  --cache-mode cold \
  --generate-report \
  --strict
```

如果希望 control layer 先运行 Sirchmunk：

```bash
python benchmarks/run_benchmark.py main \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --run-sirchmunk \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sample_ids.json \
  --baselines bm25_rag,hybrid_rag,react \
  --asset-registry benchmarks/hotpotqa/output/assets/asset_registry.jsonl \
  --cache-mode cold \
  --generate-report \
  --strict
```

正式 main 运行中，只要 `--baselines` 触发了真实 baseline 执行，也会打印同一张 `Baseline Final Report`。如果 baseline telemetry/metadata 中存在 official EM/F1，则优先使用官方指标；否则 EM/F1 回退到 `judge_correct`，保证仅暴露 judge correctness 的 baseline 也能在终端汇总中呈现可读结果。Phase 0 缓存校验会在 adapter 身份或配置元数据不一致时自动重建 stale baseline JSONL。

Main outputs：

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

`main_summary.json` 是主实验的单一状态 artifact。`paper_ready=true` 要求所有 blocking gates 都通过。

## Step 4: Run Ablations

`ablation` block 创建冻结的 LENS/Sirchmunk 机制变体，并通过 P3 queue 基础设施执行。

```bash
python benchmarks/run_benchmark.py ablation \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sample_ids.json \
  --cache-mode cold \
  --max-combinations 16 \
  --replace
```

如果需要立即执行 queued variants：

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

输出：

```text
benchmarks/hotpotqa/output/ablation/
  ablation_spec.json
  variants.json
  ablation_summary.json
benchmarks/hotpotqa/output/queue/
  ablation_queue.json
  ablation_registry.jsonl
```

默认消融矩阵围绕 frozen baseline 一次改变一个机制：search mode、knowledge reuse、position prior、intent modulation 和 loop budget。

## Step 5: Report And Status

从已有 artifacts 重新生成报告：

```bash
python benchmarks/run_benchmark.py report \
  --run-dir benchmarks/hotpotqa/output/main/runs/<run_id> \
  --table-json benchmarks/hotpotqa/output/main/evaluation/paper_table.json \
  --output-dir benchmarks/hotpotqa/output/main/report \
  --title "HotpotQA Fullwiki ResearchOps Report" \
  --strict
```

查看 main summary：

```bash
python benchmarks/run_benchmark.py status \
  --summary benchmarks/hotpotqa/output/main/main_summary.json
```

查看 queue 状态：

```bash
python benchmarks/run_benchmark.py queue \
  --queue-path benchmarks/hotpotqa/output/queue/ablation_queue.json \
  status
```

## Quality Gates

control layer 会检查 Gate 0-5：

| Gate | Scope | Blocking evidence |
|---|---|---|
| Gate 0 | 参数 | benchmark、stage、cache mode、sampling args、asset args |
| Gate 1 | Assets | registry 可读、query-ready baseline assets、结构化失败 |
| Gate 2 | Sampling | fixed sample IDs、GoldenSet、sampling protocol、checksum |
| Gate 3 | Frozen run | `stage=frozen`、确定性 cache、protocol 有效 |
| Gate 4 | Evaluation | 样本数、系统、baseline comparison 完整性 |
| Gate 5 | Report | academic validator、表格/样本配对、provenance |

Frozen paper runs 必须满足：

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

统一 control layer 使用如下规范目录：

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

Scaling 现在可通过 `assets` block 进入：

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

Update readiness 也是 assets lifecycle governance 的一部分：

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

Scaling 和 update cost 应与 warm-query accuracy 分开报告。full-corpus index 未达到 `READY` 的系统，不应在没有明确 feasibility caveat 的情况下进入 warm-query quality baseline。

## Related-Work Lifecycle Baselines

LightRAG v1.3.6 被支持为 index-heavy related-work baseline，而不是无状态 QA API。它会为每个 `D_n/documents` snapshot 构建 LightRAG `working_dir`，记录 setup/index/storage metrics，并在动态语料更新时报告 `rebuild_required=true`。

使用 `lightrag_v136` 进行 SDK-backed lifecycle 复现：

```bash
python benchmarks/run_dynamic_evaluation.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --golden-n 2000 \
  --stages 500,1000,2000 \
  --materialize symlink \
  --run-baselines \
  --baselines lightrag_v136 \
  --lightrag-query-mode hybrid
```

Appendix sensitivity analysis 可分别运行各个 query mode：

```bash
python benchmarks/run_dynamic_evaluation.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --golden-n 2000 \
  --stages 500,1000,2000 \
  --materialize symlink \
  --run-baselines \
  --baselines lightrag_v136_naive,lightrag_v136_local,lightrag_v136_global,lightrag_v136_hybrid,lightrag_v136_mix
```

也可以手动做单个 GoldenSet 的评估：

```bash
python benchmarks/run_evaluation.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --baselines lightrag_v136 \
  --lightrag-query-mode hybrid \
  --sampling-method stratified \
  --golden-n 2000 \
  --strata type,supporting_fact_bucket
```

`lightrag_v1` 保留为外部 LightRAG 预测结果和 setup metrics 的 imported-prediction baseline。若希望 benchmark runner 本地构建索引并计入 lifecycle cost，请使用 `lightrag_v136`。

需要检查的 LightRAG lifecycle 字段包括：

```text
setup_seconds
index_build_seconds
storage_bytes
indexed_documents / expected_documents
partial_index
rebuild_required
query_ready_immediately
query_mode
working_dir
```

对于 v4 论文结论，LightRAG 应进入 related-work / lifecycle / appendix 表。除非实验明确扩展 baseline scope，否则它不属于最小主表。

## Advanced Direct Scripts

Direct scripts 仍然可用于调试和向后兼容。正常工作流优先使用 `run_benchmark.py`。

| Direct script | Preferred block | 何时直接使用 |
|---|---|---|
| `run_quickstart.py` | `smoke-tune` | 调试 quickstart 内部逻辑 |
| `run_sampling.py` | `main` | 手动检查 sampling distribution |
| `run_lifecycle_eval.py` | `assets` | 旧版 lifecycle 实验 |
| `run_scaling_study.py` | `assets scaling` | 旧版 scaling run |
| `run_evaluation.py` | `main` | 手动组装表格 |
| `run_dynamic_evaluation.py` | v4 dynamic protocol | 构建 `G_n/D_n` snapshots 和可选 dynamic baselines |
| `run_report.py` | `report` | 手动重新生成报告 |
| `run_queue.py` | `queue` | 低层队列调试 |
| `run_research_loop.py` | `smoke-tune` | exploration 和 badcase tuning |

手动 sampling 示例：

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

手动 research loop 示例：

```bash
python benchmarks/run_research_loop.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --limit 50 \
  --max-iter 5 \
  --dry-run
```

只有当 direct-script 输出保留 stage、sample IDs、checksum、config hash 和 manifest provenance 时，才应将其作为统一 block 的输入。

## Supported Benchmarks

支持的 benchmark 名称由 `framework/registry.py` 解析：

```text
hotpotqa
setup_cost
freshness
storage_overhead
source_fidelity
warm_reuse
```

Mechanism benchmarks 检验 QA accuracy 无法单独覆盖的主张：

| Benchmark | Claim |
|---|---|
| `setup_cost` | 启动和预处理成本 |
| `freshness` | 动态语料新鲜度 |
| `storage_overhead` | 额外存储 artifacts |
| `source_fidelity` | 到 raw sources 的可追溯性 |
| `warm_reuse` | Cache 和复用行为 |

机制实验 smoke 示例：

```bash
python benchmarks/run_benchmark.py smoke-tune \
  --benchmark setup_cost \
  --env benchmarks/setup_cost/.env.setup_cost \
  --limit 1 \
  --skip-report
```

## Interpreting Sampled Evaluation

对于 HotpotQA fullwiki，validation split 有 7,405 条样本。如果全量评估成本过高，应使用冻结的分层子集，而不是临时 `--limit`。

推荐主 sampled evaluation：

```text
n = 2000
strata = type + supporting_fact_bucket
answer_type = monitored distribution, not primary strata
```

推荐表述：

```text
We evaluate on a fixed stratified subset of the HotpotQA fullwiki validation split under an end-to-end raw-corpus protocol. All systems use the same sample IDs and sample-ID checksum, enabling paired uncertainty estimates.
```

不要把 sampled results 写成 full benchmark results。Published full-benchmark numbers 可以作为独立 reference rows，但除非拥有同一批 sample IDs 上的 raw predictions，否则它们不是 paired comparisons。

## Troubleshooting

- `paper_ready=false`：检查 `main_summary.json` 和 `report/validation.json` 中失败的 gates。
- `Gate 1` 失败：使用 `run_benchmark.py assets` 构建或验证 assets，并检查 `asset_registry.jsonl`。
- `Gate 2` 失败：提供 `--sample-ids-file`、`--sampling-protocol`，或使用冻结协议生成的 GoldenSet。
- `Gate 3` 失败：确认 `stage=frozen`、`cache-mode cold|compiled`，并关闭 eval feedback / memory updates。
- `Gate 5` 失败：使用 `run_benchmark.py report` 重新生成报告并检查 validator 错误。
- Env 文件缺失：从 examples 创建私有 profile，并把 secrets 放入被忽略的文件中。
- LightRAG v1.3.6 被跳过：安装 `lightrag-hku` 或 `v1.3.6` Git ref，然后用 `--baselines lightrag_v136` 重试。
- LightRAG partial index：检查 `working_dir`、`indexed_documents`、`expected_documents` 和 stage `D_n/documents` snapshot。
- Imported baseline coverage 低：确保 JSONL 对每个 frozen sample ID 都包含一条预测。
