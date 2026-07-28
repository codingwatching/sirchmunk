<!-- markdownlint-disable MD033 -->

# Benchmarks ResearchOps 指南

本文档是 `benchmarks/` 模块的端到端操作指南。请把它读成一条用户旅程：先从 mock/smoke 实验开始，冻结评估 sample IDs，构建 baseline 生命周期证据，运行 frozen 主实验，最后重新生成正式报告。

常规命令入口是：

```bash
python benchmarks/run_benchmark.py <task> [options]
python benchmarks/run_benchmark.py --task <task> [options]
```

Direct scripts 仍然保留，用于调试和特殊流程。它们放在文档后半部分，避免打断主线。

## 主线总览

```text
mock/smoke exploration
→ frozen sample IDs
→ baseline assets
→ frozen main experiment
→ report/status
→ optional ablation or dynamic lifecycle appendix
```

| 阶段 | 命令 | 目的 | 能否作为论文结论？ |
|---|---|---|---|
| Mock/smoke | `run_benchmark.py smoke-tune` | 小样本探索、环境检查、报告 smoke、可选 baseline 对比 | 否 |
| 冻结样本 | `run_sampling.py create` | 创建固定分层 sample IDs 和 checksum | 仅作为抽样证据 |
| Assets | `run_benchmark.py assets` | 构建/校验 baseline 预处理、索引、存储和生命周期证据 | setup/lifecycle 证据 |
| Frozen main | `run_benchmark.py main` | 运行或组装 Sirchmunk 结果、运行主 baseline、生成表格/报告 | gate 通过后可以 |
| Report/status | `run_benchmark.py report/status` | 重新生成报告并检查 gate 状态 | 若绑定 frozen artifacts 则可以 |
| 可选 appendix | `run_benchmark.py ablation`、`run_dynamic_evaluation.py` | 机制消融与动态 `G_n/D_n` 生命周期实验 | appendix/ablation |

核心纪律很简单：exploration 可以小样本、可迭代；面向论文的结果必须使用冻结 sample IDs、确定性 frozen 设置，并保证所有系统使用同一批 sample IDs。

## Baseline 名称与适用范围

命令中每个实现只使用一个 canonical command value。

| 范围 | 推荐命令值 | 角色 |
|---|---|---|
| Quickstart lexical smoke | `bm25` | 快速回归检查用本地 lexical baseline |
| Quickstart local RAG smoke | `naive_rag` | 小规模本地 RAG smoke baseline |
| 论文 sparse RAG | `bm25_rag` | 主对比中的 fixed-chunk BM25 RAG 行 |
| 论文 hybrid RAG | `hybrid_rag` | BM25 + dense reciprocal-rank fusion RAG 行 |
| 论文 tool-use agent | `react` | 普通 ReAct/search-agent baseline |
| LightRAG SDK lifecycle | `lightrag_v136`、`lightrag_v136_<mode>` | index-heavy related-work/lifecycle baseline |
| Imported LightRAG v1 | `lightrag_v1` | 预计算 prediction/setup 导入 |
| Imported GraphRAG | `graphrag` | 预计算 prediction/setup 导入 |
| LENS ablation | `lens_full`、`lens_no_prior`、`lens_no_seq`、`lens_no_reuse` | 机制消融 |
| Custom adapter | `module:factory` | 高级自定义 `BaselineAdapter` factory |

推荐列表：

```text
smoke comparison: bm25,naive_rag,bm25_rag,hybrid_rag,react
paper main:       bm25_rag,hybrid_rag,react
asset build:      bm25,bm25_rag,naive_rag,react
LightRAG modes:   lightrag_v136_naive,lightrag_v136_local,lightrag_v136_global,lightrag_v136_hybrid,lightrag_v136_mix
```

`run_evaluation.py` 支持完整 paper-main 列表，包括 `hybrid_rag`。`run_baseline_assets.py` 负责其本地支持的 asset methods 的资产/生命周期记录；只有当 registry 已包含你希望 frozen gate 强制校验的每个 method 的 ready record 时，才把 `--asset-registry` 传给 `main`。

## 准备环境

安装 benchmark 依赖：

```bash
pip install -r requirements/core.txt -r requirements/benchmarks.txt
```

从示例文件创建私有 env 文件：

```bash
cp benchmarks/.env.global.example benchmarks/.env.global
cp benchmarks/hotpotqa/env.hotpotqa.base.example benchmarks/hotpotqa/.env.hotpotqa.base
cp benchmarks/hotpotqa/env.hotpotqa.exploration.example benchmarks/hotpotqa/.env.hotpotqa.exploration
cp benchmarks/hotpotqa/env.hotpotqa.frozen.example benchmarks/hotpotqa/.env.hotpotqa.frozen
```

Provider 凭证只能放在被忽略的私有文件或 shell 环境变量中：

```text
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen3.7-plus
LLM_API_KEY=<your-api-key>
```

在 `benchmarks/hotpotqa/.env.hotpotqa.base` 中配置 HotpotQA 数据路径：

```text
HOTPOT_DATASET_DIR=/path/to/hotpotqa_dataset
HOTPOT_WIKI_CORPUS_DIRNAME=enwiki-20171001-pages-meta-current-withlinks-abstracts
```

如果 `HOTPOT_DATASET_DIR` 直接指向 `fullwiki/` parquet 目录，还需要显式设置 raw wiki corpus：

```text
HOTPOT_DATASET_DIR=/path/to/hotpotqa_dataset/fullwiki
HOTPOT_WIKI_CORPUS_DIR=/path/to/hotpotqa_dataset/enwiki-20171001-pages-meta-current-withlinks-abstracts
```

配置加载优先级为：

```text
benchmarks/.env.global < benchmarks/hotpotqa/.env.hotpotqa.base < profile env < os.environ
```

Mock/smoke 使用 exploration profile。只有在 sample IDs 和运行设置冻结后，才使用 frozen profile。

## Step 1: Mock/Smoke Exploration

这一阶段回答新用户的第一个问题：“我的本地环境、数据路径、检索、评估、artifact 写入和报告生成是否能跑通？”它故意不作为论文证据。

运行最小 smoke：

```bash
python benchmarks/run_benchmark.py smoke-tune \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --limit 20 \
  --seed 42 \
  --max-iter 1 \
  --context-corpus-mode sample
```

运行覆盖当前本地/论文 baseline family 的完整 smoke 对比：

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

典型 smoke 输出：

```text
benchmarks/hotpotqa/output/exploration/runs/<run_id>/results/metrics.json
benchmarks/hotpotqa/output/exploration/runs/<run_id>/results/predictions.jsonl
benchmarks/hotpotqa/output/exploration/runs/<run_id>/reports/report.md
benchmarks/hotpotqa/output/exploration/runs/<run_id>/reports/validation.json
benchmarks/hotpotqa/output/exploration/quickstart_eval/paper_table.md
```

`quickstart_ok=True` 表示本地链路健康。`paper_ready=False` 是预期结果，因为 exploration artifacts 会被有意阻止进入论文结论。

## Step 2: Freeze The Sample IDs

Smoke 通过后，冻结评估集合。对于 HotpotQA fullwiki，推荐主 sampled protocol 是按 `type` 和 `supporting_fact_bucket` 分层的 `n=2000`，validation population 默认规模为 7,405。

```bash
python benchmarks/run_sampling.py create \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --method stratified \
  --target-n 2000 \
  --seed 42 \
  --strata type,supporting_fact_bucket \
  --allocation proportional \
  --min-per-stratum 1 \
  --expected-population-size 7405 \
  --output-dir benchmarks/hotpotqa/output/main/sampling
```

后续所有 frozen runs 都使用生成的 sample ID 文件：

```text
benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_2000_sample_ids.json
```

在运行昂贵系统前校验 manifest：

```bash
python benchmarks/run_sampling.py validate \
  --manifest benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_2000_manifest.json
```

不要用临时 `--limit` 替代这个文件来生成论文结论。只有所有系统使用相同 sample IDs 和 checksum 时，sampled result 才是有效的 paired comparison。

## Step 3: Build Baseline Assets

Assets 阶段记录生命周期证据：预处理、索引、存储、可行性、结构化失败原因，以及 Sirchmunk 的 no-index 行。它与 warm-query accuracy 分开。

构建当前本地支持的 asset set：

```bash
python benchmarks/run_benchmark.py assets \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
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

查看 registry：

```bash
python benchmarks/run_benchmark.py status \
  --asset-registry benchmarks/hotpotqa/output/assets/asset_registry.jsonl \
  --benchmark hotpotqa \
  --methods bm25,bm25_rag,naive_rag,react \
  --stage frozen \
  --reusable-only
```

典型 asset 输出：

```text
benchmarks/hotpotqa/output/assets/asset_registry.jsonl
benchmarks/hotpotqa/output/assets/asset_summary.json
benchmarks/hotpotqa/output/assets/lifecycle/baseline_lifecycle.jsonl
benchmarks/hotpotqa/output/assets/tables/feasibility_table.md
```

失败 baseline assets 也会保留，并带有结构化原因，例如 `timeout`、`oom`、`disk_exceeded`、`api_budget_exceeded`、`dependency_missing`、`partial_index` 或 `index_validation_failed`。

## Step 4: Run The Frozen Main Experiment

Frozen main 阶段使用固定 IDs 和 frozen 设置。你可以让 control layer 运行 Sirchmunk，也可以传入已有 Sirchmunk predictions JSONL。

运行 Sirchmunk 和当前完整 paper-main baseline set：

```bash
python benchmarks/run_benchmark.py main \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --run-sirchmunk \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_2000_sample_ids.json \
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

或从已有 Sirchmunk predictions 组装 frozen main：

```bash
python benchmarks/run_benchmark.py main \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sirchmunk-results benchmarks/hotpotqa/output/main/runs/<run_id>/results/predictions.jsonl \
  --run-artifact-dir benchmarks/hotpotqa/output/main/runs/<run_id> \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_2000_sample_ids.json \
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

如果启用 `--asset-registry`，frozen asset gate 会要求 registry 对 `--baselines` 声明的每个 method 都有 reusable asset record。只有 registry 对该列表完整时才传入它。

Main 输出：

```text
benchmarks/hotpotqa/output/main/runs/<run_id>/results/predictions.jsonl
benchmarks/hotpotqa/output/main/evaluation/paper_table.json
benchmarks/hotpotqa/output/main/evaluation/paper_table.md
benchmarks/hotpotqa/output/main/evaluation/paper_table.tex
benchmarks/hotpotqa/output/main/report/report.md
benchmarks/hotpotqa/output/main/report/validation.json
benchmarks/hotpotqa/output/main/main_summary.json
```

`main_summary.json` 是单一 run-status artifact。`paper_ready=true` 要求所有 blocking gates 都通过。

## Step 5: Regenerate Report And Inspect Status

从已有 frozen artifacts 重新生成报告：

```bash
python benchmarks/run_benchmark.py report \
  --run-dir benchmarks/hotpotqa/output/main/runs/<run_id> \
  --table-json benchmarks/hotpotqa/output/main/evaluation/paper_table.json \
  --output-dir benchmarks/hotpotqa/output/main/report \
  --title "HotpotQA Fullwiki ResearchOps Report" \
  --stage frozen \
  --strict
```

检查 frozen run summary：

```bash
python benchmarks/run_benchmark.py status \
  --summary benchmarks/hotpotqa/output/main/main_summary.json
```

只要 `--baselines` 触发真实 baseline 执行，终端会打印 `Baseline Final Report`，包含 `Baseline`、`N`、`Acc`、`EM`、`F1`、`Cov`、`Evd`、`Avg`、`P95`、`Tok/Q`、`Fail` 和 `Notes`，无需打开 JSON 表格即可快速查看结果。

## Quality Gates

Frozen paper runs 会经过 Gate 0-5 检查：

| Gate | 范围 | Blocking evidence |
|---|---|---|
| Gate 0 | 参数 | benchmark、stage、cache mode、sampling args、asset args |
| Gate 1 | Assets | 请求 asset reuse 时，registry 可读且 baseline assets 可复用 |
| Gate 2 | Sampling | fixed sample IDs、GoldenSet、sampling protocol、checksum |
| Gate 3 | Frozen run | `stage=frozen`、确定性 cache、有效 protocol |
| Gate 4 | Evaluation | 样本数、系统、baseline comparison 完整性 |
| Gate 5 | Report | academic validator、table/sample pairing、provenance |

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

<details>
<summary>Optional: Ablation</summary>

创建 frozen LENS/Sirchmunk 机制变体：

```bash
python benchmarks/run_benchmark.py ablation \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_2000_sample_ids.json \
  --cache-mode cold \
  --max-combinations 16 \
  --replace
```

立即执行 queued variants：

```bash
python benchmarks/run_benchmark.py ablation \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_2000_sample_ids.json \
  --cache-mode cold \
  --max-combinations 16 \
  --run \
  --max-concurrent 1 \
  --replace
```

核心 variants 是 `lens_full`、`lens_no_prior` 和 `lens_no_seq`。除非论文明确研究 warm-start amortization，否则 `lens_no_reuse` 保持 appendix-only。

</details>

<details>
<summary>Optional: Scaling, Update Readiness, And LightRAG Lifecycle</summary>

通过 assets task 运行 scaling：

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

运行 update-readiness governance：

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

只有需要 SDK-backed related-work lifecycle 行时才安装 LightRAG v1.3.6：

```bash
pip install git+https://github.com/HKUDS/LightRAG.git@v1.3.6
```

运行默认 LightRAG lifecycle mode：

```bash
python benchmarks/run_dynamic_evaluation.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --golden-n 2000 \
  --seed 42 \
  --stages 500,1000,2000 \
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

为 appendix sensitivity 跑全量 LightRAG query modes：

```bash
python benchmarks/run_dynamic_evaluation.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --golden-n 2000 \
  --seed 42 \
  --stages 500,1000,2000 \
  --strata type,supporting_fact_bucket \
  --materialize symlink \
  --background-ratio 3.0 \
  --background-seed 42 \
  --run-baselines \
  --baselines lightrag_v136_naive,lightrag_v136_local,lightrag_v136_global,lightrag_v136_hybrid,lightrag_v136_mix \
  --lightrag-max-files 0 \
  --lightrag-max-file-chars 300000
```

Scaling 和 update cost 应与 warm-query accuracy 分开报告。full-corpus index 未达到 `READY` 的系统，不应在没有 feasibility caveat 的情况下进入 warm-query baseline。

</details>

<details>
<summary>Advanced Direct Scripts</summary>

正常流程优先使用 `run_benchmark.py`。调试或 facade task 尚未覆盖的流程才直接使用下列脚本。

| Direct script | 推荐角色 | 何时直接使用 |
|---|---|---|
| `run_quickstart.py` | `smoke-tune` | 调试 quickstart 内部逻辑 |
| `run_sampling.py` | 冻结样本 | 创建/校验显式 sampling artifacts |
| `run_baseline_assets.py` | `assets` | 调试 asset registry 与 lifecycle records |
| `run_evaluation.py` | `main` | 手动组装表格或使用非默认 evaluation flags |
| `run_dynamic_evaluation.py` | Dynamic appendix | 构建 `G_n/D_n` snapshots 和 dynamic baselines |
| `run_report.py` | `report` | 手动重新生成报告 |
| `run_queue.py` | `queue` | 低层队列调试 |
| `run_research_loop.py` | Exploration | Badcase tuning 和 dry-run 分析 |

带 imported predictions 的手动 evaluation：

```bash
python benchmarks/run_evaluation.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sirchmunk-results benchmarks/hotpotqa/output/main/runs/<run_id>/results/predictions.jsonl \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_2000_sample_ids.json \
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
  --generate-report
```

只有 direct-script 输出保留 stage、sample IDs、checksum、config hash 和 manifest provenance 时，才将其作为统一 tasks 的输入。

</details>

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
  --seed 42 \
  --skip-report
```

## Troubleshooting

- `paper_ready=false`：检查 `benchmarks/hotpotqa/output/main/main_summary.json` 和 `benchmarks/hotpotqa/output/main/report/validation.json`。
- `Gate 1` 失败：只有当 registry 对每个 frozen `--baselines` method 都包含 ready reusable record 时，才传入 `--asset-registry`；否则先重建匹配 assets。
- `Gate 2` 失败：传入 frozen `sampling_stratified_42_2000_sample_ids.json`、sampling protocol 或合法 GoldenSet manifest。
- `Gate 3` 失败：确认 `stage=frozen`、`cache-mode cold|compiled`，并关闭 eval feedback / memory updates。
- `Gate 5` 失败：使用 `run_benchmark.py report` 重新生成报告并检查 validator 错误。
- Env 文件缺失：从 examples 创建私有 profile，并把 secrets 放入被忽略的文件中。
- LightRAG v1.3.6 被跳过：安装 `v1.3.6` Git ref，然后用 `--baselines lightrag_v136` 重试。
- Imported baseline coverage 低：确保每个 frozen sample ID 都正好有一行 prediction。
