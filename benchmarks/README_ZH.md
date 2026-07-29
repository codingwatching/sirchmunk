<!-- markdownlint-disable MD033 -->

# Benchmarks ResearchOps 指南

本文档是 `benchmarks/` 模块的端到端操作指南。请把它读成一条用户旅程：先从 mock/smoke 实验开始，冻结评估 sample IDs，构建 baseline 生命周期证据，构建动态 `G_n/D_n` artifacts，运行 frozen 主实验，最后重新生成正式报告。

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
→ dynamic G_n/D_n artifacts
→ frozen main experiment
→ report/status
→ optional ablation appendix
```

| 阶段 | 命令 | 目的 | 能否作为论文结论？ |
|---|---|---|---|
| Mock/smoke | `run_benchmark.py smoke-tune` | 小样本探索、环境检查、报告 smoke、可选 baseline 对比 | 否 |
| 冻结样本 | `run_sampling.py create` | 创建固定分层 sample IDs 和 checksum | 仅作为抽样证据 |
| Assets | `run_benchmark.py assets` | 构建/校验 baseline 预处理、索引、存储和生命周期证据 | setup/lifecycle 证据 |
| Dynamic G/D | `run_benchmark.py dynamic` | 构建 nested `G_n/D_n` sample/corpus bindings、可选 dynamic baselines 以及可选的 stale-index 对照臂 | stage checks 通过后可以 |
| Frozen main | `run_benchmark.py main` | 运行或组装 Sirchmunk 结果、运行主 baseline、生成表格/报告 | gate 通过后可以 |
| Report/status | `run_benchmark.py report/status` | 重新生成报告并检查 gate 状态 | 若绑定 frozen artifacts 则可以 |
| 可选 appendix | `run_benchmark.py ablation` | 机制消融 | appendix/ablation |

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

`HOTPOT_MAX_CONCURRENT` 控制两件事：LENS 运行时同时处理多少个样本，以及同时并行评估多少个竞品系统。单个竞品内部的样本并发是另一个开关 `HOTPOT_BASELINE_SAMPLE_CONCURRENT`，默认不设（`0`），即每个 `BaselineAdapter` 保留自己的声明，通常为串行。每次 baseline suite 运行开始时会打印生效值：

```text
[Suite] system-level concurrency=5 over 3 baseline(s); per-sample concurrency: bm25_rag=1, hybrid_rag=1, react=1
```

抬高 `HOTPOT_BASELINE_SAMPLE_CONCURRENT` 的意义在于：墙钟成本完全由单一竞品主导。在 `G_125_D_125` stage 上实测：

| Baseline | 平均延迟 | Tokens/样本 | 1,750 样本位下的串行耗时 |
|---|---|---|---|
| `bm25_rag` | 3.0s | 4.1K | 1.4h |
| `hybrid_rag` | 3.7s | 3.8K | 1.8h |
| `react` | 75.2s | 32.2K | **36.6h** |

由于各系统本来并行，ReAct 单独就决定了整个正式流水线的墙钟时间。由此引出两点。

第一，延迟只在相同并发度下才可跨系统比较。LENS 自身的单样本延迟本就是在 `HOTPOT_MAX_CONCURRENT` 信号量内测得的，所以串行的 baseline 与并发的 LENS 从未在同等条件下被测量。与其回避这一点，不如记录下来：每条结果都会写入 `query_budget.measured_sample_concurrency`；而与并发无关的成本列（`avg_tokens`、`avg_llm_calls`、`avg_oracle_calls`）仍作为成本主证据，延迟只是次要的、依赖环境的数字。

第二，抬高并发会消耗超时余量。ReAct 观测到的最慢样本为 99.3s，而 `SAMPLE_TIMEOUT_SECONDS=300`，约 3 倍余量。若竞争把某个样本拖到超过该值，它会被记为超时失败而不是慢成功，因此应逐步抬升并检查失败计数。查询路径共享单一可变实例的竞品可用 `supports_query_concurrency() -> False` 退出；LightRAG 已如此声明，无论配置如何都保持串行。

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

### Corpus Modes And Smoke Boundaries

`smoke-tune` 默认使用 `--context-corpus-mode sample` 来做快速 HotpotQA 健康检查。在该模式下，每个 sampled HotpotQA parquet `context` 会被物化成本地 `.txt` 文件，并且 sample 模式会强制 `HOTPOT_REQUIRE_CONTEXT_ANSWERABLE=true`。这会让 smoke 集合刻意变成闭合且可回答的语料。

sample-context 分数只能视为 pipeline health metrics。BM25、Naive RAG、BM25-RAG 和 Hybrid-RAG 会在 baseline preparation 阶段对 evaluation-set sample contexts 建索引，因此这些 smoke 行会带有 `evaluation_set_context_index` 风险。ReAct 不构建同一个全局索引，但仍然检索 gold-adjacent 的 per-sample context，因此也不是 raw-corpus 证据。

请明确区分这些 corpus modes：

| Mode | 适用场景 | 能否面向论文？ |
|---|---|---|
| `sample` | 在 answerable HotpotQA sample contexts 上快速 smoke/debug | 否；validator 会产生 `sample_context_corpus` 和 `evaluation_set_context_index` errors |
| `wiki` | Raw HotpotQA wiki corpus 检查 | 可以，前提是 frozen samples、pairing 和 gates 均通过 |
| `hybrid` | sample+wiki 诊断对照 | 不能直接与 raw-corpus claims 对比；validator 会产生 hybrid warning |

Evaluation tables 现在会记录 `corpus_provenance`、`corpus_risk` 和每个 baseline 的 `baseline_index_scope`。sample-context Markdown 表格也会打印显式 warning。面向论文的 runs 应优先使用 raw wiki 或 dynamic `G_n/D_n` snapshots，并要求 academic validator 没有 error-level corpus issue。

## Step 2: Freeze The Sample IDs

Smoke 通过后，冻结评估集合。对于 HotpotQA fullwiki，推荐主 sampled protocol 是按 `type` 和 `supporting_fact_bucket` 分层的 `n=500`，validation population 默认规模为 7,405。`n=500` 是建议上限：它已能把总体分层比例复现到 0.11 个百分点以内，同时保持所有 baseline 的查询成本可控。

抽样受 raw-corpus 同步性门控约束。parquet split 定义了问题及其 `supporting_facts` 标题，但 raw enwiki dump 必须真的包含这些文章，而 parquet 本身不记录它属于哪份 dump。因此 `create` 在冻结任何样本前，会先把每个被引用的文章对齐到 dump，当某个 supporting-fact 文章缺失时拒绝写出 sample IDs。任何时候都可以独立检查：

```bash
python benchmarks/run_sampling.py check-corpus-sync \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen
```

在官方 `enwiki-20171001-pages-meta-current-withlinks-abstracts` dump 上，validation split 完全闭合：

```text
shard_count                   = 15517
evidence_title_closure        = 100.0   (13781/13781，阻断性)
context_title_closure         = 100.0   (58293/58293，仅提示)
question_closure              = 100.0   (7405/7405 可解析)
passed                        = True
```

evidence 闭合是阻断性的，因为 supporting-fact 文章缺失的问题根本无法从快照中作答；context distractor 闭合会报告但不阻断，因为缺少 distractor 只会让快照略变简单。若接受“本次运行不具备主表资格”，可用 `--allow-corpus-desync` 强行冻结。

首次检查会扫描 dump 一次（约 30s），并在 `benchmarks/hotpotqa/.work/corpus_index/` 下缓存一份以 dump fingerprint 为键的标题索引；后续检查与 stage 构建在约 1s 内复用它。标题索引是由 dump 确定性派生的产物，因此刻意放在 `.cache` 之外，可在 cold-cache 清理后存活。`--rebuild-corpus-index` 强制重扫。dump 身份会以 `wiki_corpus_fingerprint` 记录在 dataset manifest 和 sample-ids metadata 中，因此替换、截断或扩展 dump 都会改变记录的 fingerprint，而不会静默通过。

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

后续所有 frozen runs 都使用生成的 sample ID 文件：

```text
benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_500_sample_ids.json
```

在运行昂贵系统前校验 manifest：

```bash
python benchmarks/run_sampling.py validate \
  --manifest benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_500_manifest.json
```

不要用临时 `--limit` 替代这个文件来生成论文结论。只有所有系统使用相同 sample IDs 和 checksum 时，sampled result 才是有效的 paired comparison。

## Step 3: Build Baseline Assets

Assets 阶段记录生命周期证据：预处理、索引、存储、可行性、结构化失败原因，以及 Sirchmunk 的 no-index 行。它是 Step 4 warm-query accuracy 之前的 frozen lifecycle preflight，二者职责分离。

在最终 main 实验前执行这个 frozen asset preflight：

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

将这条命令作为最终实验的默认 Step 3。`--limit 20` 只加载一个小 型 `golden_like` sample set，用于 baseline path inference 和 `baseline.prepare()` preflight；它不是最终评估样本数。最终 Step 4 样本量由 Step 2 的 frozen sample ID 文件控制，HotpotQA sampled main protocol 通常是 `n=500`。

最终路径的参数说明：

| 参数 | 最终命令取值 | 设置原因 |
|---|---|---|
| `--output-dir` | `benchmarks/hotpotqa/output` | 让 assets、main runs、evaluation tables 和 reports 都落在同一个 benchmark output root 下 |
| `--methods` | `bm25,bm25_rag,naive_rag,react` | 构建当前 assets facade 支持的 lifecycle rows，并包含 Sirchmunk 的 no-index row |
| `--limit` | `20` | 快速 frozen asset preflight；不要为了匹配 Step 4 而改成 `500` |
| `--corpus-scale` | `fullwiki` | 将 asset evidence 标记为面向 HotpotQA fullwiki |
| `--build-timeout` | `86400` | 对每个 baseline asset build 强制 24h wall-clock timeout |
| `--max-disk-bytes` | `500000000000` | 为 fullwiki asset feasibility 记录 500GB disk budget |
| `--max-ram-bytes`、`--max-llm-tokens`、`--max-api-cost-usd` | `0` | 本命令不设显式上限；只有资源受限运行才设正值 |
| `--retry-count` | `0` | 保持失败分类可复现 |
| `--stage frozen`、`--strict` | 启用 | 面向论文的 asset evidence 必须使用；strict 会在 blocked/failed assets 时返回非零退出码 |

在 raw wiki 模式下，小 `--limit` 通常会解析到同一个 fullwiki 语料目录，因此索引规模主要由 `--corpus-scale` 和文件上限控制。在 sample 或 hybrid 语料模式下，`--limit` 会改变被物化的 sample-context paths 数量，因此会改变 asset 输入规模。

默认不要把 Step 3 registry 传给 Step 4。下面最终 Step 4 命令使用 `bm25_rag,hybrid_rag,react`，而当前 assets facade 构建的是 `bm25,bm25_rag,naive_rag,react` 的 registry rows。只有当 registry 已经包含 Step 4 每个 exact method 的 ready record 时，才给 Step 4 附加 `--asset-registry`。

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

最终 frozen main 命令使用固定 sample IDs、cold cache、strict gates，以及核心 paper-main 在线 baseline set：`bm25_rag,hybrid_rag,react`。

运行 Sirchmunk 和最终核心 paper-main baselines：

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

如果 Sirchmunk predictions 已经存在，则用下面命令组装同一张 frozen main 表，不重新运行 Sirchmunk：

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

`lightrag_v1` 或 `graphrag` 等 related-work imports 不属于这条可直接执行的核心命令，因为它们需要预计算 prediction/setup 文件。等这些文件存在后再单独执行 assembly run 加入它们；如果需要非默认 SDK LightRAG 参数，请直接使用 `run_evaluation.py`。

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
| Gate 5 | Report | academic validator、table/sample pairing、corpus provenance/risk |

Frozen paper runs 必须满足：

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

cold cache 只有在显式授权清理时才会真正生效：对 frozen `main`/ablation 运行需导出 `CACHE_ALLOW_CLEAR=true`，否则 cache report 会记录 `cold cache requested but allow_clear=False`，Gate 5 会阻断该运行。清理范围由 `CacheManager` 限制在 benchmark work path 下的缓存目录；由 dump 派生的 corpus title index 位于 `.cache` 之外，刻意在 cold 清理后存活。`CACHE_DRY_RUN=true` 只记录动作不删除。

Academic validator 会将 corpus boundary violations 视为 blocking issues。`sample_context_corpus` 表示表格使用了 answerable HotpotQA sample contexts，而不是 raw corpus。`evaluation_set_context_index` 表示至少一个 baseline 索引了 evaluation-set sample context。`hybrid_context_corpus` 是 warning，表示 sample+wiki 结果需要单独措辞，不能直接与 raw-corpus rows 对比。

<details>
<summary>Optional: Ablation</summary>

创建 frozen LENS/Sirchmunk 机制变体：

```bash
python benchmarks/run_benchmark.py ablation \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --sample-ids-file benchmarks/hotpotqa/output/main/sampling/sampling_stratified_42_500_sample_ids.json \
  --cache-mode cold \
  --max-combinations 16 \
  --replace
```

立即执行 queued variants：

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
  --materialize copy \
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
  --materialize copy \
  --limit 20 \
  --seed 42 \
  --bm25-max-files 20000 \
  --naive-rag-max-files 5000
```

只有需要 SDK-backed related-work lifecycle 行时才安装 LightRAG v1.3.6：

```bash
pip install git+https://github.com/HKUDS/LightRAG.git@v1.3.6
```

通过 dynamic task 运行默认 LightRAG lifecycle mode：

```bash
python benchmarks/run_benchmark.py dynamic \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --golden-n 500 \
  --seed 42 \
  --stages 125,250,500 \
  --strata type,supporting_fact_bucket \
  --materialize copy \
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
python benchmarks/run_benchmark.py dynamic \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --golden-n 500 \
  --seed 42 \
  --stages 125,250,500 \
  --strata type,supporting_fact_bucket \
  --materialize copy \
  --background-ratio 3.0 \
  --background-seed 42 \
  --run-baselines \
  --baselines lightrag_v136_naive,lightrag_v136_local,lightrag_v136_global,lightrag_v136_hybrid,lightrag_v136_mix \
  --lightrag-max-files 0 \
  --lightrag-max-file-chars 300000
```

Scaling 和 update cost 应与 warm-query accuracy 分开报告。full-corpus index 未达到 `READY` 的系统，不应在没有 feasibility caveat 的情况下进入 warm-query baseline。

Dynamic task 默认包含 paper-facing baselines `bm25_rag,hybrid_rag,react`；显式传入 `--baselines` 时以 CLI 参数为准。dynamic task 不会隐式运行 LENS：面向论文的 `G_n/D_n` 主表需要显式加入 `lens_full`（推荐 `--baselines bm25_rag,hybrid_rag,react,lens_full`），这也为 stale-index arm 提供了实测的 index-free LENS 对照行。面向论文的运行请使用 `--materialize copy`：基于 ripgrep 的检索（LENS 的 rga 通道、ReAct 关键词搜索）不会跟随符号链接，symlink 快照会静默致盲所有 grep 系统。`--baseline-max-files` 需不小于最大 `D_n` 快照的文件数，避免固定索引 baseline 把 evidence 目录截断在索引之外。输出包括：

```text
benchmarks/hotpotqa/output/dynamic_eval/tables/dynamic_main_results.*
benchmarks/hotpotqa/output/dynamic_eval/tables/lifecycle_main.*
benchmarks/hotpotqa/output/dynamic_eval/tables/budget_quality.*
benchmarks/hotpotqa/output/dynamic_eval/tables/update_readiness.*
benchmarks/hotpotqa/output/dynamic_eval/tables/snapshot_audit.*
```

### Stale-index 对照臂

`update_readiness` 只记录系统是否*声明*需要重建索引，无法说明这个重建要求在答案质量上的代价。加上 `--stale-index-arm` 可以直接测量该代价：

```bash
python benchmarks/run_benchmark.py dynamic \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --golden-n 500 \
  --seed 42 \
  --stages 125,250,500 \
  --strata type,supporting_fact_bucket \
  --materialize copy \
  --run-baselines \
  --baselines bm25_rag,hybrid_rag,react \
  --stale-index-arm \
  --staleness-max-samples 200
```

对每个跃迁 `D_{n-1} -> D_n`，该臂只在新增问题集（`delta = G_n \ G_{n-1}`，其 supporting evidence 文章仅存在于 `D_n`）上对比两次运行：

| 臂 | 索引构建于 | 查询语料 | 含义 |
|---|---|---|---|
| fresh | `D_n` | `D_n` | 已付出重建成本 |
| stale | `D_{n-1}` | `D_n` | 语料已增长，索引未重建 |

stale 臂复用上一 stage 的系统实例并跳过其 prepare 步骤，因此 index-heavy 系统只能从触及不到新证据的旧索引作答，而 index-free 系统在查询时读取当前语料。预期解读：

| 系统类别 | 成员 | 预期 `Ev.Rec Gap` |
|---|---|---|
| `index_dependent` | `bm25_rag`、`hybrid_rag`、`lightrag_v136*` | 为正 |
| `index_free` | `react`、`lens_full`、`lens_no_prior`、`lens_no_seq` | 接近零 |

接近零的那一行是实测对照结果，而不是假设，因此该臂会对所有被请求的系统执行，包括 LENS 自身。查询预算有限时可用 `--staleness-max-samples` 限制每个跃迁的 delta 问题数。

新增产出：

```text
benchmarks/hotpotqa/output/dynamic_eval/tables/stale_index_quality.*
benchmarks/hotpotqa/output/dynamic_eval/runs/<stage>/staleness/baseline_<name>.jsonl
benchmarks/hotpotqa/output/dynamic_eval/runs/<stage>/stage_records/<name>_staleness_record.json
```

每条 staleness 记录都固定了 `from_corpus_checksum`、`to_corpus_checksum`、`delta_sample_id_checksum` 与 `stale_index_prepared_on`，因此报告出的落差可追溯到唯一的索引状态和唯一的 delta 问题集。`dynamic_eval_manifest.json` 会记录 `stale_index_arm`，并按系统类别聚合 `staleness_summary`。

### 嵌套 stage 的抽样保真度

分层的父集只能保证在它自身规模上分层比例正确。从随机洗牌后的父序直接切前缀，会使每个更小的 stage 退化为简单随机子样本，从而偏离总体分布，甚至整层丢失稀有层。因此嵌套 stage 改用分层均衔序（stratum-balanced order）推导：每个层的成员均匀铺开在序列上，使每个 stage 既保持比例，又仍是下一个 stage 的严格子集。

在 HotpotQA fullwiki validation 总体上实测（7,405 题，`type` x `supporting_fact_bucket` 共 8 个层，最稀有层 0.32%）：

| Stage | 洗牌父序前缀 | 分层均衔序 |
|---|---|---|
| `G_125` | 漂移 7.30pp，2 个层为空 | 漂移 0.70pp，0 空层 |
| `G_250` | 漂移 2.90pp，1 个层为空 | 漂移 0.30pp，0 空层 |
| `G_500` | 漂移 0.11pp，0 空层 | 漂移 0.11pp，0 空层 |

每个 stage 都会在 `nested_sample_manifest.json` 和各自的 sample-ids 文件中记录 `strata_distribution`、`proportion_delta_by_stratum`、`max_abs_proportion_delta` 与 `empty_strata`，并用 `reference_scope` 标明对比基准（父 manifest 记录了总体分布时为 `population`）。学术 validator 在漂移超过 5pp 时告警，因此这些数值是可直接校验的，而不是假设。仅当需要精确复现旧的 parent-order 产物时，才向 `derive_nested_sample_sets` 传入 `balance_strata=False`。

### Raw-corpus 同步性门控

dynamic task 在构建任何快照前，会对已冻结的父集重新检查语料同步性，并将报告以 `corpus_sync` 记录在 `dynamic_eval_manifest.json` 中。supporting-fact 文章缺失会以显式同步错误中止运行，而不是在后续快照构建中才崩溃；`--allow-corpus-desync` 将其降级为非阻断告警，并标记本次运行不具备主表资格。`--rebuild-corpus-index` 强制重扫 dump。快照标题解析复用同一份缓存索引，因此 evidence 文章只从包含它的那一个 shard 读取，而不是流式扫描整个 dump。

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
| `run_dynamic_evaluation.py` | `dynamic` | 调试 `G_n/D_n` snapshots、dynamic baselines 和 stale-index 对照臂 |
| `run_report.py` | `report` | 手动重新生成报告 |
| `run_queue.py` | `queue` | 低层队列调试 |
| `run_research_loop.py` | Exploration | Badcase tuning 和 dry-run 分析 |

带 imported predictions 的手动 evaluation：

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

只有 direct-script 输出保留 stage、sample IDs、checksum、config hash、manifest provenance 和 corpus provenance/risk 时，才将其作为统一 tasks 的输入。

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
- `Gate 2` 失败：传入 frozen `sampling_stratified_42_500_sample_ids.json`、sampling protocol 或合法 GoldenSet manifest。
- `Gate 3` 失败：确认 `stage=frozen`、`cache-mode cold|compiled`，并关闭 eval feedback / memory updates。
- `Gate 5` 失败：使用 `run_benchmark.py report` 重新生成报告并检查 validator 错误。
- Env 文件缺失：从 examples 创建私有 profile，并把 secrets 放入被忽略的文件中。
- LightRAG v1.3.6 被跳过：安装 `v1.3.6` Git ref，然后用 `--baselines lightrag_v136` 重试。
- Imported baseline coverage 低：确保每个 frozen sample ID 都正好有一行 prediction。
