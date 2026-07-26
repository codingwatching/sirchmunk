# Benchmarks ResearchOps 使用指南

本模块提供用于在研究级条件下评估 Sirchmunk 的实验基础设施。它面向这样一类实验场景：核心主张并不是简单证明某个问答系统在所有情况下都更准确，而是证明一个系统能够直接作用于动态原始数据，同时保持可追溯性、启动/预处理成本透明性和具有竞争力的答案质量。

默认参考流程是 HotpotQA fullwiki，但该框架本身并不绑定某一个 benchmark。同一套生命周期也支持面向机制的实验，例如启动/预处理成本（setup cost）、数据新鲜度（freshness）、存储开销（storage overhead）、源数据保真度（source fidelity）和缓存复用（warm reuse）。

## 设计哲学

benchmark 实验栈遵循 ResearchOps 哲学：每一个被报告的数字都应当能够从结构化 artifact 中复现、归因和证伪，而不是依赖日志片段或叙事性解释来重建。

因此，该框架将四类关注点明确分离：

- 执行：通过统一 runner 执行 benchmark adapter，并提供 checkpoint、retry、cache policy 和预算保护。
- 评估：在同一 GoldenSet 上比较不同系统，并使用共享 benchmark judge 对预测结果重新评分。
- 治理：区分探索运行与冻结评估运行，防止针对测试集的调参进入论文级结论。
- 报告：仅基于机器可读 artifacts 生成指标优先的表格和报告，并通过 validator 门控检查来源信息缺失或不公平对比。

对于论文实验，应将官方任务指标作为主要证据。对于 HotpotQA，official exact match 和 token F1 是主要答案质量指标。基于 LLM 的语义判定只能作为显式标记的辅助分析，不得替代主结果表中的官方指标。

## 目录结构

```text
benchmarks/
  framework/              # ResearchOps 执行、protocol、artifact、queue 和分析层
  evaluation/             # GoldenSet、baseline 评估、统计、表格、报告、验证
  baselines/              # BaselineAdapter 实现与导入预测适配器
  hotpotqa/               # HotpotQA adapter、judge、evidence、metrics、env examples
  setup_cost/             # 机制 benchmark：启动/预处理成本
  freshness/              # 机制 benchmark：动态数据新鲜度
  storage_overhead/       # 机制 benchmark：存储开销
  source_fidelity/        # 机制 benchmark：原始来源可追溯性
  warm_reuse/             # 机制 benchmark：warm cache / 复用行为
  run_queue.py            # P3 queue 和无人值守执行 CLI
  run_evaluation.py       # baseline 对比和论文表格 CLI
  run_lifecycle_eval.py   # full-corpus baseline 构建/索引可行性 CLI
  run_scaling_study.py    # 多规模可行性与摊销成本 CLI
  run_report.py           # 指标优先的报告生成 CLI
  run_research_loop.py    # 探索和研究循环 CLI
```

支持的 benchmark 名称由 `framework/registry.py` 解析。当前名称包括 `hotpotqa`、`setup_cost`、`freshness`、`storage_overhead`、`source_fidelity` 和 `warm_reuse`。

## 端到端流程

推荐流程包含四个阶段。这些阶段被有意隔离，以防止探索性优化污染冻结后的论文评估。

### 阶段 0：准备环境与数据

基于提供的 example 创建私有运行时环境文件。不要提交密钥。

```bash
cp benchmarks/.env.global.example benchmarks/.env.global
cp benchmarks/hotpotqa/env.hotpotqa.base.example benchmarks/hotpotqa/.env.hotpotqa.base
cp benchmarks/hotpotqa/env.hotpotqa.exploration.example benchmarks/hotpotqa/.env.hotpotqa.exploration
cp benchmarks/hotpotqa/env.hotpotqa.frozen.example benchmarks/hotpotqa/.env.hotpotqa.frozen
cp benchmarks/hotpotqa/env.hotpotqa.mock.example benchmarks/hotpotqa/.env.hotpotqa.mock
export LLM_API_KEY="..."
```

HotpotQA 使用分层配置：

```text
benchmarks/.env.global
  全局 LLM/provider 默认配置
benchmarks/hotpotqa/.env.hotpotqa.base
  HotpotQA 共享的数据路径、语料路径、搜索参数和 guard 默认值
--env benchmarks/hotpotqa/.env.hotpotqa.exploration
  仅保留 exploration 阶段差异项
--env benchmarks/hotpotqa/.env.hotpotqa.frozen
  仅保留 frozen evaluation 阶段差异项
os.environ
  最高优先级的运行时覆盖，例如 LLM_API_KEY
```

加载优先级为：

```text
.env.global < .env.hotpotqa.base < profile env < os.environ
```

对于 HotpotQA fullwiki，请在 `.env.hotpotqa.base` 中配置数据集和语料路径。`HOTPOT_DATASET_DIR` 可以指向 dataset 根目录，也可以直接指向 `fullwiki/` parquet 目录；若直接指向 `fullwiki/`，请显式设置 `HOTPOT_WIKI_CORPUS_DIR` 到 raw Wikipedia 语料目录。

```text
# 方式 A：dataset 根目录
HOTPOT_DATASET_DIR=/Users/jason/work/github/sirchmunk_work/data/hotpotqa_dataset
HOTPOT_WIKI_CORPUS_DIRNAME=enwiki-20171001-pages-meta-current-withlinks-abstracts

# 方式 B：直接指向 fullwiki parquet 目录（当前本地配置）
HOTPOT_DATASET_DIR=/Users/jason/work/github/sirchmunk_work/data/hotpotqa_dataset/fullwiki
HOTPOT_WIKI_CORPUS_DIR=/Users/jason/work/github/sirchmunk_work/data/hotpotqa_dataset/enwiki-20171001-pages-meta-current-withlinks-abstracts
```

当前本地期望目录结构：

```text
/Users/jason/work/github/sirchmunk_work/data/hotpotqa_dataset/
  fullwiki/
    validation-*.parquet
    test-*.parquet
    train-*.parquet
  enwiki-20171001-pages-meta-current-withlinks-abstracts/
    ... raw wiki files ...
```

exploration profile 用于冒烟测试和开发子集。frozen profile 仅用于论文级评估。base env 应承载共享配置；profile env 文件只保留阶段相关的覆盖项。

无需外部 API 的 smoke test 可使用私有 env 开启 deterministic mock LLM：

```text
HOTPOT_MOCK_LLM=true
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=<your-api-key>
LLM_MODEL_NAME=qwen3.7-plus
```

真实 LLM 运行时保持 `HOTPOT_MOCK_LLM=false`，并通过 shell 或私有 ignored env 文件设置真实 `LLM_API_KEY`，不要提交真实密钥。

### Mock Smoke Test（不调用外部 LLM）

使用被 git 忽略的私有 mock profile，可在 10 条样本上验证 runner / retrieval / judge / artifact 全链路：

```bash
printf 'skip\n' | python benchmarks/run_research_loop.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.mock \
  --limit 10 \
  --max-iter 1 \
  --dry-run \
  --log-level INFO
```

Mock run 的目标不是答案正确率；mock LLM 是确定性占位模型，预期 accuracy 可能为 0。成功标准是链路完整跑通：10/10 样本完成、corpus validation 通过、predictions/metrics/artifacts 写出、BadCase 分析生成。

### 阶段 1：探索或冒烟运行

探索阶段用于调试、集成检查和候选配置搜索。它应使用有限样本量，并且不得作为最终论文结论。

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

探索阶段可以使用 warm cache 和辅助诊断，但除非实验明确研究自适应行为，否则应保持 eval feedback 关闭。探索阶段 artifacts 适合用于调试和 badcase 分析，不应用于最终结论。共享的数据路径和搜索默认值应保留在 `.env.hotpotqa.base` 中；profile 文件只覆盖阶段相关的配置键。

### 阶段 2：冻结 Sirchmunk 评估（frozen evaluation）

冻结评估生成 Sirchmunk 的主结果。runner 和 protocol validator 会在该阶段执行更严格的约束。

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

冻结评估要求：

- `stage` 必须为 `frozen`。
- Cache mode 必须为 `cold` 或 `compiled`；`warm`、`none` 和 dry-run cache 会被拒绝。
- Eval feedback 必须关闭。
- 自适应 memory updates 必须关闭。如果启用 memory，则必须记录版本并设为只读。
- LLM judge 必须在主表中关闭，除非被显式标记为辅助指标。
- run artifact 必须记录样本数、sample ID checksum、config hash、git snapshot、system specs、dataset manifest、cache report、predictions 和 per-sample evaluation。

一次成功的冻结运行会在 benchmark 输出目录下写入 artifacts，例如：

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

### 阶段 3：Baseline 对比

使用 `run_evaluation.py` 在同一 GoldenSet 上比较 Sirchmunk 与 baselines。该框架会强制 sample ID 一致性，并记录 setup cost 以支持公平比较。

对于本地 baselines：

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

对于 LightRAG 或 GraphRAG 等导入式 baselines，需要同时提供预测结果文件和 setup metrics：

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

导入的预测 JSONL 应包含每个 sample ID 的一条预测：

```json
{"sample_id":"example-id","prediction":"answer text","elapsed":3.2,"tokens_used":1024}
```

setup metrics JSON 应报告预处理和索引构建成本：

```json
{
  "setup_seconds": 3600.0,
  "preprocessing_seconds": 1200.0,
  "index_build_seconds": 2400.0,
  "storage_bytes": 123456789,
  "indexed_documents": 5233329
}
```

论文级 baseline comparison 要求：

- 所有非 published systems 使用完全相同的 sample ID 集合；
- 每个 baseline 都提供 setup metrics；
- 导入式 baseline 的 prediction coverage 至少为 95%；
- 已分类的 baseline failures 低于 validator threshold；
- 在具备 raw per-sample predictions 的情况下使用配对统计检验。

### 阶段 3.5：Full-Corpus Feasibility 评估

对于 HotpotQA fullwiki 这类大规模语料，LightRAG、GraphRAG、RAPTOR 等重预处理 baselines 不能只看 query-time quality。预处理/建索引是否能在声明资源预算内完成，本身就是论文实验结论的一部分。若 baseline 在预算内 timeout / OOM / token budget exceeded，应保留在 feasibility 表中并标注结构化失败原因，而不是从主表中静默移除。

```bash
python benchmarks/run_lifecycle_eval.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
  --baselines bm25,naive_rag \
  --corpus-scale fullwiki \
  --build-timeout 86400 \
  --max-disk-bytes 500000000000
```

该命令会在 `lifecycle_eval/` 下输出：

```text
baseline_lifecycle.jsonl
<baseline>_latest.json
tables/feasibility_table.{md,tex,json}
lifecycle_summary.json
```

真实接入 LightRAG / GraphRAG / RAPTOR 时，使用 `module:factory` 形式：

```bash
--baselines bm25,my_lightrag_adapter:create_lightrag_v1
```

factory 应返回 `BaselineAdapter`，通常是 `IndexingSdkBaseline`。其中 `prepare_fn` 负责构建外部索引，`validate_fn` 负责验证 full-corpus index 是否 query-ready。

### 阶段 3.6：多规模 Scaling Study 与摊销成本

为了判断竞品失败是否来自语料规模，而不是单次实现问题，可以在多个 corpus scale 上运行同一 lifecycle 协议。

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

输出位于 `scaling_study/`：

```text
corpus_subsets/                 # deterministic subset manifests / symlink dirs
<scale>/lifecycle/              # per-scale lifecycle records
scaling_lifecycle_records.jsonl
scaling_metrics.json            # seconds_per_doc / bytes_per_doc / failure reason
amortized_cost_curves.json      # C_avg(Q) curves
scaling_study_summary.json
```

摊销生命周期成本定义为：

$$C_{\text{avg}}(Q) = \frac{C_{\text{build}} + C_{\text{update}}}{Q} + C_{\text{query}}$$

报告时应区分 full-corpus feasibility 与 warm-query quality。只有 full-corpus index 状态为 `READY` 的系统才能进入 warm-query 主表。

### 阶段 3.7：Dynamic Update 与 Ablation 规划

P2 工具提供动态文档实验和 LENS 内部机制消融的基础能力：

- `framework.dynamic_update.DynamicUpdateManager`：创建 add/delete/update 后的语料版本，并记录 update cost / rebuild-required 结果。
- `framework.ablation_matrix.default_lens_ablation_spec()`：生成保守的一次只改变一个轴的 LENS 消融组合，覆盖 search mode、knowledge reuse、position prior、intent modulation 和 loop budget。

动态更新实验应报告 update time、是否需要 full rebuild、freshness accuracy 和更新后的 query quality。消融实验应保持 frozen protocol，不应与 exploration 调参混用。

### 阶段 4：报告生成与验证

从 frozen run artifact 和 paper table JSON 生成指标优先的报告：

```bash
python benchmarks/run_report.py \
  --run-dir benchmarks/hotpotqa/output/frozen/runs/<run_id> \
  --table-json benchmarks/hotpotqa/output/paper_table/paper_table.json \
  --output-dir benchmarks/hotpotqa/output/paper_table/report \
  --title "HotpotQA Fullwiki ResearchOps Report"
```

report 生成器会生成：

```text
report.md
report.tex
validation.json
figures/
  accuracy_latency.svg
  setup_cost.svg
  storage_overhead.svg
```

只有当证据包适合论文级使用时，validator 才会返回 `passed=true`。如果 report 状态为 `BLOCKED`，则在解决所有错误级问题之前，不应将其用于论文结论。

## 质量门控

validator 会检查以下几类证据：

- Artifact 完整性：protocol、manifest、config snapshot、git snapshot、system specs、dataset manifest、metrics、predictions 和 per-sample evaluation。
- Frozen-stage 完整性：stage 为 frozen，cache policy 是确定性的，eval feedback 关闭，memory 不是自适应的，且启用 LLM judge 时必须是辅助指标。
- 样本配对：所有非 published systems 共享同一个 sample ID checksum。
- Baseline 公平性：setup metrics 存在，且 baseline failures 被显式分类。
- Imported baseline 覆盖率：prediction coverage 被报告，并满足最低阈值。
- 失败治理：timeout、budget-exceeded、prediction、judge 和 import-missing failures 会被显式暴露，而不是静默折叠进 answer quality。

## Research Loop 使用方式

`run_research_loop.py` 用于 exploration、badcase analysis 和配置搜索。它不是最终冻结评估结论的推荐入口。

```bash
python benchmarks/run_research_loop.py \
  --benchmark hotpotqa \
  --env benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --limit 50 \
  --max-iter 5 \
  --dry-run
```

对于 multi-benchmark exploration：

```bash
python benchmarks/run_research_loop.py \
  --multi \
  --add-bm hotpotqa=benchmarks/hotpotqa/.env.hotpotqa.exploration \
  --add-bm setup_cost=benchmarks/setup_cost/.env.setup_cost \
  --limit 30 \
  --shadow-fraction 0.10 \
  --dry-run
```

使用 research-loop 运行结果理解失败模式和候选配置。只有在 protocol、seed、sample policy、cache policy 和指标层级固定之后，才应将某个配置提升为 frozen evaluation。

## 机制性 Benchmarks

机制性 benchmarks 用于检验问答准确率（QA accuracy）单独无法覆盖的主张：

- `setup_cost`：启动和预处理成本。
- `freshness`：反映动态语料更新的能力。
- `storage_overhead`：不同方法所需的额外 artifacts。
- `source_fidelity`：到 raw source files 的可追溯性。
- `warm_reuse`：cache reuse 条件下的行为。

示例冒烟运行：

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

当论文主张涉及动态原始数据检索、setup cost、freshness 或 source fidelity 时，应将这些 mechanism benchmarks 与 QA metrics 一并报告。

## 结果解释指南

不要将单一总体准确率解释为充分证据。对于研究用途，至少应检查：

- official EM 和 token F1；
- evidence recall 和 source grounding；
- 延迟分布和 token 使用量；
- setup time、preprocessing time、index build time 和 storage bytes；
- 按类别统计的 failure counts；
- sample ID checksum 和 GoldenSet configuration；
- validator status 和 artifact provenance。

这种表述框架是有意保守的。它将实验视为可审计的 evidence package，而不是一次性的脚本执行结果。
