#!/usr/bin/env python3
"""benchmarks/run_evaluation.py — 竞品横向评估 CLI

此脚本与 run_research_loop.py 完全独立：
  - run_research_loop.py : Sirchmunk 自改进循环（迭代优化自身）
  - run_evaluation.py    : 横向对比（固定测试集，生成论文表格）

使用方式::

    # 1. 导入竞品预测 JSONL + 本文结果 → 生成完整比较表格
    python benchmarks/run_evaluation.py \
      --benchmark hotpotqa \
      --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
      --import-baseline "GPT-4o (zero-shot)=output/gpt4o_preds.jsonl" \
      --import-baseline-setup "GPT-4o (zero-shot)=output/gpt4o_setup_metrics.json" \
      --sirchmunk-results benchmarks/hotpotqa/output/results_YYYYMMDD.jsonl \
      --output-dir benchmarks/hotpotqa/output/paper_table/

    # 2. 只填写已发表数字（不重新 Judge，直接写入表格）
    python benchmarks/run_evaluation.py \\
      --benchmark hotpotqa \\
      --env benchmarks/hotpotqa/.env.hotpotqa.frozen \\
      --import-published "Reported System:acc=45.0,cov=80.0,lat=5.2" \\
      --sirchmunk-results benchmarks/hotpotqa/output/results_YYYYMMDD.jsonl \\
      --output-dir benchmarks/hotpotqa/output/paper_table/

    # 3. 仅生成表格（不运行任何 baseline，纯汇聚已有结果）
    python benchmarks/run_evaluation.py \\
      --benchmark hotpotqa \\
      --env benchmarks/hotpotqa/.env.hotpotqa.frozen \\
      --sirchmunk-results benchmarks/hotpotqa/output/results_YYYYMMDD.jsonl \\
      --table-only \\
      --output-dir benchmarks/hotpotqa/output/paper_table/
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import sys
from pathlib import Path

# ── sys.path 注入 ─────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).parent.resolve()   # benchmarks/
_PROJECT_ROOT = _SCRIPT_DIR.parent               # sirchmunk/
_SRC = _PROJECT_ROOT / "src"

for _p in (str(_SCRIPT_DIR), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


from framework.registry import load_benchmark_adapter, supported_benchmarks  # noqa: E402
from evaluation.sampling_protocol import (  # noqa: E402
    DEFAULT_HOTPOTQA_POPULATION_SIZE,
    DEFAULT_HOTPOTQA_STRATA,
    create_sampling_protocol,
    load_sampling_protocol,
    validate_sampling_manifest,
    write_sample_ids,
)


def _load_bm_adapter(benchmark: str, env_file: str):
    """加载 BenchmarkAdapter（通过共享registry）。"""
    return load_benchmark_adapter(benchmark, env_file)


def _parse_published(spec: str) -> dict:
    """解析 'Name:acc=29.3,cov=100.0,lat=5.2' 格式。"""
    name, _, rest = spec.partition(":")
    metrics = {}
    for kv in rest.split(","):
        kv = kv.strip()
        if "=" in kv:
            k, _, v = kv.partition("=")
            metrics[k.strip()] = float(v.strip())
    return {"name": name.strip(), **metrics}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="竞品横向评估 — 生成论文比较表格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--benchmark", "-b", required=True,
                   choices=supported_benchmarks(),
                   help="目标 benchmark 名称")
    p.add_argument("--env", "-e", required=True,
                   help=".env 配置文件路径")
    p.add_argument("--sirchmunk-results", default=None, dest="sirchmunk_results",
                   help="本文系统（LENS/Sirchmunk）的结果 JSONL 文件路径")
    p.add_argument("--baselines", default="",
                   help=(
                       "运行的竞品列表（逗号分隔）。Paper main 推荐 bm25_rag,hybrid_rag,react；"
                       "quickstart/local smoke 可使用 bm25,naive_rag；"
                       "related-work/lifecycle: lightrag_v136 或 lightrag_v136_<mode>；"
                       "imported: lightrag_v1, graphrag；ablation: lens_full,lens_no_prior,lens_no_seq,lens_no_reuse；"
                       "custom: module:factory。Long-context baseline 当前明确不纳入实现范围。"
                   ))
    p.add_argument("--import-baseline", action="append", dest="import_baseline",
                   metavar="NAME=PATH",
                   help="导入预计算预测并重新 Judge（可多次）")
    p.add_argument("--import-baseline-setup", action="append", dest="import_baseline_setup",
                   metavar="NAME=PATH",
                   help="导入预计算baseline的setup metrics JSON（与--import-baseline同名匹配，可多次）")
    p.add_argument("--import-published", action="append", dest="import_published",
                   metavar="'Name:acc=XX,cov=XX,lat=XX'",
                   help="直接导入已发表数字（无需 Judge，可多次）")
    p.add_argument("--golden-n", type=int, default=150, dest="golden_n",
                   help="GoldenSet 大小（默认 150，0=全量；stratified 时推荐 500，也是采样协议的建议上限）")
    p.add_argument("--golden-seed", type=int, default=42, dest="golden_seed",
                   help="GoldenSet 随机种子（默认 42）")
    p.add_argument("--sampling-method", default="simple_random",
                   choices=["simple_random", "stratified", "full", "diagnostic_rare", "fixed_ids"],
                   dest="sampling_method",
                   help="GoldenSet抽样方法；论文主实验推荐 stratified")
    p.add_argument("--strata", default=",".join(DEFAULT_HOTPOTQA_STRATA),
                   help="stratified抽样的分层键，逗号分隔；默认 type,supporting_fact_bucket")
    p.add_argument("--sampling-allocation", default="proportional",
                   choices=["proportional", "equal", "uniform"],
                   dest="sampling_allocation",
                   help="stratified抽样的层内分配策略")
    p.add_argument("--min-per-stratum", type=int, default=1, dest="min_per_stratum",
                   help="stratified抽样每个非空stratum最少样本数")
    p.add_argument("--sampling-protocol", default="", dest="sampling_protocol",
                   help="已冻结的sampling protocol JSON路径；提供后覆盖CLI抽样参数")
    p.add_argument("--sample-ids-file", default="", dest="sample_ids_file",
                   help="固定sample IDs JSON路径；提供后使用fixed_ids协议")
    p.add_argument("--expected-population-size", type=int, default=0, dest="expected_population_size",
                   help="期望总体规模；HotpotQA fullwiki validation默认7405")
    p.add_argument("--sampling-report-dir", default="", dest="sampling_report_dir",
                   help="额外写出sampling protocol/manifest/sample_ids的目录")
    p.add_argument("--create-golden-only", action="store_true", dest="create_golden_only",
                   help="只创建并校验GoldenSet，不运行/汇聚baseline")
    p.add_argument("--output-dir", default=None, dest="output_dir",
                   help="输出目录（默认 benchmarks/{benchmark}/output/paper_table/）")
    p.add_argument("--table-only", action="store_true", dest="table_only",
                   help="跳过 baseline 运行，仅汇聚已有结果生成表格")
    p.add_argument("--ours-name", default=None, dest="ours_name",
                   help="本文系统在表格中的展示名（默认 'LENS (ours)'）")
    p.add_argument("--caption", default="",
                   help="LaTeX 表格标题")
    p.add_argument("--lightrag-predictions", default="", dest="lightrag_predictions",
                   help="LightRAG v1 预计算预测 JSONL 路径")
    p.add_argument("--lightrag-setup-metrics", default="", dest="lightrag_setup_metrics",
                   help="LightRAG v1 setup metrics JSON 路径")
    p.add_argument("--lightrag-query-mode", default="hybrid", dest="lightrag_query_mode",
                   choices=["naive", "local", "global", "hybrid", "mix"],
                   help="LightRAG v1.3.6 SDK baseline query mode，默认 hybrid")
    p.add_argument("--lightrag-working-dir", default="", dest="lightrag_working_dir",
                   help="LightRAG v1.3.6 working_dir；为空时按benchmark/stage work_path隔离")
    p.add_argument("--lightrag-max-files", type=int, default=0, dest="lightrag_max_files",
                   help="LightRAG v1.3.6 最多索引文件数，0=不限制")
    p.add_argument("--lightrag-max-file-chars", type=int, default=300000, dest="lightrag_max_file_chars",
                   help="LightRAG v1.3.6 单文件最大读取字符数")
    p.add_argument("--graphrag-predictions", default="", dest="graphrag_predictions",
                   help="GraphRAG 预计算预测 JSONL 路径")
    p.add_argument("--graphrag-setup-metrics", default="", dest="graphrag_setup_metrics",
                   help="GraphRAG setup metrics JSON 路径")
    p.add_argument("--bm25-max-files", type=int, default=20000, dest="bm25_max_files",
                   help="BM25本地索引最多读取文件数")
    p.add_argument("--naive-rag-max-files", type=int, default=5000, dest="naive_rag_max_files",
                   help="NaiveRAG本地索引最多读取文件数")
    p.add_argument("--hybrid-max-files", type=int, default=5000, dest="hybrid_max_files",
                   help="Hybrid-RAG最多读取文件数")
    p.add_argument("--hybrid-bm25-top-k", type=int, default=20, dest="hybrid_bm25_top_k",
                   help="Hybrid-RAG BM25候选chunk数")
    p.add_argument("--hybrid-dense-top-k", type=int, default=20, dest="hybrid_dense_top_k",
                   help="Hybrid-RAG dense候选chunk数")
    p.add_argument("--hybrid-final-top-k", type=int, default=5, dest="hybrid_final_top_k",
                   help="Hybrid-RAG最终送入LLM的chunk数")
    p.add_argument("--hybrid-dense-backend", default="hash", dest="hybrid_dense_backend",
                   choices=["hash", "sirchmunk", "sirchmunk_embedding", "embedding_util"],
                   help="Hybrid-RAG dense backend；默认hash保持可复现，sirchmunk_embedding使用项目EmbeddingUtil")
    p.add_argument("--hybrid-dense-dim", type=int, default=256, dest="hybrid_dense_dim",
                   help="Hybrid-RAG hash dense backend维度")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   dest="log_level")
    p.add_argument("--baseline-sample-timeout", type=float, default=0.0, dest="baseline_sample_timeout",
                   help="单个 baseline 样本预测/Judge 的超时秒数，0=不限制")
    p.add_argument("--baseline-max-runtime", type=float, default=0.0, dest="baseline_max_runtime",
                   help="单个 baseline 总运行预算秒数，0=不限制")
    p.add_argument("--baseline-max-total-tokens", type=int, default=0, dest="baseline_max_total_tokens",
                   help="单个 baseline token 预算，0=不限制")
    p.add_argument("--baseline-max-api-cost-usd", type=float, default=0.0, dest="baseline_max_api_cost_usd",
                   help="单个 baseline API 成本预算（美元），0=不限制")
    p.add_argument("--baseline-max-disk-bytes", type=int, default=0, dest="baseline_max_disk_bytes",
                   help="baseline 输出目录最大磁盘用量，0=不限制")
    p.add_argument("--baseline-min-free-disk-bytes", type=int, default=0, dest="baseline_min_free_disk_bytes",
                   help="baseline 输出目录所在磁盘最小剩余字节数，0=不限制")
    p.add_argument("--generate-report", action="store_true", dest="generate_report",
                   help="生成metric-first学术报告与质量门控结果")
    p.add_argument("--run-artifact-dir", default="", dest="run_artifact_dir",
                   help="Sirchmunk run artifact目录（包含protocol/manifest/results）")
    p.add_argument("--report-output-dir", default="", dest="report_output_dir",
                   help="报告输出目录（默认 output-dir/report）")
    p.add_argument("--context-corpus-provenance", default="", dest="context_corpus_provenance",
                   help="Corpus provenance for this evaluation: sample, wiki, hybrid, dynamic_snapshot, etc.")
    p.add_argument("--context-corpus-risk", default="", dest="context_corpus_risk",
                   help="Comma-separated corpus risk tags, e.g. oracle_sample_context,evaluation_set_context_index")
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    _setup_logging(args.log_level)
    logger = logging.getLogger("run_evaluation")

    env_file = str(Path(args.env).resolve())
    if not Path(env_file).exists():
        logger.error("env 文件不存在: %s", env_file)
        return 1

    try:
        bm_adapter = _load_bm_adapter(args.benchmark, env_file)
    except Exception as exc:
        logger.error("加载 BenchmarkAdapter 失败: %s", exc)
        return 1

    # 输出目录
    out_dir = args.output_dir or str(
        Path(_SCRIPT_DIR) / args.benchmark / "output" / "paper_table"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    baseline_dir = str(Path(out_dir) / "baselines")

    ours_name = args.ours_name or "LENS (ours)"
    baseline_guard_config = _baseline_guard_config(args)
    corpus_metadata = _evaluation_corpus_metadata(args, bm_adapter)
    if corpus_metadata.get("corpus_provenance") == "sample":
        logger.warning(
            "Evaluation uses sample-context corpus (%s); results are smoke health checks, not raw-corpus claims.",
            corpus_metadata.get("corpus_risk", ""),
        )
    if args.sampling_protocol:
        logger.info("Using --sampling-protocol=%s; CLI sampling-method/golden-n/strata options are ignored.", args.sampling_protocol)
    sampling_protocol = _build_sampling_protocol(args, bm_adapter)

    # ── 初始化 PaperTableGenerator ──────────────────────────────────
    from evaluation.table_generator import PaperTableGenerator
    from framework.runner import UnifiedExperimentRunner

    gen = PaperTableGenerator(
        benchmark_name=args.benchmark.upper(),
        our_system_name=ours_name,
    )
    explicit_baseline_report_rows = []

    question_type_key = "question_type"
    try:
        question_type_key = bm_adapter.get_analysis_schema().get("primary_group_key", "question_type")
    except Exception:
        pass

    golden_set = None
    if args.create_golden_only or args.sirchmunk_results or (args.baselines and not args.table_only) or (args.import_baseline and not args.table_only):
        _, golden_set = _get_or_create_golden_set(args, bm_adapter, sampling_protocol)
        _attach_sampling_metadata(gen, golden_set, corpus_metadata)
        _write_sampling_report(args, golden_set)
        logger.info(
            "GoldenSet: %d questions seed=%d checksum=%s method=%s",
            golden_set.n_questions,
            golden_set.seed,
            golden_set.sample_id_checksum(),
            golden_set.sampling_protocol.get("method", "unknown"),
        )
        if args.create_golden_only:
            print(json.dumps(_golden_summary(golden_set), indent=2, ensure_ascii=False))
            return 0

    # ── 加载本文结果 ─────────────────────────────────────────────────
    if args.sirchmunk_results:
        results_path = str(Path(args.sirchmunk_results).resolve())
        if not Path(results_path).exists():
            logger.error("sirchmunk-results 文件不存在: %s", results_path)
            return 1
        logger.info("加载本文结果: %s", results_path)
        sirchmunk_results = UnifiedExperimentRunner.load_results_from_jsonl(results_path)
        if golden_set is not None:
            golden_set.verify_results_sample_ids(sirchmunk_results, system_name=ours_name)
        gen.add_system_results(
            system_name=ours_name,
            results=sirchmunk_results,
            is_ours=True,
            question_type_key=question_type_key,
        )
        logger.info("本文系统: %d 条结果", len(sirchmunk_results))

    # ── 运行真实 / SDK 竞品 ─────────────────────────────────────────
    if args.baselines and not args.table_only:
        from evaluation.suite import BaselineEvaluationSuite

        if golden_set is None:
            _, golden_set = _get_or_create_golden_set(args, bm_adapter, sampling_protocol)
            _attach_sampling_metadata(gen, golden_set, corpus_metadata)

        baseline_list = []
        for raw_bname in args.baselines.split(","):
            raw_bname = raw_bname.strip()
            if not raw_bname:
                continue
            try:
                baseline_list.append(_load_baseline_spec(raw_bname, args, bm_adapter=bm_adapter))
            except Exception as exc:
                logger.error("加载竞品 '%s' 失败: %s", raw_bname, exc)
                return 1

        if baseline_list:
            suite = BaselineEvaluationSuite(
                bm_adapter=bm_adapter,
                baselines=baseline_list,
                output_dir=baseline_dir,
                guard_config=baseline_guard_config,
                corpus_metadata=corpus_metadata,
            )
            baseline_results = await suite.run(golden_set)
            for bm, results in baseline_results.items():
                # 找 citation_name
                citation = next(
                    (b.citation_name for b in baseline_list if b.name == bm),
                    bm,
                )
                explicit_baseline_report_rows.append((citation, results))
                gen.add_system_results(system_name=citation, results=results, question_type_key=question_type_key)
                logger.info("竞品 '%s': %d 条结果", citation, len(results))

    # ── 导入预计算 JSONL 竞品（重新 Judge）────────────────────────────
    if args.import_baseline and not args.table_only:
        from evaluation.suite import BaselineEvaluationSuite
        from baselines import ManualImportAdapter

        if golden_set is None:
            _, golden_set = _get_or_create_golden_set(args, bm_adapter, sampling_protocol)
            _attach_sampling_metadata(gen, golden_set, corpus_metadata)

        import_adapters = []
        import_setup_paths = _parse_named_paths(args.import_baseline_setup or [])
        for spec in args.import_baseline:
            if "=" not in spec:
                logger.warning("无效 --import-baseline 格式: '%s'，期望 NAME=PATH", spec)
                continue
            name, _, path = spec.partition("=")
            path = str(Path(path.strip()).resolve())
            if not Path(path).exists():
                logger.warning("预测文件不存在: %s", path)
                continue
            display_name = name.strip()
            adapter = ManualImportAdapter(
                name=display_name.lower().replace(" ", "_"),
                citation_name=display_name,
                predictions_path=path,
                setup_metrics_path=import_setup_paths.get(display_name, ""),
            )
            logger.info("导入竞品 '%s': %d 条预测", adapter.citation_name, adapter.loaded_count)
            import_adapters.append(adapter)

        if import_adapters:
            suite = BaselineEvaluationSuite(
                bm_adapter=bm_adapter,
                baselines=import_adapters,
                output_dir=baseline_dir,
                guard_config=baseline_guard_config,
                corpus_metadata=corpus_metadata,
            )
            for res_map in [await suite.run(golden_set)]:
                for bm, results in res_map.items():
                    citation = next(
                        (a.citation_name for a in import_adapters if a.name == bm), bm
                    )
                    gen.add_system_results(system_name=citation, results=results, question_type_key=question_type_key)

    # ── 直接导入已发表数字（无需 Judge）──────────────────────────────
    for spec in (args.import_published or []):
        try:
            parsed = _parse_published(spec)
            gen.add_published_metrics(
                system_name=parsed.pop("name"),
                accuracy=parsed.get("acc", 0),
                coverage=parsed.get("cov", 0),
                avg_latency=parsed.get("lat", 0),
                avg_tokens=parsed.get("tok", 0),
            )
            logger.info("已发表数字: '%s'", spec.split(":")[0])
        except Exception as exc:
            logger.warning("解析 --import-published 失败: '%s' (%s)", spec, exc)

    # ── 生成表格 ─────────────────────────────────────────────────────
    paths = gen.generate(
        output_dir=out_dir,
        caption=args.caption,
    )
    print("\n✅ 论文表格已生成：")
    for fmt, path in paths.items():
        print(f"   {fmt.upper():8}: {path}")
    if args.baselines and not args.table_only:
        _print_explicit_baseline_report(explicit_baseline_report_rows)

    if args.generate_report:
        from evaluation.report_generator import ReportGenerator

        report_out = args.report_output_dir or str(Path(out_dir) / "report")
        report_paths = ReportGenerator().generate(
            run_dir=args.run_artifact_dir or None,
            table_json=paths.get("json"),
            output_dir=report_out,
            title=f"{args.benchmark.upper()} ResearchOps Report",
        )
        print("\n✅ 学术报告已生成：")
        for name, path in report_paths.items():
            print(f"   {name.upper():12}: {path}")
    print()

    return 0


def _print_explicit_baseline_report(rows: list[tuple[str, list]]) -> None:
    """Print final report for baselines explicitly requested by --baselines."""
    print("\n== Baseline Final Report ==")
    if not rows:
        print("  (no baseline results produced)")
        return
    columns = [
        ("Baseline", 28, "<"),
        ("N", 5, ">"),
        ("Acc", 5, ">"),
        ("EM", 5, ">"),
        ("F1", 5, ">"),
        ("Cov", 5, ">"),
        ("Evd", 5, ">"),
        ("Avg", 6, ">"),
        ("P95", 6, ">"),
        ("Tok/Q", 7, ">"),
        ("Fail", 4, ">"),
    ]

    def _cell(value, width: int, align: str) -> str:
        text = str(value)
        if len(text) > width:
            text = text[: max(width - 1, 0)] + "~"
        return f"{text:<{width}}" if align == "<" else f"{text:>{width}}"

    def _row(values) -> str:
        return " | ".join(
            _cell(value, width, align)
            for value, (_, width, align) in zip(values, columns)
        )

    header = _row(label for label, _, _ in columns) + " | Notes"
    separator = "-+-".join("-" * width for _, width, _ in columns) + "-+------"
    print(header)
    print(separator)
    for name, results in rows:
        metrics = _baseline_report_metrics(results)
        print(
            _row([
                name,
                metrics["n"],
                f"{metrics['accuracy']:.1f}",
                f"{metrics['em']:.1f}",
                f"{metrics['f1']:.1f}",
                f"{metrics['coverage']:.1f}",
                f"{metrics['evidence_recall']:.1f}",
                f"{metrics['avg_latency']:.1f}s",
                f"{metrics['p95_latency']:.1f}s",
                f"{metrics['avg_tokens']:.1f}",
                metrics["failures"],
            ])
            + f" | {metrics['notes']}"
        )
    print()


def _baseline_report_metrics(results: list) -> dict:
    n = len(results)
    if not n:
        return {
            "n": 0,
            "accuracy": 0.0,
            "em": 0.0,
            "f1": 0.0,
            "coverage": 0.0,
            "evidence_recall": 0.0,
            "avg_latency": 0.0,
            "p95_latency": 0.0,
            "avg_tokens": 0.0,
            "failures": 0,
            "notes": "empty",
        }
    latencies = [_safe_float(getattr(row, "elapsed", 0.0)) for row in results]
    tokens = [_baseline_tokens(row) for row in results]
    failures = [row for row in results if getattr(row, "error", None) or getattr(row, "failure_reason", "")]
    failure_types = sorted({str(getattr(row, "failure_reason", "") or "error") for row in failures})
    official_em_values = [_metric_value_or_none(row, "official_em", "em") for row in results]
    official_f1_values = [_metric_value_or_none(row, "official_f1", "f1") for row in results]
    official_em = [value for value in official_em_values if value is not None]
    official_f1 = [value for value in official_f1_values if value is not None]
    if not official_em:
        official_em = [1.0 if bool(getattr(row, "judge_correct", False)) else 0.0 for row in results]
    if not official_f1:
        official_f1 = [1.0 if bool(getattr(row, "judge_correct", False)) else 0.0 for row in results]
    notes = ",".join(failure_types[:3]) if failure_types else ""
    imported = any(_metadata(row).get("imported_baseline") for row in results)
    if imported:
        missing = sum(1 for row in results if _failure_reason(row) == "import_missing")
        notes = (notes + "; " if notes else "") + f"import_missing={missing}"
    return {
        "n": n,
        "accuracy": sum(1 for row in results if bool(getattr(row, "judge_correct", False))) / n * 100,
        "em": sum(official_em) / n * 100,
        "f1": sum(official_f1) / n * 100,
        "coverage": sum(1 for row in results if bool(getattr(row, "coverage", False))) / n * 100,
        "evidence_recall": sum(_safe_float(getattr(row, "evidence_recall", 0.0)) for row in results) / n * 100,
        "avg_latency": sum(latencies) / n,
        "p95_latency": _percentile(latencies, 0.95),
        "avg_tokens": sum(tokens) / n,
        "failures": len(failures),
        "notes": notes,
    }


def _baseline_tokens(row) -> float:
    telemetry = getattr(row, "telemetry", {}) or {}
    if isinstance(telemetry, dict):
        total = telemetry.get("total_tokens")
        judge = telemetry.get("judge_tokens", 0)
        if total is not None:
            return _safe_float(total) + _safe_float(judge)
    return _safe_float(getattr(row, "tokens_used", 0)) + _safe_float(getattr(row, "judge_tokens", 0))


def _metric_value_or_none(row, *keys: str) -> float | None:
    for source in (_telemetry(row), _metadata(row)):
        for key in keys:
            if key in source:
                return _safe_float(source.get(key))
    return None


def _failure_reason(row) -> str:
    return str(getattr(row, "failure_reason", "") or _metadata(row).get("failure_reason", "") or "")


def _telemetry(row) -> dict:
    value = getattr(row, "telemetry", {}) or {}
    return value if isinstance(value, dict) else {}


def _metadata(row) -> dict:
    value = getattr(row, "metadata", {}) or {}
    return value if isinstance(value, dict) else {}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[index]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _baseline_guard_config(args: argparse.Namespace) -> dict:
    return {
        "sample_timeout_seconds": args.baseline_sample_timeout,
        "max_runtime_seconds": args.baseline_max_runtime,
        "max_total_tokens": args.baseline_max_total_tokens,
        "max_api_cost_usd": args.baseline_max_api_cost_usd,
        "max_disk_usage_bytes": args.baseline_max_disk_bytes,
        "min_free_disk_bytes": args.baseline_min_free_disk_bytes,
    }


def _parse_named_paths(specs: list[str]) -> dict[str, str]:
    out = {}
    for spec in specs or []:
        if "=" not in spec:
            continue
        name, _, path = spec.partition("=")
        out[name.strip()] = str(Path(path.strip()).resolve())
    return out


def _load_baseline_spec(raw_name: str, args: argparse.Namespace, bm_adapter=None):
    from baselines import (
        BM25RAGBaseline,
        GraphRAGBaseline,
        HybridRAGBaseline,
        LightRAGV136Baseline,
        LightRAGV1Baseline,
        LocalBM25Baseline,
        NaiveRAGBaseline,
        ReActSearchBaseline,
    )
    from baselines.base_adapter import BaselineAdapter

    lower = raw_name.strip().lower()
    if lower == "bm25":
        return LocalBM25Baseline(max_files=args.bm25_max_files)
    if lower == "bm25_rag":
        return BM25RAGBaseline(max_files=args.bm25_max_files)
    if lower == "hybrid_rag":
        return HybridRAGBaseline(
            max_files=args.hybrid_max_files,
            bm25_top_k=args.hybrid_bm25_top_k,
            dense_top_k=args.hybrid_dense_top_k,
            final_top_k=args.hybrid_final_top_k,
            dense_backend=args.hybrid_dense_backend,
            dense_dim=args.hybrid_dense_dim,
        )
    if lower == "react":
        return ReActSearchBaseline()
    if lower == "naive_rag":
        return NaiveRAGBaseline(max_files=args.naive_rag_max_files)
    if lower == "lightrag_v1":
        return LightRAGV1Baseline(
            predictions_path=args.lightrag_predictions,
            setup_metrics_path=args.lightrag_setup_metrics,
        )
    lightrag_modes = ("naive", "local", "global", "hybrid", "mix")
    if lower == "lightrag_v136" or lower in {f"lightrag_v136_{mode}" for mode in lightrag_modes}:
        mode = args.lightrag_query_mode
        for candidate in lightrag_modes:
            if lower == f"lightrag_v136_{candidate}":
                mode = candidate
                break
        return LightRAGV136Baseline(
            query_mode=mode,
            working_dir=args.lightrag_working_dir,
            max_files=args.lightrag_max_files,
            max_file_chars=args.lightrag_max_file_chars,
        )
    if lower == "graphrag":
        return GraphRAGBaseline(
            predictions_path=args.graphrag_predictions,
            setup_metrics_path=args.graphrag_setup_metrics,
        )
    if lower in ("lens_full", "lens_no_prior", "lens_no_seq", "lens_no_reuse"):
        if bm_adapter is None:
            raise ValueError("LENS ablation baseline requires bm_adapter")
        from ablations import build_single_lens_ablation
        run_cfg = {}
        try:
            run_cfg = bm_adapter.get_run_config()
        except Exception:
            pass
        return build_single_lens_ablation(
            bm_adapter=bm_adapter,
            profile_name=lower,
            max_token_budget=int(run_cfg.get("max_token_budget", 128000) or 128000),
            top_k_files=int(run_cfg.get("top_k_files", 5) or 5),
        )
    if ":" in raw_name:
        module_name, _, factory_name = raw_name.partition(":")
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        baseline = factory()
        if not isinstance(baseline, BaselineAdapter):
            raise TypeError(f"Factory {raw_name} did not return BaselineAdapter")
        return baseline
    raise ValueError(
        "Unknown baseline. Use bm25 and naive_rag for quickstart/local smoke; "
        "bm25_rag, hybrid_rag, and react for paper main; "
        "related-work/lifecycle: lightrag_v136 or lightrag_v136_<mode>; "
        "imported: lightrag_v1, graphrag; "
        "ablation: lens_full, lens_no_prior, lens_no_seq, lens_no_reuse; custom: module:factory. "
        "Long-context is intentionally excluded from the current implementation scope."
    )


def _build_sampling_protocol(args: argparse.Namespace, bm_adapter) -> object:
    if args.sampling_protocol:
        return load_sampling_protocol(args.sampling_protocol)
    if args.sample_ids_file:
        args.sampling_method = "fixed_ids"
    population_size = _population_size_of(bm_adapter)
    expected_population_size = args.expected_population_size
    if not expected_population_size and args.benchmark == "hotpotqa":
        expected_population_size = DEFAULT_HOTPOTQA_POPULATION_SIZE
    method = args.sampling_method
    target_n = 0 if method in {"full", "diagnostic_rare", "fixed_ids"} else args.golden_n
    strata = args.strata if method == "stratified" else ""
    split = "validation"
    try:
        split = str(bm_adapter.get_run_config().get("split") or "validation")
    except Exception:
        pass
    return create_sampling_protocol(
        benchmark=args.benchmark,
        split=split,
        population_size=population_size,
        method=method,
        seed=args.golden_seed,
        target_n=target_n,
        strata=strata,
        allocation=args.sampling_allocation,
        min_per_stratum=args.min_per_stratum,
        expected_population_size=expected_population_size,
        sample_ids_file=str(Path(args.sample_ids_file).expanduser().resolve()) if args.sample_ids_file else "",
    )


def _population_size_of(bm_adapter) -> int:
    try:
        if hasattr(bm_adapter, "describe_split"):
            return int(bm_adapter.describe_split().get("population_size", 0) or 0)
    except Exception:
        pass
    try:
        return len(bm_adapter.load_samples(limit=0, seed=42))
    except Exception:
        return 0


def _get_or_create_golden_set(args: argparse.Namespace, bm_adapter, sampling_protocol):
    from evaluation.golden_set import GoldenSetManager

    manager = GoldenSetManager(str(_SCRIPT_DIR / args.benchmark))
    target_n = int(getattr(sampling_protocol, "target_n", args.golden_n) or 0)
    if getattr(sampling_protocol, "method", "") in {"full", "diagnostic_rare", "fixed_ids"}:
        target_n = 0
    golden_set = manager.get_or_create(
        adapter=bm_adapter,
        seed=int(getattr(sampling_protocol, "seed", args.golden_seed)),
        n=target_n,
        sampling_protocol=sampling_protocol,
    )
    validation = validate_sampling_manifest(golden_set.sampling_manifest)
    if not validation["passed"]:
        raise ValueError(f"Invalid GoldenSet sampling manifest: {validation['errors']}")
    return manager, golden_set


def _attach_sampling_metadata(gen, golden_set, corpus_metadata: dict | None = None) -> None:
    if golden_set is None or not hasattr(gen, "set_sampling_metadata"):
        return
    metadata = _golden_summary(golden_set)
    if corpus_metadata:
        metadata.update(corpus_metadata)
    gen.set_sampling_metadata(metadata)


def _evaluation_corpus_metadata(args: argparse.Namespace, bm_adapter) -> dict:
    run_config = {}
    try:
        run_config = bm_adapter.get_run_config()
    except Exception:
        run_config = {}
    provenance = (args.context_corpus_provenance or run_config.get("context_corpus_mode") or "").strip().lower()
    if provenance in {"context", "sample_context"}:
        provenance = "sample"
    if not provenance:
        provenance = "unknown"
    risk = (args.context_corpus_risk or "").strip()
    if not risk:
        if provenance == "sample":
            risk = "oracle_sample_context,evaluation_set_context_index"
        elif provenance == "hybrid":
            risk = "sample_context_plus_raw_wiki"
        elif provenance in {"wiki", "raw_wiki"}:
            risk = "raw_wiki"
    return {
        "corpus_provenance": provenance,
        "corpus_risk": risk,
        "context_corpus_mode": str(run_config.get("context_corpus_mode") or provenance),
        "require_context_answerable": bool(run_config.get("require_context_answerable", False)),
    }


def _write_sampling_report(args: argparse.Namespace, golden_set) -> None:
    if not args.sampling_report_dir or golden_set is None:
        return
    out_dir = Path(args.sampling_report_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    method = str(golden_set.sampling_protocol.get("method", "sampling"))
    seed = int(golden_set.sampling_protocol.get("seed", golden_set.seed) or golden_set.seed)
    target_n = golden_set.sampling_protocol.get("target_n", golden_set.n_questions)
    stem = f"sampling_{method}_{seed}_{target_n or 'full'}"
    (out_dir / f"{stem}_protocol.json").write_text(
        json.dumps(golden_set.sampling_protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / f"{stem}_manifest.json").write_text(
        json.dumps(golden_set.sampling_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_sample_ids(
        out_dir / f"{stem}_sample_ids.json",
        golden_set.sample_ids(),
        metadata={
            "benchmark": golden_set.benchmark,
            "sample_id_checksum": golden_set.sample_id_checksum(),
            "golden_set_checksum": golden_set.checksum,
        },
    )


def _golden_summary(golden_set) -> dict:
    manifest = golden_set.sampling_manifest or {}
    protocol = golden_set.sampling_protocol or manifest.get("protocol", {}) or {}
    method = str(protocol.get("method") or "")
    population_size = int(golden_set.population_size or manifest.get("population_size") or 0)
    n_questions = int(golden_set.n_questions or manifest.get("actual_n") or 0)
    if method == "full" and population_size and n_questions == population_size:
        evaluation_scope = "full_validation"
    elif method == "diagnostic_rare":
        evaluation_scope = "diagnostic_rare"
    elif method == "fixed_ids":
        evaluation_scope = "fixed_sample_ids"
    else:
        evaluation_scope = "sampled_evaluation"
    return {
        "benchmark": golden_set.benchmark,
        "n_questions": n_questions,
        "population_size": population_size,
        "evaluation_scope": evaluation_scope,
        "benchmark_label": _benchmark_label(golden_set.benchmark, evaluation_scope, n_questions),
        "sample_id_checksum": golden_set.sample_id_checksum(),
        "golden_set_checksum": golden_set.checksum,
        "sampling_protocol": protocol,
        "sampling_manifest": manifest,
        "sampling_validation": validate_sampling_manifest(manifest),
    }


def _benchmark_label(benchmark: str, evaluation_scope: str, n_questions: int) -> str:
    if benchmark == "hotpotqa":
        base = "HotpotQA fullwiki validation"
    else:
        base = benchmark
    if evaluation_scope == "full_validation":
        return base
    if evaluation_scope == "diagnostic_rare":
        return f"{base} diagnostic rare subset (n={n_questions})"
    if evaluation_scope == "fixed_sample_ids":
        return f"{base} fixed sample-ID subset (n={n_questions})"
    return f"{base} sampled subset (n={n_questions})"


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
