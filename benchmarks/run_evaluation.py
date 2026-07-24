#!/usr/bin/env python3
"""benchmarks/run_evaluation.py — 竞品横向评估 CLI

此脚本与 run_research_loop.py 完全独立：
  - run_research_loop.py : Sirchmunk 自改进循环（迭代优化自身）
  - run_evaluation.py    : 横向对比（固定测试集，生成论文表格）

使用方式::

    # 1. 用 Mock 竞品运行端到端集成测试（验证 pipeline 通畅）
    python benchmarks/run_evaluation.py \\
      --benchmark hotpotqa \\
      --env benchmarks/hotpotqa/.env.hotpotqa \\
      --baselines mock \\
      --golden-n 20 \\
      --sirchmunk-results benchmarks/hotpotqa/output/results_YYYYMMDD.jsonl \\
      --output-dir benchmarks/hotpotqa/output/paper_table/

    # 2. 导入竞品预测 JSONL + 本文结果 → 生成完整比较表格
    python benchmarks/run_evaluation.py \\
      --benchmark hotpotqa \\
      --env benchmarks/hotpotqa/.env.hotpotqa \\
      --import-baseline "GPT-4o (zero-shot)=output/gpt4o_preds.jsonl" \\
      --import-baseline-setup "GPT-4o (zero-shot)=output/gpt4o_setup_metrics.json" \\
      --sirchmunk-results benchmarks/hotpotqa/output/results_YYYYMMDD.jsonl \\
      --output-dir benchmarks/hotpotqa/output/paper_table/

    # 3. 只填写已发表数字（不重新 Judge，直接写入表格）
    python benchmarks/run_evaluation.py \\
      --benchmark hotpotqa \\
      --env benchmarks/hotpotqa/.env.hotpotqa \\
      --import-published "Reported System:acc=45.0,cov=80.0,lat=5.2" \\
      --sirchmunk-results benchmarks/hotpotqa/output/results_YYYYMMDD.jsonl \\
      --output-dir benchmarks/hotpotqa/output/paper_table/

    # 4. 仅生成表格（不运行任何 baseline，纯汇聚已有结果）
    python benchmarks/run_evaluation.py \\
      --benchmark hotpotqa \\
      --env benchmarks/hotpotqa/.env.hotpotqa \\
      --sirchmunk-results benchmarks/hotpotqa/output/results_YYYYMMDD.jsonl \\
      --table-only \\
      --output-dir benchmarks/hotpotqa/output/paper_table/
"""
from __future__ import annotations

import argparse
import asyncio
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
                   help="运行的竞品列表（逗号分隔）: mock, gold_copy, random, bm25, naive_rag, lightrag_v1, graphrag")
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
                   help="GoldenSet 大小（默认 150，0=全量）")
    p.add_argument("--golden-seed", type=int, default=42, dest="golden_seed",
                   help="GoldenSet 随机种子（默认 42）")
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
    p.add_argument("--graphrag-predictions", default="", dest="graphrag_predictions",
                   help="GraphRAG 预计算预测 JSONL 路径")
    p.add_argument("--graphrag-setup-metrics", default="", dest="graphrag_setup_metrics",
                   help="GraphRAG setup metrics JSON 路径")
    p.add_argument("--bm25-max-files", type=int, default=20000, dest="bm25_max_files",
                   help="BM25本地索引最多读取文件数")
    p.add_argument("--naive-rag-max-files", type=int, default=5000, dest="naive_rag_max_files",
                   help="NaiveRAG本地索引最多读取文件数")
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

    # ── 初始化 PaperTableGenerator ──────────────────────────────────
    from evaluation.table_generator import PaperTableGenerator
    from framework.runner import UnifiedExperimentRunner

    gen = PaperTableGenerator(
        benchmark_name=args.benchmark.upper(),
        our_system_name=ours_name,
    )

    question_type_key = "question_type"
    try:
        question_type_key = bm_adapter.get_analysis_schema().get("primary_group_key", "question_type")
    except Exception:
        pass

    golden_set = None
    if args.sirchmunk_results or (args.baselines and not args.table_only) or (args.import_baseline and not args.table_only):
        from evaluation.golden_set import GoldenSetManager
        manager = GoldenSetManager(str(_SCRIPT_DIR / args.benchmark))
        golden_set = manager.get_or_create(
            adapter=bm_adapter,
            seed=args.golden_seed,
            n=args.golden_n,
        )
        logger.info(
            "GoldenSet: %d questions seed=%d checksum=%s",
            golden_set.n_questions,
            args.golden_seed,
            golden_set.sample_id_checksum(),
        )

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

    # ── 运行 Mock / SDK 竞品 ─────────────────────────────────────────
    if args.baselines and not args.table_only:
        from evaluation.suite import BaselineEvaluationSuite
        from baselines import (
            ConstantMockBaseline,
            FixedAccuracyMockBaseline,
            GoldCopyMockBaseline,
            GraphRAGBaseline,
            LightRAGV1Baseline,
            LocalBM25Baseline,
            NaiveRAGBaseline,
            RandomAnswerMockBaseline,
        )

        if golden_set is None:
            from evaluation.golden_set import GoldenSetManager
            manager = GoldenSetManager(str(_SCRIPT_DIR / args.benchmark))
            golden_set = manager.get_or_create(
                adapter=bm_adapter,
                seed=args.golden_seed,
                n=args.golden_n,
            )

        gold_map = golden_set.to_gold_map()
        baseline_list = []
        for bname in args.baselines.split(","):
            bname = bname.strip().lower()
            if bname == "mock" or bname == "constant":
                baseline_list.append(ConstantMockBaseline())
            elif bname == "random":
                baseline_list.append(RandomAnswerMockBaseline(seed=args.golden_seed))
            elif bname == "gold_copy":
                baseline_list.append(GoldCopyMockBaseline(gold_map=gold_map))
            elif bname in ("bm25", "bm25_local"):
                baseline_list.append(LocalBM25Baseline(max_files=args.bm25_max_files))
            elif bname in ("naive_rag", "naive_rag_local"):
                baseline_list.append(NaiveRAGBaseline(max_files=args.naive_rag_max_files))
            elif bname in ("lightrag", "lightrag_v1"):
                baseline_list.append(LightRAGV1Baseline(
                    predictions_path=args.lightrag_predictions,
                    setup_metrics_path=args.lightrag_setup_metrics,
                ))
            elif bname == "graphrag":
                baseline_list.append(GraphRAGBaseline(
                    predictions_path=args.graphrag_predictions,
                    setup_metrics_path=args.graphrag_setup_metrics,
                ))
            elif bname.startswith("fixed_acc_"):
                # e.g. fixed_acc_30 → 30%
                try:
                    pct = int(bname.split("_")[-1]) / 100
                except ValueError:
                    pct = 0.3
                baseline_list.append(FixedAccuracyMockBaseline(
                    gold_map=gold_map, target_accuracy=pct, seed=args.golden_seed
                ))
            else:
                logger.warning("未知竞品名称: '%s'，跳过", bname)

        if baseline_list:
            suite = BaselineEvaluationSuite(
                bm_adapter=bm_adapter,
                baselines=baseline_list,
                output_dir=baseline_dir,
                guard_config=baseline_guard_config,
            )
            baseline_results = await suite.run(golden_set)
            for bm, results in baseline_results.items():
                # 找 citation_name
                citation = next(
                    (b.citation_name for b in baseline_list if b.name == bm),
                    bm,
                )
                gen.add_system_results(system_name=citation, results=results, question_type_key=question_type_key)
                logger.info("竞品 '%s': %d 条结果", citation, len(results))

    # ── 导入预计算 JSONL 竞品（重新 Judge）────────────────────────────
    if args.import_baseline and not args.table_only:
        from evaluation.suite import BaselineEvaluationSuite
        from baselines import ManualImportAdapter

        if golden_set is None:
            from evaluation.golden_set import GoldenSetManager
            manager = GoldenSetManager(str(_SCRIPT_DIR / args.benchmark))
            golden_set = manager.get_or_create(
                adapter=bm_adapter, seed=args.golden_seed, n=args.golden_n
            )

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
    print(f"\n✅ 论文表格已生成：")
    for fmt, path in paths.items():
        print(f"   {fmt.upper():8}: {path}")

    if args.generate_report:
        from evaluation.report_generator import ReportGenerator

        report_out = args.report_output_dir or str(Path(out_dir) / "report")
        report_paths = ReportGenerator().generate(
            run_dir=args.run_artifact_dir or None,
            table_json=paths.get("json"),
            output_dir=report_out,
            title=f"{args.benchmark.upper()} ResearchOps Report",
        )
        print(f"\n✅ 学术报告已生成：")
        for name, path in report_paths.items():
            print(f"   {name.upper():12}: {path}")
    print()

    return 0


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


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
