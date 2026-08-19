#!/usr/bin/env python3
"""benchmarks/run_research_loop.py — Research Loop CLI entry point

Usage examples::

    # Single-benchmark loop (50 sampled questions, at most 5 iterations)
    python benchmarks/run_research_loop.py \\
      --benchmark hotpotqa \\
      --env benchmarks/hotpotqa/.env.hotpotqa.exploration \\
      --limit 50 --max-iter 5

    # Multi-benchmark joint optimization (Pareto gate / exploration only)
    python benchmarks/run_research_loop.py \\
      --multi \\
      --add-bm hotpotqa=benchmarks/hotpotqa/.env.hotpotqa.exploration \\
      --add-bm setup_cost=benchmarks/setup_cost/.env.setup_cost \\
      --limit 30 --shadow-fraction 0.10

    # P0 mechanism experiment
    python benchmarks/run_research_loop.py \\
      --benchmark setup_cost \\
      --env benchmarks/setup_cost/.env.setup_cost \\
      --limit 1 --max-iter 1

    # Dry-run mode (no .env file is written)
    python benchmarks/run_research_loop.py \\
      --benchmark hotpotqa \\
      --env benchmarks/hotpotqa/.env.hotpotqa.exploration \\
      --dry-run --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# ── sys.path injection so framework and benchmarks resolve from any CWD ──
_SCRIPT_DIR = Path(__file__).parent.resolve()   # benchmarks/
_PROJECT_ROOT = _SCRIPT_DIR.parent              # sirchmunk/
_SRC = _PROJECT_ROOT / "src"

for _p in (str(_SCRIPT_DIR), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ────────────────────────────────────────────────────────────────────

from framework.orchestrator import ResearchOrchestrator  # noqa: E402
from framework.registry import load_benchmark_adapter, supported_benchmarks  # noqa: E402


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

def _load_adapter(benchmark: str, env_file: str):
    """Load adapter through the shared benchmark registry."""
    return load_benchmark_adapter(benchmark, env_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research Loop — 自动化 benchmark 研究流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Single- or multi-benchmark selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--benchmark", "-b",
        choices=supported_benchmarks(),
        help="单 benchmark 模式：指定 benchmark 名称",
    )
    mode_group.add_argument(
        "--multi",
        action="store_true",
        help="多 benchmark 联合优化模式（配合 --add-bm 使用）",
    )
    # Single-benchmark only
    parser.add_argument(
        "--env", "-e",
        default=None,
        help=".env 配置文件路径（--benchmark 模式专用）",
    )
    # Multi-benchmark only
    parser.add_argument(
        "--add-bm",
        action="append",
        dest="add_bm",
        metavar="NAME=ENV_FILE",
        help=(
            "添加一个 benchmark（可多次指定）。"
            "格式: hotpotqa=benchmarks/hotpotqa/.env.hotpotqa"
        ),
    )
    # Shared arguments
    parser.add_argument(
        "--limit", "-l",
        type=int, default=None,
        help="每次实验每个 benchmark 的最大样本数（0=全量；省略时使用 profile env 中的 HOTPOT_LIMIT，若无则 0）",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="随机种子（默认 42）",
    )
    parser.add_argument(
        "--max-iter", "-n",
        type=int, default=5,
        dest="max_iter",
        help="最大迭代轮数（默认 5）",
    )
    parser.add_argument(
        "--shadow-fraction",
        type=float, default=0.10,
        dest="shadow_fraction",
        help="Shadow eval 采样比例（默认 0.10），仅对 --multi 有效",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float, default=1.0,
        dest="convergence_threshold",
        help="收敛判断阈值（accuracy delta 百分点，默认 1.0）",
    )
    parser.add_argument(
        "--convergence-window",
        type=int, default=3,
        dest="convergence_window",
        help="收敛判断连续轮数（默认 3）",
    )
    parser.add_argument(
        "--experiments-path",
        default=None,
        dest="experiments_path",
        help="实验记录 JSONL 路径（单模式默认 benchmarks/experiments.jsonl，多模式默认 benchmarks/multi_experiments.jsonl）",
    )
    parser.add_argument(
        "--skip-run",
        default=None,
        dest="skip_run",
        metavar="JSONL_PATH",
        help="跳过实验运行，直接从已有 JSONL 结果做分析（内单模式）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="演练模式：分析改进建议但不实际修改 .env 文件",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="日志级别（默认 INFO）",
    )
    return parser.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence overly noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _resolve_limit(cli_limit, adapter, default: int) -> int:
    if cli_limit is not None:
        return max(int(cli_limit), 0)
    getter = getattr(adapter, "get_profile_limit", None)
    if callable(getter):
        return max(int(getter(default)), 0)
    return default


async def _main() -> int:
    args = _parse_args()
    _setup_logging(args.log_level)

    logger = logging.getLogger("run_research_loop")

    # ── Multi-benchmark mode ────────────────────────────────────
    if args.multi:
        if not args.add_bm:
            logger.error("多模式需要指定至少两个 --add-bm，如: --add-bm hotpotqa=...")
            return 1

        adapters = []
        for spec in args.add_bm:
            if "=" not in spec:
                logger.error("无效 --add-bm 格式: '%s'，期望 NAME=ENV_FILE", spec)
                return 1
            bm_name, _, env_path = spec.partition("=")
            bm_name = bm_name.strip()
            env_path = str(Path(env_path.strip()).resolve())
            if not Path(env_path).exists():
                logger.error("env 文件不存在: %s", env_path)
                return 1
            try:
                adapter = _load_adapter(bm_name, env_path)
                adapters.append(adapter)
                logger.info("注册 adapter: %s  env=%s", bm_name, env_path)
            except Exception as exc:
                logger.error("加载 adapter '%s' 失败: %s", bm_name, exc)
                return 1

        if len(adapters) < 2:
            logger.warning("建议注册至少 2 个 benchmark 以发挥联合优化价值")

        experiments_path = str(Path(
            args.experiments_path or "benchmarks/multi_experiments.jsonl"
        ).resolve())

        from framework.multi_orchestrator import MultiAdapterOrchestrator
        orch = MultiAdapterOrchestrator(
            adapters=adapters,
            experiments_path=experiments_path,
            dry_run=args.dry_run,
        )
        try:
            await orch.run(
                max_iterations=args.max_iter,
                limit_per_bm=args.limit if args.limit is not None else 0,
                seed=args.seed,
                shadow_fraction=args.shadow_fraction,
                convergence_threshold=args.convergence_threshold,
                convergence_window=args.convergence_window,
            )
        except KeyboardInterrupt:
            print("\n\n  [Interrupted] 研究循环已中断。")
        except Exception as exc:
            logger.exception("多 benchmark 循环异常: %s", exc)
            return 1
        return 0

    # ── Single-benchmark mode (existing logic) ─────
    if not args.env:
        logger.error("单模式需要指定 --env 参数")
        return 1

    logger.info("Research Loop starting — benchmark=%s", args.benchmark)

    env_file = str(Path(args.env).resolve())
    if not Path(env_file).exists():
        logger.error("env 文件不存在: %s", env_file)
        return 1

    try:
        adapter = _load_adapter(args.benchmark, env_file)
    except Exception as exc:
        logger.error("加载 adapter 失败: %s", exc)
        return 1

    effective_limit = _resolve_limit(args.limit, adapter, default=0)

    experiments_path = str(Path(
        args.experiments_path or "benchmarks/experiments.jsonl"
    ).resolve())

    orchestrator = ResearchOrchestrator(
        adapter=adapter,
        experiments_path=experiments_path,
        dry_run=args.dry_run,
    )
    try:
        await orchestrator.run(
            max_iterations=args.max_iter,
            limit=effective_limit,
            seed=args.seed,
            convergence_threshold=args.convergence_threshold,
            convergence_window=args.convergence_window,
            skip_run_path=args.skip_run,
        )
    except KeyboardInterrupt:
        print("\n\n  [Interrupted] 研究循环已中断，已记录的实验不受影响。")
    except Exception as exc:
        logger.exception("研究循环异常退出: %s", exc)
        return 1

    return 0


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
