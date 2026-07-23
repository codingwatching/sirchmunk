#!/usr/bin/env python3
"""ResearchOps P3 experiment queue CLI.

Examples::

    python benchmarks/run_queue.py add-matrix \
      --add-bm setup_cost=benchmarks/setup_cost/.env.setup_cost \
      --systems sirchmunk --seeds 42,43 --cache-modes cold,warm --limit 1

    python benchmarks/run_queue.py run --max-concurrent 2 --max-tasks 4

    python benchmarks/run_queue.py list
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SCRIPT_DIR), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from framework.experiment_queue import ExperimentQueue, QueueExecutor, QueueTaskStatus  # noqa: E402
from framework.experiment_registry import ExperimentRegistry  # noqa: E402
from framework.registry import supported_benchmarks  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResearchOps P3 experiment queue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--queue-path", default="benchmarks/experiment_queue.json", help="queue JSON path")
    parser.add_argument("--registry-path", default="benchmarks/experiment_registry.jsonl", help="registry JSONL path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("add-matrix", help="enqueue benchmark × system × seed × cache matrix")
    add.add_argument("--add-bm", action="append", required=True, metavar="NAME=ENV", help="benchmark/env pair; repeatable")
    add.add_argument("--systems", default="sirchmunk", help="comma-separated systems, default sirchmunk")
    add.add_argument("--seeds", default="42", help="comma-separated seeds")
    add.add_argument("--cache-modes", default="cold", help="comma-separated cache modes: cold,warm,compiled,none")
    add.add_argument("--stage", choices=["exploration", "frozen"], default="frozen", help="experiment stage")
    add.add_argument("--limit", type=int, default=0, help="sample limit per run")
    add.add_argument("--priority", type=int, default=100, help="lower value runs earlier")
    add.add_argument("--task-max-attempts", type=int, default=1, dest="task_max_attempts", help="queue-level attempts per task")
    add.add_argument("--replace", action="store_true", help="replace duplicate task ids")
    add.add_argument("--config-json", default="", help="extra config overrides as JSON object")
    add.add_argument(
        "--external-command",
        action="append",
        default=[],
        metavar="SYSTEM=JSON_ARRAY",
        help="command for an external system, e.g. lightrag='[\"python\",\"run.py\"]'",
    )
    add.add_argument("--allow-cache-clear", action="store_true", help="allow cold cache clearing inside work_path")
    add.add_argument("--cache-dry-run", action="store_true", help="record cache actions without clearing")

    list_cmd = sub.add_parser("list", help="list queued tasks")
    list_cmd.add_argument("--status", default="", choices=[""] + [s.value for s in QueueTaskStatus], help="optional status filter")

    sub.add_parser("status", help="show queue summary")

    cancel = sub.add_parser("cancel", help="cancel a queued task")
    cancel.add_argument("task_id")

    run = sub.add_parser("run", help="run pending tasks")
    run.add_argument("--max-concurrent", type=int, default=1, help="max concurrent queue tasks")
    run.add_argument("--max-tasks", type=int, default=0, help="max tasks to run this invocation, 0=all ready")
    run.add_argument("--no-resume", action="store_true", help="disable sample-level resume")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    queue = ExperimentQueue(Path(args.queue_path).resolve())
    registry = ExperimentRegistry(Path(args.registry_path).resolve())

    if args.cmd == "add-matrix":
        benchmarks = _parse_benchmarks(args.add_bm)
        overrides = _parse_config_json(args.config_json)
        if args.allow_cache_clear:
            overrides["cache_allow_clear"] = True
        if args.cache_dry_run:
            overrides["cache_dry_run"] = True
        tasks = queue.add_matrix(
            benchmarks=benchmarks,
            systems=_split_csv(args.systems),
            seeds=[int(seed) for seed in _split_csv(args.seeds)],
            cache_modes=_split_csv(args.cache_modes),
            stage=args.stage,
            limit=args.limit,
            priority=args.priority,
            max_attempts=args.task_max_attempts,
            config_overrides=overrides,
            commands_by_system=_parse_external_commands(args.external_command),
            replace=args.replace,
        )
        print(f"已入队 {len(tasks)} 个任务，queue={queue.path}")
        _print_tasks(tasks)
        return

    if args.cmd == "list":
        tasks = queue.list(status=args.status)
        _print_tasks(tasks)
        return

    if args.cmd == "status":
        print(json.dumps(queue.summary(), indent=2, ensure_ascii=False))
        return

    if args.cmd == "cancel":
        ok = queue.cancel(args.task_id)
        print("cancelled" if ok else "not cancelled")
        return

    if args.cmd == "run":
        executor = QueueExecutor(
            queue,
            registry=registry,
            max_concurrent=args.max_concurrent,
            resume=not args.no_resume,
        )
        results = asyncio.run(executor.run_pending(max_tasks=args.max_tasks))
        print(f"已执行 {len(results)} 个任务")
        _print_tasks(results)
        print(f"registry={registry.path}")
        return


def _parse_benchmarks(specs: List[str]) -> Dict[str, str]:
    benchmarks: Dict[str, str] = {}
    supported = set(supported_benchmarks())
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"无效 --add-bm: {spec}; 期望 NAME=ENV")
        name, _, env = spec.partition("=")
        name = name.strip()
        if name not in supported:
            raise SystemExit(f"未知 benchmark: {name}; supported={', '.join(sorted(supported))}")
        env_path = str(Path(env.strip()).resolve())
        if not Path(env_path).exists():
            raise SystemExit(f"env 文件不存在: {env_path}")
        benchmarks[name] = env_path
    return benchmarks


def _parse_config_json(text: str) -> Dict[str, object]:
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--config-json 不是合法 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("--config-json 必须是 JSON object")
    return data


def _parse_external_commands(specs: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"无效 --external-command: {spec}; 期望 SYSTEM=JSON_ARRAY")
        system, _, command_json = spec.partition("=")
        try:
            command = json.loads(command_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--external-command 命令不是合法 JSON array: {exc}") from exc
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise SystemExit("--external-command 的命令必须是字符串数组")
        out[system.strip()] = command
    return out


def _split_csv(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _print_tasks(tasks) -> None:
    if not tasks:
        print("无任务")
        return
    headers = ["status", "priority", "task_id", "benchmark", "system", "seed", "cache", "stage", "attempts", "error"]
    print("\t".join(headers))
    for task in tasks:
        print("\t".join([
            task.status.value,
            str(task.priority),
            task.task_id,
            task.benchmark,
            task.system,
            str(task.seed),
            task.cache_mode,
            task.stage,
            str(task.attempts),
            (task.error or "")[:80].replace("\n", " "),
        ]))


if __name__ == "__main__":
    main()
