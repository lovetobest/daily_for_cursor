"""Run the sample sales pipeline.

Usage:
    python3 -m dag
    python3 -m dag --fail-check
    python3 -m dag --workers 1 --delay 0
"""

from __future__ import annotations

import argparse

from dag.demo import build_pipeline
from dag.format import format_events, format_graph, format_results, format_timeline
from dag.scheduler import Scheduler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the sample sales-pipeline DAG.")
    parser.add_argument("--workers", type=int, default=4, help="maximum concurrent tasks")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="base seconds of simulated work; 0 runs the tasks instantly",
    )
    parser.add_argument(
        "--fail-check",
        action="store_true",
        help="fail quality_check so publish is skipped and the report branch still runs",
    )
    args = parser.parse_args(argv)

    dag = build_pipeline(fail_check=args.fail_check, delay=args.delay)
    print(format_graph(dag))
    print()
    print(
        f"schedule  workers={args.workers}  "
        "a task starts when its own parents have succeeded"
    )
    report = Scheduler(max_workers=args.workers).run(dag)
    print(format_events(report))
    print()
    print(format_timeline(report))
    print()
    print(format_results(report))
    print()
    print("run failed" if report.failed else "run succeeded")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
