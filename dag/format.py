"""Text renderings of a DAG and of one scheduler run."""

from __future__ import annotations

from dag.graph import DAG
from dag.scheduler import RunReport, Status


def format_graph(dag: DAG) -> str:
    lines = [f"DAG {dag.name}", "edges:"]
    edges = dag.edges()
    if edges:
        for upstream_id, downstream_id in edges:
            lines.append(f"  {upstream_id} -> {downstream_id}")
    else:
        lines.append("  (none)")
    lines.append("topological levels:")
    levels = dag.topological_levels()
    if levels:
        for index, level in enumerate(levels):
            lines.append(f"  [{index}] {', '.join(level)}")
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def format_events(report: RunReport) -> str:
    if not report.events:
        return "(no events)"
    lines = []
    for event in report.events:
        detail = f"  {event.detail}" if event.detail else ""
        lines.append(f"{event.elapsed_ms:8.1f} ms  {event.kind:<7} {event.task_id}{detail}")
    return "\n".join(lines)


def format_timeline(report: RunReport, *, width: int = 36) -> str:
    """Draw each task against the run clock. ``#`` marks time spent inside the task."""
    if width < 1:
        raise ValueError("width must be >= 1")
    spans: dict[str, list[float]] = {}
    first_seen: dict[str, float] = {}
    for event in report.events:
        first_seen.setdefault(event.task_id, event.elapsed_ms)
        if event.kind in {"start", "success", "fail"}:
            spans.setdefault(event.task_id, []).append(event.elapsed_ms)
    end = max((event.elapsed_ms for event in report.events), default=0.0)
    ordered = sorted(
        report.status,
        key=lambda task_id: (first_seen.get(task_id, float("inf")), task_id),
    )
    lines = []
    for task_id in ordered:
        status = report.status[task_id]
        if status is Status.SKIPPED:
            lines.append(f"{task_id:<18} skipped")
            continue
        times = spans.get(task_id, [])
        if len(times) < 2:
            lines.append(f"{task_id:<18} {status.value}")
            continue
        start, stop = times[0], times[-1]
        bar = _bar(start, stop, end, width)
        lines.append(f"{task_id:<18} |{bar}| {start:7.1f}-{stop:7.1f} ms")
    return "\n".join(lines)


def format_results(report: RunReport) -> str:
    lines = []
    for task_id, status in report.status.items():
        attempts = report.attempts[task_id]
        detail = _result_detail(report, task_id, status)
        lines.append(f"{task_id:<18} {status.value:<8} attempts={attempts:<2} {detail}")
    return "\n".join(lines)


def _result_detail(report: RunReport, task_id: str, status: Status) -> str:
    if status is Status.SUCCESS:
        return _short(repr(report.values[task_id]))
    if status is Status.FAILED:
        error = report.errors[task_id]
        return _short(f"{type(error).__name__}: {error}")
    for event in report.events:
        if event.task_id == task_id and event.kind == "skip":
            return event.detail
    return ""


def _short(text: str, limit: int = 72) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _bar(start: float, stop: float, end: float, width: int) -> str:
    scale = end if end > 0 else 1.0
    left = min(width - 1, int(start / scale * width))
    right = min(width, int(stop / scale * width))
    if right <= left:
        right = left + 1
    return "." * left + "#" * (right - left) + "." * (width - right)
