"""Run a :class:`dag.graph.DAG` on a thread pool.

The runner keeps a ready set: a pending task whose every parent has succeeded.
Ready tasks are submitted until ``max_workers`` slots are full. When a task
finishes, newly unblocked tasks are submitted immediately, including tasks
that do not depend on other still-running work.

Failure policy: an exception that survives the task's retries marks that task
failed. Every descendant is skipped. Tasks that do not depend on the failure
still run.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import Condition
from typing import Any

from dag.graph import DAG


class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Event:
    """One scheduler observation, ordered by ``seq`` under the runner lock."""

    seq: int
    elapsed_ms: float
    task_id: str
    kind: str
    detail: str = ""


@dataclass
class RunReport:
    """Outcome of one :meth:`Scheduler.run` call.

    ``levels`` is the bulk-synchronous wave list from Kahn's algorithm.
    ``events`` is the dynamic schedule that actually ran.
    """

    status: dict[str, Status]
    values: dict[str, Any]
    errors: dict[str, BaseException]
    attempts: dict[str, int]
    events: tuple[Event, ...]
    levels: list[list[str]]

    @property
    def succeeded(self) -> bool:
        """True when every task completed successfully. An empty DAG succeeds."""
        return all(item is Status.SUCCESS for item in self.status.values())

    @property
    def failed(self) -> bool:
        return any(item is Status.FAILED for item in self.status.values())

    def __repr__(self) -> str:
        parts = ", ".join(f"{task_id}={status.value}" for task_id, status in self.status.items())
        return f"RunReport({parts})"


class Scheduler:
    """In-process dynamic scheduler.

    ``max_workers`` caps how many task functions run at once. The mapping passed
    to a task is a shallow copy of its parents' return values; nested objects
    inside those values are shared across tasks.
    """

    def __init__(self, max_workers: int = 4) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = max_workers

    def run(self, dag: DAG) -> RunReport:
        levels = dag.topological_levels()
        execution = _Execution(self.max_workers, dag, levels)
        return execution.run()


class _Execution:
    """Mutable bookkeeping for a single :meth:`Scheduler.run` call."""

    def __init__(self, max_workers: int, dag: DAG, levels: list[list[str]]) -> None:
        self.max_workers = max_workers
        self.levels = levels
        self.tasks = {task_id: dag[task_id] for task_id in dag}
        self.status = {task_id: Status.PENDING for task_id in self.tasks}
        self.values: dict[str, Any] = {}
        self.errors: dict[str, BaseException] = {}
        self.attempts = {task_id: 0 for task_id in self.tasks}
        self.events: list[Event] = []
        self.remaining = set(self.tasks)
        self.running: set[str] = set()
        self.inflight = 0
        self.seq = 0
        self.started = time.perf_counter()
        self.cond = Condition()
        self.executor: ThreadPoolExecutor | None = None

    def run(self) -> RunReport:
        executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="dag")
        self.executor = executor
        try:
            with self.cond:
                self._pump()
                while self.inflight:
                    self.cond.wait()
        finally:
            executor.shutdown(wait=True)

        if self.remaining:
            unfinished = ", ".join(sorted(self.remaining))
            raise RuntimeError(f"scheduler stopped with unfinished tasks: {unfinished}")

        ordered = [task_id for level in self.levels for task_id in level]
        return RunReport(
            status={task_id: self.status[task_id] for task_id in ordered},
            values={task_id: self.values[task_id] for task_id in ordered if task_id in self.values},
            errors={task_id: self.errors[task_id] for task_id in ordered if task_id in self.errors},
            attempts={task_id: self.attempts[task_id] for task_id in ordered},
            events=tuple(self.events),
            levels=self.levels,
        )

    def _emit(self, task_id: str, kind: str, detail: str = "") -> None:
        self.seq += 1
        self.events.append(
            Event(
                seq=self.seq,
                elapsed_ms=(time.perf_counter() - self.started) * 1000.0,
                task_id=task_id,
                kind=kind,
                detail=detail,
            )
        )

    def _ready_ids(self) -> list[str]:
        ready: list[str] = []
        for task_id in self.remaining:
            if task_id in self.running:
                continue
            parents = self.tasks[task_id].upstream
            if all(self.status[parent] is Status.SUCCESS for parent in parents):
                ready.append(task_id)
        ready.sort()
        return ready

    def _skip_blocked(self) -> None:
        """Skip tasks that can never run because a parent failed or was skipped."""
        changed = True
        while changed:
            changed = False
            for task_id in list(self.remaining):
                if task_id in self.running:
                    continue
                blocked_by = next(
                    (
                        parent
                        for parent in self.tasks[task_id].upstream
                        if self.status[parent] in (Status.FAILED, Status.SKIPPED)
                    ),
                    None,
                )
                if blocked_by is None:
                    continue
                self.status[task_id] = Status.SKIPPED
                self.remaining.remove(task_id)
                self._emit(
                    task_id,
                    "skip",
                    f"upstream {blocked_by} is {self.status[blocked_by].value}",
                )
                changed = True

    def _pump(self) -> None:
        """Caller holds ``cond``. Fill free worker slots with ready tasks."""
        assert self.executor is not None
        self._skip_blocked()
        for task_id in self._ready_ids():
            if self.inflight >= self.max_workers:
                break
            snapshot = {parent: self.values[parent] for parent in self.tasks[task_id].upstream}
            self.running.add(task_id)
            self.inflight += 1
            try:
                self.executor.submit(self._worker, task_id, snapshot)
            except Exception:
                self.running.discard(task_id)
                self.inflight -= 1
                raise

    def _worker(self, task_id: str, upstream: dict[str, Any]) -> None:
        task = self.tasks[task_id]
        ok = False
        value: Any = None
        error: BaseException | None = None
        attempt = 0
        try:
            with self.cond:
                self.status[task_id] = Status.RUNNING
                self._emit(task_id, "start")
            for attempt in range(1, task.retries + 2):
                try:
                    value = task.fn(upstream)
                except Exception as exc:
                    error = exc
                    if attempt <= task.retries:
                        with self.cond:
                            self.attempts[task_id] = attempt
                            self._emit(task_id, "retry", f"{type(exc).__name__}: {exc}")
                        continue
                    break
                else:
                    ok = True
                    error = None
                    break
        finally:
            with self.cond:
                self.attempts[task_id] = attempt
                self.running.discard(task_id)
                self.remaining.discard(task_id)
                self.inflight -= 1
                if ok:
                    self.status[task_id] = Status.SUCCESS
                    self.values[task_id] = value
                    self._emit(task_id, "success")
                else:
                    self.status[task_id] = Status.FAILED
                    if error is None:
                        error = RuntimeError("task failed before completing an attempt")
                    self.errors[task_id] = error
                    self._emit(task_id, "fail", f"{type(error).__name__}: {error}")
                try:
                    self._pump()
                finally:
                    self.cond.notify_all()
