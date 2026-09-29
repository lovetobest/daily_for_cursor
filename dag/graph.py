"""Tasks and the directed acyclic graph that connects them.

A node is one task. A directed edge ``u -> v`` means ``v`` consumes the result of
``u`` and must not start until ``u`` has succeeded. The graph is acyclic so a
finite execution order always exists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


class CycleError(ValueError):
    """Raised when edges form a directed cycle.

    ``nodes`` lists every task that Kahn's algorithm could not finish. That set
    contains the cycle and any task that is only reachable through the cycle.
    """

    def __init__(self, nodes: list[str]) -> None:
        self.nodes = tuple(nodes)
        joined = ", ".join(self.nodes)
        super().__init__(f"cycle detected; unfinished tasks: {joined}")


class Task:
    """One unit of work registered on a :class:`DAG`.

    ``fn`` is called as ``fn(upstream)`` where ``upstream`` maps each direct
    parent task id to the value that parent returned. The task object itself is
    what ``>>`` connects; the original function is not wrapped for direct calls.
    """

    def __init__(
        self,
        task_id: str,
        fn: Callable[[Mapping[str, Any]], Any],
        *,
        retries: int,
        dag: DAG,
    ) -> None:
        self.task_id = task_id
        self.fn = fn
        self.retries = retries
        self.dag = dag
        self._upstream: list[str] = []
        self._downstream: list[str] = []

    @property
    def upstream(self) -> tuple[str, ...]:
        return tuple(self._upstream)

    @property
    def downstream(self) -> tuple[str, ...]:
        return tuple(self._downstream)

    def __rshift__(self, other: Task | TaskSet | list[Task] | tuple[Task, ...]) -> Task | TaskSet:
        """Declare ``self -> other``. A list on the right fans out to every item."""
        targets = _coerce_tasks(other)
        for target in targets:
            _link(self, target)
        if isinstance(other, Task):
            return other
        return TaskSet(targets)

    def __rrshift__(self, other: list[Task] | tuple[Task, ...]) -> Task:
        """Declare ``other -> self`` when ``other`` is a plain list of tasks."""
        for source in _coerce_tasks(other):
            _link(source, self)
        return self

    def __repr__(self) -> str:
        return f"Task({self.task_id!r})"


class TaskSet:
    """Several tasks acting as one side of ``>>`` so group edges can chain."""

    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = tuple(tasks)

    def __rshift__(self, other: Task | TaskSet | list[Task] | tuple[Task, ...]) -> Task | TaskSet:
        targets = _coerce_tasks(other)
        for source in self._tasks:
            for target in targets:
                _link(source, target)
        if isinstance(other, Task):
            return other
        return TaskSet(targets)

    def __rrshift__(self, other: list[Task] | tuple[Task, ...]) -> TaskSet:
        for source in _coerce_tasks(other):
            for target in self._tasks:
                _link(source, target)
        return self

    def __repr__(self) -> str:
        ids = ", ".join(task.task_id for task in self._tasks)
        return f"TaskSet({ids})"


class DAG:
    """A named collection of tasks and dependency edges."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._tasks: dict[str, Task] = {}
        self._edges: list[tuple[str, str]] = []

    def task(
        self,
        fn: Callable[[Mapping[str, Any]], Any] | None = None,
        *,
        retries: int = 0,
        task_id: str | None = None,
    ) -> Task | Callable[[Callable[[Mapping[str, Any]], Any]], Task]:
        """Register a task function. Usable as ``@dag.task`` or ``@dag.task(retries=2)``."""

        def decorate(func: Callable[[Mapping[str, Any]], Any]) -> Task:
            if retries < 0:
                raise ValueError("retries must be >= 0")
            resolved_id = func.__name__ if task_id is None else task_id
            if not resolved_id:
                raise ValueError("task_id must be non-empty")
            if resolved_id in self._tasks:
                raise ValueError(f"duplicate task id: {resolved_id}")
            node = Task(resolved_id, func, retries=retries, dag=self)
            self._tasks[resolved_id] = node
            return node

        if fn is not None:
            return decorate(fn)
        return decorate

    def edges(self) -> list[tuple[str, str]]:
        """Dependency pairs ``(upstream, downstream)`` in the order they were declared."""
        return list(self._edges)

    def topological_levels(self) -> list[list[str]]:
        """Group tasks into waves with Kahn's algorithm.

        Wave *k* contains every task whose parents all sit in earlier waves.
        Tasks inside one wave have no edges between them, so a bulk-synchronous
        runner could execute that wave in parallel. :class:`dag.scheduler.Scheduler`
        uses a finer rule and may start a later task while an unrelated task
        from an earlier wave is still running.
        """
        missing = [
            f"{task.task_id!r}->{upstream_id!r}"
            for task in self._tasks.values()
            for upstream_id in task._upstream
            if upstream_id not in self._tasks
        ]
        if missing:
            raise ValueError("unknown upstream tasks: " + ", ".join(missing))

        indegree = {task_id: len(task.upstream) for task_id, task in self._tasks.items()}
        ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
        levels: list[list[str]] = []
        seen = 0
        while ready:
            levels.append(ready)
            seen += len(ready)
            nxt: list[str] = []
            for task_id in ready:
                for downstream_id in self._tasks[task_id].downstream:
                    indegree[downstream_id] -= 1
                    if indegree[downstream_id] == 0:
                        nxt.append(downstream_id)
            ready = sorted(nxt)
        if seen != len(self._tasks):
            leftover = sorted(task_id for task_id, degree in indegree.items() if degree > 0)
            raise CycleError(leftover)
        return levels

    def __getitem__(self, task_id: str) -> Task:
        return self._tasks[task_id]

    def __contains__(self, task_id: object) -> bool:
        return task_id in self._tasks

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks)

    def __repr__(self) -> str:
        return f"DAG({self.name!r}, tasks={len(self._tasks)})"


def _coerce_tasks(other: Task | TaskSet | list[Task] | tuple[Task, ...]) -> list[Task]:
    if isinstance(other, Task):
        return [other]
    if isinstance(other, TaskSet):
        return list(other._tasks)
    if isinstance(other, (list, tuple)):
        tasks = list(other)
        if not tasks or not all(isinstance(item, Task) for item in tasks):
            raise TypeError("expected a task or a non-empty sequence of tasks")
        return tasks
    raise TypeError("expected a task or a sequence of tasks")


def _link(upstream: Task, downstream: Task) -> None:
    if upstream.dag is not downstream.dag:
        raise ValueError(
            f"cannot connect {upstream.task_id!r} to {downstream.task_id!r}: different DAGs"
        )
    if upstream.task_id == downstream.task_id:
        raise ValueError(f"task {upstream.task_id!r} cannot depend on itself")
    if upstream.task_id in downstream._upstream:
        return
    upstream._downstream.append(downstream.task_id)
    downstream._upstream.append(upstream.task_id)
    upstream.dag._edges.append((upstream.task_id, downstream.task_id))
