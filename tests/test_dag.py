"""Tests for the DAG graph and the dynamic scheduler."""

import threading
import unittest

from dag import DAG, CycleError, Scheduler, Status, Task
from dag.demo import build_pipeline
from dag.format import format_graph, format_results, format_timeline


class GraphTests(unittest.TestCase):
    def test_chain_fan_out_and_fan_in_levels(self) -> None:
        dag = DAG("diamond")

        @dag.task
        def a(_upstream):
            return "a"

        @dag.task
        def b(_upstream):
            return "b"

        @dag.task
        def c(_upstream):
            return "c"

        @dag.task
        def d(_upstream):
            return "d"

        a >> [b, c] >> d
        self.assertEqual(
            dag.edges(),
            [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        )
        self.assertEqual(dag.topological_levels(), [["a"], ["b", "c"], ["d"]])

    def test_list_on_the_left_fans_in(self) -> None:
        dag = DAG("fan-in")

        @dag.task
        def a(_upstream):
            return 1

        @dag.task
        def b(_upstream):
            return 1

        @dag.task
        def c(_upstream):
            return 1

        [a, b] >> c
        self.assertEqual(sorted(c.upstream), ["a", "b"])
        self.assertEqual(dag.topological_levels(), [["a", "b"], ["c"]])

    def test_duplicate_edge_is_stored_once(self) -> None:
        dag = DAG("dup")

        @dag.task
        def a(_upstream):
            return 1

        @dag.task
        def b(_upstream):
            return 1

        a >> b
        a >> b
        self.assertEqual(dag.edges(), [("a", "b")])
        self.assertEqual(dag.topological_levels(), [["a"], ["b"]])

    def test_cycle_names_unfinished_tasks(self) -> None:
        dag = DAG("cycle")

        @dag.task
        def a(_upstream):
            return 1

        @dag.task
        def b(_upstream):
            return 1

        @dag.task
        def c(_upstream):
            return 1

        a >> b >> c >> a
        with self.assertRaises(CycleError) as caught:
            dag.topological_levels()
        self.assertCountEqual(caught.exception.nodes, ("a", "b", "c"))

    def test_self_edge_is_rejected(self) -> None:
        dag = DAG("self")

        @dag.task
        def a(_upstream):
            return 1

        with self.assertRaisesRegex(ValueError, "cannot depend on itself"):
            a >> a

    def test_tasks_from_different_dags_cannot_connect(self) -> None:
        left = DAG("left")
        right = DAG("right")

        @left.task
        def a(_upstream):
            return 1

        @right.task
        def b(_upstream):
            return 1

        with self.assertRaisesRegex(ValueError, "different DAGs"):
            a >> b

    def test_duplicate_task_id_is_rejected(self) -> None:
        dag = DAG("dup-id")

        @dag.task
        def a(_upstream):
            return 1

        with self.assertRaisesRegex(ValueError, "duplicate task id"):
            dag.task(task_id="a")(lambda _upstream: 2)

    def test_empty_task_id_is_rejected(self) -> None:
        dag = DAG("empty-id")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            dag.task(task_id="")(lambda _upstream: 1)

    def test_negative_retries_are_rejected(self) -> None:
        dag = DAG("retries")
        with self.assertRaisesRegex(ValueError, "retries"):
            dag.task(retries=-1)(lambda _upstream: 1)


class SchedulerTests(unittest.TestCase):
    def test_parents_pass_only_their_own_results(self) -> None:
        dag = DAG("values")

        @dag.task
        def a(_upstream):
            return 2

        @dag.task
        def b(upstream):
            self.assertEqual(set(upstream), {"a"})
            return upstream["a"] + 3

        @dag.task
        def c(upstream):
            self.assertEqual(set(upstream), {"a"})
            return upstream["a"] * 4

        a >> [b, c]
        report = Scheduler().run(dag)
        self.assertEqual(report.values, {"a": 2, "b": 5, "c": 8})
        self.assertTrue(report.succeeded)
        self.assertEqual(report.attempts["b"], 1)

    def test_dependency_order_beats_task_id_order(self) -> None:
        dag = DAG("order")

        @dag.task
        def b(_upstream):
            return "b"

        @dag.task(task_id="a")
        def later(_upstream):
            return "a"

        b >> later
        report = Scheduler(max_workers=1).run(dag)
        starts = [event.task_id for event in report.events if event.kind == "start"]
        self.assertEqual(starts, ["b", "a"])

    def test_single_worker_starts_ready_roots_by_id(self) -> None:
        dag = DAG("serial-roots")

        @dag.task
        def b(_upstream):
            return "b"

        @dag.task
        def a(_upstream):
            return "a"

        report = Scheduler(max_workers=1).run(dag)
        starts = [event.task_id for event in report.events if event.kind == "start"]
        self.assertEqual(starts, ["a", "b"])

    def test_independent_tasks_run_at_the_same_time(self) -> None:
        dag = DAG("parallel")
        barrier = threading.Barrier(2, timeout=2)

        @dag.task
        def left(_upstream):
            barrier.wait()
            return "left"

        @dag.task
        def right(_upstream):
            barrier.wait()
            return "right"

        report = Scheduler(max_workers=2).run(dag)
        self.assertEqual(report.values, {"left": "left", "right": "right"})

    def test_fast_branch_does_not_wait_for_unrelated_task(self) -> None:
        dag = DAG("overlap")
        release_slow = threading.Event()
        quality_started = threading.Event()

        @dag.task
        def extract(_upstream):
            return 1

        @dag.task
        def slow_users(upstream):
            self.assertTrue(release_slow.wait(timeout=2))
            return upstream["extract"]

        @dag.task
        def fast_orders(upstream):
            return upstream["extract"]

        @dag.task
        def quality(upstream):
            quality_started.set()
            return upstream["fast_orders"]

        extract >> [slow_users, fast_orders]
        fast_orders >> quality

        holder: dict[str, object] = {}

        def run() -> None:
            holder["report"] = Scheduler(max_workers=4).run(dag)

        runner = threading.Thread(target=run, daemon=True)
        runner.start()
        started = quality_started.wait(timeout=2)
        release_slow.set()
        runner.join(timeout=2)
        self.assertTrue(started)
        self.assertFalse(runner.is_alive())
        report = holder["report"]
        self.assertTrue(report.succeeded)

    def test_failed_parent_skips_the_join_and_keeps_the_sibling(self) -> None:
        dag = DAG("partial")
        called = []

        @dag.task
        def a(_upstream):
            raise RuntimeError("boom")

        @dag.task
        def b(_upstream):
            called.append("b")
            return "sibling"

        @dag.task
        def left(_upstream):
            called.append("left")
            return "nope"

        @dag.task
        def join(_upstream):
            called.append("join")
            return "nope"

        a >> left
        [left, b] >> join
        report = Scheduler().run(dag)
        self.assertEqual(called, ["b"])
        self.assertIs(report.status["a"], Status.FAILED)
        self.assertIs(report.status["left"], Status.SKIPPED)
        self.assertIs(report.status["b"], Status.SUCCESS)
        self.assertIs(report.status["join"], Status.SKIPPED)
        self.assertTrue(report.failed)
        self.assertNotIn("join", report.values)
        self.assertIn("boom", str(report.errors["a"]))

    def test_skip_propagates_through_a_chain(self) -> None:
        dag = DAG("chain-skip")
        called = []

        @dag.task
        def a(_upstream):
            raise RuntimeError("root")

        @dag.task
        def b(_upstream):
            called.append("b")
            return 1

        @dag.task
        def c(_upstream):
            called.append("c")
            return 1

        a >> b >> c
        report = Scheduler().run(dag)
        self.assertEqual(called, [])
        self.assertIs(report.status["b"], Status.SKIPPED)
        self.assertIs(report.status["c"], Status.SKIPPED)

    def test_retry_then_success(self) -> None:
        dag = DAG("retry")
        calls = {"n": 0}

        @dag.task(retries=2)
        def flaky(_upstream):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("not yet")
            return "yes"

        report = Scheduler().run(dag)
        self.assertEqual(calls["n"], 3)
        self.assertEqual(report.values["flaky"], "yes")
        self.assertEqual(report.attempts["flaky"], 3)
        kinds = [event.kind for event in report.events if event.task_id == "flaky"]
        self.assertEqual(kinds, ["start", "retry", "retry", "success"])

    def test_retries_exhausted_skip_downstream(self) -> None:
        dag = DAG("give-up")

        @dag.task(retries=1)
        def flaky(_upstream):
            raise RuntimeError("nope")

        @dag.task
        def after(upstream):
            return upstream["flaky"]

        flaky >> after
        report = Scheduler().run(dag)
        self.assertEqual(report.attempts["flaky"], 2)
        self.assertIs(report.status["flaky"], Status.FAILED)
        self.assertIs(report.status["after"], Status.SKIPPED)
        kinds = [event.kind for event in report.events if event.task_id == "flaky"]
        self.assertEqual(kinds, ["start", "retry", "fail"])

    def test_event_sequence_is_contiguous_and_time_ordered(self) -> None:
        dag = DAG("events")

        @dag.task
        def a(_upstream):
            return 1

        @dag.task
        def b(upstream):
            return upstream["a"] + 1

        a >> b
        report = Scheduler().run(dag)
        self.assertEqual([event.seq for event in report.events], [1, 2, 3, 4])
        times = [event.elapsed_ms for event in report.events]
        self.assertEqual(times, sorted(times))
        success_a = next(
            event.seq for event in report.events if event.task_id == "a" and event.kind == "success"
        )
        start_b = next(
            event.seq for event in report.events if event.task_id == "b" and event.kind == "start"
        )
        self.assertLess(success_a, start_b)
        row_ids = [line.split()[0] for line in format_timeline(report).splitlines()]
        self.assertEqual(row_ids, ["a", "b"])

    def test_cycle_is_rejected_before_any_task_runs(self) -> None:
        dag = DAG("no-run")
        called = []

        @dag.task
        def a(_upstream):
            called.append("a")
            return 1

        @dag.task
        def b(_upstream):
            called.append("b")
            return 1

        a >> b >> a
        with self.assertRaises(CycleError):
            Scheduler().run(dag)
        self.assertEqual(called, [])

    def test_empty_dag_succeeds(self) -> None:
        report = Scheduler().run(DAG("empty"))
        self.assertTrue(report.succeeded)
        self.assertFalse(report.failed)
        self.assertEqual(report.events, ())

    def test_max_workers_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_workers"):
            Scheduler(max_workers=0)

    def test_registered_task_is_a_task_object(self) -> None:
        dag = DAG("type")

        @dag.task
        def a(_upstream):
            return 1

        self.assertIsInstance(a, Task)
        self.assertIn("a", dag)


class DemoPipelineTests(unittest.TestCase):
    def test_pipeline_totals_and_publish_message(self) -> None:
        report = Scheduler().run(build_pipeline())
        self.assertTrue(report.succeeded)
        self.assertEqual(report.values["build_report"]["total"], 65)
        self.assertEqual(report.values["build_report"]["by_user"], {"Ada": 35, "Max": 30})
        self.assertEqual(report.values["notify"], "revenue=65 (Ada=35, Max=30)")
        self.assertEqual(
            report.values["publish"],
            "published revenue=65 (Ada=35, Max=30); orders=3",
        )
        self.assertEqual(
            report.levels,
            [
                ["extract"],
                ["normalize_orders", "normalize_users"],
                ["build_report", "quality_check"],
                ["notify"],
                ["publish"],
            ],
        )
        text = format_graph(build_pipeline())
        self.assertIn("extract -> normalize_users", text)
        self.assertIn("[1] normalize_orders, normalize_users", text)
        rendered = format_results(report)
        self.assertIn("publish", rendered)
        self.assertIn("success", rendered)

    def test_failed_check_skips_publish_and_keeps_the_report(self) -> None:
        report = Scheduler().run(build_pipeline(fail_check=True))
        self.assertTrue(report.failed)
        self.assertIs(report.status["quality_check"], Status.FAILED)
        self.assertIs(report.status["publish"], Status.SKIPPED)
        self.assertIs(report.status["build_report"], Status.SUCCESS)
        self.assertIs(report.status["notify"], Status.SUCCESS)
        self.assertNotIn("publish", report.values)
        self.assertEqual(report.values["build_report"]["total"], 65)
        rendered = format_results(report)
        self.assertIn("forced quality check failure", rendered)
        self.assertIn("skipped", rendered)


if __name__ == "__main__":
    unittest.main()
