"""A small sales pipeline used to show the scheduler.

``extract`` fans out into two normalizers. ``quality_check`` only needs the
orders branch, so it can run while ``normalize_users`` is still in progress.
``publish`` waits for both the notification and the quality check. Failing the
check skips ``publish`` and leaves the report branch in place.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from dag.graph import DAG


def build_pipeline(*, fail_check: bool = False, delay: float = 0.0) -> DAG:
    """Build the sample DAG.

    ``delay`` is a base sleep in seconds so a CLI run can show overlap.
    Tests leave it at ``0``. ``fail_check`` makes ``quality_check`` raise.
    """
    if delay < 0:
        raise ValueError("delay must be >= 0")

    dag = DAG("sales_pipeline")

    def pause(scale: float) -> None:
        if delay > 0:
            time.sleep(delay * scale)

    @dag.task
    def extract(_upstream: Mapping[str, Any]) -> dict[str, Any]:
        pause(0.5)
        return {
            "users": [
                {"id": 1, "name": "Ada"},
                {"id": 2, "name": "Max"},
            ],
            "orders": [
                {"user_id": 1, "sku": "book", "amount": 30},
                {"user_id": 1, "sku": "pen", "amount": 5},
                {"user_id": 2, "sku": "book", "amount": 30},
            ],
        }

    @dag.task
    def normalize_users(upstream: Mapping[str, Any]) -> dict[int, str]:
        pause(2.0)
        return {user["id"]: user["name"] for user in upstream["extract"]["users"]}

    @dag.task
    def normalize_orders(upstream: Mapping[str, Any]) -> list[dict[str, Any]]:
        pause(0.5)
        return list(upstream["extract"]["orders"])

    @dag.task
    def quality_check(upstream: Mapping[str, Any]) -> dict[str, Any]:
        pause(0.4)
        orders = upstream["normalize_orders"]
        if fail_check:
            raise ValueError("forced quality check failure")
        if any(order["amount"] < 0 for order in orders):
            raise ValueError("negative amount")
        return {"ok": True, "order_count": len(orders)}

    @dag.task
    def build_report(upstream: Mapping[str, Any]) -> dict[str, Any]:
        pause(0.5)
        names: dict[int, str] = upstream["normalize_users"]
        orders = upstream["normalize_orders"]
        by_user: dict[str, int] = {}
        for order in orders:
            name = names[order["user_id"]]
            by_user[name] = by_user.get(name, 0) + order["amount"]
        return {"by_user": by_user, "total": sum(by_user.values())}

    @dag.task
    def notify(upstream: Mapping[str, Any]) -> str:
        pause(0.3)
        report = upstream["build_report"]
        parts = ", ".join(
            f"{name}={amount}" for name, amount in sorted(report["by_user"].items())
        )
        return f"revenue={report['total']} ({parts})"

    @dag.task
    def publish(upstream: Mapping[str, Any]) -> str:
        pause(0.3)
        check = upstream["quality_check"]
        message = upstream["notify"]
        return f"published {message}; orders={check['order_count']}"

    extract >> [normalize_users, normalize_orders]
    normalize_users >> build_report
    normalize_orders >> [build_report, quality_check]
    build_report >> notify
    [notify, quality_check] >> publish
    return dag
