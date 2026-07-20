"""Manual-only: prerequisite `docker compose up -d postgres temporal temporal-ui worker`
must already be running. Not run in CI (no live Temporal server there).
Run explicitly with: uv run pytest -m manual
"""

import uuid

import pytest
from temporalio.client import Client

from app.config import get_settings
from app.temporal.workflows import AuditWorkflow

pytestmark = pytest.mark.manual


async def test_noop_workflow_completes_end_to_end() -> None:
    settings = get_settings()
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    result = await client.execute_workflow(
        AuditWorkflow.run,
        "manual-test-run-id",
        id=f"test-workflow-{uuid.uuid4()}",
        task_queue=settings.temporal_task_queue,
    )
    assert result == "stub-completed:manual-test-run-id"
