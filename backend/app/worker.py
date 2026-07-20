import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from app.config import get_settings
from app.temporal.activities import run_audit_graph
from app.temporal.workflows import AuditWorkflow


async def main() -> None:
    settings = get_settings()
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[AuditWorkflow],
        activities=[run_audit_graph],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
