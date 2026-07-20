from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from app.temporal.activities import run_audit_graph


@workflow.defn
class AuditWorkflow:
    """Phase 0: no-op end-to-end wiring check. Real retry policy (3 attempts,
    15 min timeout) and the actual graph invocation land in Phase 2 task 2.6."""

    @workflow.run
    async def run(self, run_id: str) -> str:
        return await workflow.execute_activity(
            run_audit_graph,
            run_id,
            start_to_close_timeout=timedelta(seconds=30),
        )
