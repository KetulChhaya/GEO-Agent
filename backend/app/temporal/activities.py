from temporalio import activity


@activity.defn
async def run_audit_graph(run_id: str) -> str:
    """Phase 0 stub -- real implementation (invoking the LangGraph pipeline)
    lands in Phase 2, task 2.6."""
    activity.logger.info("run_audit_graph stub invoked for run_id=%s", run_id)
    return f"stub-completed:{run_id}"
