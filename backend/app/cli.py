"""Command-line entry for running the audit graph by hand (Phase 1, task 1.5).

    python -m app.cli audit https://example.com

Runs the partial Phase-1 graph (crawl -> chunk/embed) against a real Postgres
checkpointer and prints a summary of the resulting state. This is the debug/dev
path; Temporal becomes the real execution path from Phase 5 (the CLI stays as a
tool). Needs OPENAI_API_KEY, a reachable Postgres, and network access.
"""

import argparse
import asyncio
import uuid
from typing import Any

from app.graph.builder import build_graph, open_checkpointer
from app.graph.state import AuditState


def _print_summary(run_id: str, state: dict[str, Any]) -> None:
    """Pretty-print the final graph state. `ainvoke` returns the state as a plain
    dict (LangGraph serializes the Pydantic model back out)."""
    pages = state.get("pages", [])
    errors = state.get("errors", [])

    print("\n" + "=" * 60)
    print(f"  audit {run_id}")
    print("=" * 60)
    print(f"  url         : {state.get('url')}")
    print(f"  status      : {state.get('status')}")
    print(f"  brand_name  : {state.get('brand_name')}")
    print(f"  pages       : {len(pages)}")
    print(f"  chunks      : {state.get('chunk_count', 0)}")
    if pages:
        print("  crawled     :")
        for page in pages:
            print(f"    - {page.url}  ({page.word_count} words)")
    if errors:
        print("  errors      :")
        for err in errors:
            print(f"    - {err}")
    print("=" * 60)


async def _run_audit(url: str) -> None:
    run_id = str(uuid.uuid4())
    # thread_id = run_id ties this run to its checkpoints, so a retry with the
    # same id resumes from the last completed node instead of re-crawling.
    async with open_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        result = await graph.ainvoke(
            AuditState(run_id=run_id, url=url),
            config={"configurable": {"thread_id": run_id}},
        )
    _print_summary(run_id, dict(result))


def main() -> None:
    parser = argparse.ArgumentParser(prog="app.cli", description="CiteSight audit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="run the audit graph against a URL")
    audit_parser.add_argument("url", help="the site URL to audit, e.g. https://example.com")

    args = parser.parse_args()
    if args.command == "audit":
        asyncio.run(_run_audit(args.url))


if __name__ == "__main__":
    main()
