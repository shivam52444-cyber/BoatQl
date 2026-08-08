"""
Manages the LangSmith annotation queue that flags runs needing human
review: anything that needed more than one SQL-generation attempt, or
failed all retries outright. These are exactly the cases your automated
eval can't fully judge -- a human should look at *why* the model needed
a retry (bad prompt? genuinely ambiguous schema? bad guardrail?).
"""

from langsmith import Client

QUEUE_NAME = "ai-sql-analyst-review"

_queue_id_cache = None


def get_or_create_queue(client: Client = None) -> str:
    global _queue_id_cache
    if _queue_id_cache is not None:
        return _queue_id_cache

    client = client or Client()

    existing = list(client.list_annotation_queues(name=QUEUE_NAME))
    if existing:
        _queue_id_cache = str(existing[0].id)
        return _queue_id_cache

    queue = client.create_annotation_queue(
        name=QUEUE_NAME,
        description="Queries that needed a retry or failed all retries -- "
                     "review to see if it's a prompt issue, schema gap, or "
                     "a validator bug.",
    )
    _queue_id_cache = str(queue.id)
    return _queue_id_cache


def queue_run_for_review(run_id: str, reason: str, client: Client = None) -> None:
    """Adds a run to the review queue. Never raises -- annotation queueing
    is a nice-to-have, it should never break the actual user-facing request
    if LangSmith is slow/unavailable."""
    if not run_id:
        return
    try:
        client = client or Client()
        queue_id = get_or_create_queue(client)
        client.add_runs_to_annotation_queue(queue_id, run_ids=[run_id])
        print(f"  [review queue] added run {run_id} ({reason})")
    except Exception as e:
        print(f"  [review queue] failed to queue run {run_id}: {e}")


if __name__ == "__main__":
    queue_id = get_or_create_queue()
    print(f"Queue ready: {queue_id}")