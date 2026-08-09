#!/bin/sh
set -eu

export DATA_DIR="${DATA_DIR:-/app/Dataset}"
export DB_PATH="${DB_PATH:-/app/data/database.db}"
READY_MARKER="${DB_PATH}.ready"

echo "===> Checking database..."
if [ ! -s "$DB_PATH" ] || [ ! -f "$READY_MARKER" ]; then
    echo "     Not found -- downloading dataset and building DuckDB database..."
    python DatasetDownload.py
    python setdb.py
    touch "$READY_MARKER"
else
    echo "     Already present at $DB_PATH, skipping."
fi

echo "===> Checking vector store..."
CHROMA_DIR="${CHROMA_PERSIST_DIR:-/app/chroma_db}"
if [ -d "$CHROMA_DIR" ] && [ "$(ls -A "$CHROMA_DIR" 2>/dev/null)" ]; then
    echo "     Already present at $CHROMA_DIR, skipping."
else
    echo "     Building..."
    python Vector_store.py
fi

echo "===> Starting API server..."
# exec replaces this shell with uvicorn, so SIGTERM from `docker stop`
# reaches uvicorn directly for a clean shutdown instead of being swallowed
# by the wrapper shell.
exec "$@"