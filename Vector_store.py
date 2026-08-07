"""
Embeds schema table docs into a local Chroma vector store, and exposes
retrieve_relevant_tables() for semantic "which tables matter for this
query" lookup.

Uses sentence-transformers/all-MiniLM-L6-v2 run LOCALLY (downloaded once,
then no network calls needed) -- avoids depending on HF's hosted Inference
API, which is unreliable/inconsistent for many models. Groq doesn't offer
an embeddings endpoint, so this is a separate model from whichever LLM
(Groq/OpenAI) you use for SQL generation.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

from embed_schema import build_embedding_docs

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "schema_tables"


def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )


def build_vector_store():
    """Run once (or whenever schema changes) to (re)populate the store."""
    collection = get_collection()
    docs = build_embedding_docs()

    collection.upsert(
        ids=[d["table_name"] for d in docs],
        documents=[d["content"] for d in docs],
        metadatas=[{"table_name": d["table_name"]} for d in docs],
    )

    print(f"Indexed {len(docs)} tables into '{COLLECTION_NAME}'.")
    return collection


def retrieve_relevant_tables(query: str, top_k: int = 5, collection=None) -> list[str]:
    """Returns the top_k table names most semantically relevant to the query."""
    collection = collection or get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    return [m["table_name"] for m in results["metadatas"][0]]


if __name__ == "__main__":
    build_vector_store()

    test_query = "top 5 products by revenue last quarter"
    tables = retrieve_relevant_tables(test_query)
    print(f"\nQuery: {test_query!r}")
    print("Retrieved tables:", tables)