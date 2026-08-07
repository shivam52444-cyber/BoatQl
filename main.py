import time

t0 = time.perf_counter()
import chromadb
print(f"chromadb import: {time.perf_counter() - t0:.2f}s")

t0 = time.perf_counter()
from chromadb.utils import embedding_functions
embed_fn = embedding_functions.DefaultEmbeddingFunction()
print(f"DefaultEmbeddingFunction() init: {time.perf_counter() - t0:.2f}s")

t0 = time.perf_counter()
result = embed_fn(["test query"])
print(f"first embed call: {time.perf_counter() - t0:.2f}s")

t0 = time.perf_counter()
result = embed_fn(["another query"])
print(f"second embed call: {time.perf_counter() - t0:.2f}s")