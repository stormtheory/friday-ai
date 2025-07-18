# modules/rag.py
# Written by StormTheory
# https://github.com/stormtheory/friday-ai

import faiss, pickle, os
import numpy as np
from sentence_transformers import SentenceTransformer
from config import CONTEXT_DIR

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context_rag(query, thread, top_k=5):
    thread_vector_dir = os.path.join(CONTEXT_DIR, "vector_store", thread)

    if not os.path.exists(thread_vector_dir):
        return ""

    query_vec = embed_model.encode([query])
    query_vec = np.array(query_vec).astype("float32")

    all_results = []

    for fname in os.listdir(thread_vector_dir):
        if not fname.endswith(".index"):
            continue  # skip non-index files

        base = fname[:-6]  # strip ".index"
        index_path = os.path.join(thread_vector_dir, f"{base}.index")
        meta_path = os.path.join(thread_vector_dir, f"{base}_meta.pkl")

        if not os.path.exists(meta_path):
            continue

        try:
            index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                chunks = pickle.load(f)

            D, I = index.search(query_vec, min(top_k, index.ntotal))

            for dist, idx in zip(D[0], I[0]):
                if 0 <= idx < len(chunks):
                    all_results.append((dist, chunks[idx]))

        except Exception as e:
            print(f"⚠️ Failed to load vector data from {base}: {e}")
            continue

    if not all_results:
        return ""

    # Sort by distance (lower is better)
    all_results.sort(key=lambda x: x[0])
    top_chunks = [text for _, text in all_results[:top_k]]

    return "\n\n".join(top_chunks)

