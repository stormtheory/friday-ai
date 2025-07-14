# Written by StormTheory
# https://github.com/stormtheory/friday-ai

# modules/rag.py
import faiss, pickle, os
import numpy as np
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context_rag(query, thread, top_k=5):
    index_path = f"vector_store/{thread}.index"
    meta_path = f"vector_store/{thread}_meta.pkl"

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return ""

    query_vec = embed_model.encode([query])
    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    D, I = index.search(np.array(query_vec), top_k)
    results = [chunks[i] for i in I[0] if i < len(chunks)]
    return "\n\n".join(results)
