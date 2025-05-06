import faiss
import numpy as np

index = None
documents = []

def build_index(chunks):
    global index, documents
    from embedder import get_embeddings
    embeddings = get_embeddings(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    documents = chunks

def search(query, top_k=5):
    from embedder import get_embeddings
    query_embedding = get_embeddings([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]
