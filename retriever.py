import faiss
import numpy as np

def build_vector_store(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


def retrieve_chunks(query_embedding , index , chunks , k =3):
    distances , indices = index.search(query_embedding , k)
    
    results = [chunks[i] for i in indices[0]]
    
    return results