from sentence_transformers import SentenceTransformer

model_id = "all-MiniLM-L6-v2"

model = SentenceTransformer(model_id)

def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    
    return embeddings