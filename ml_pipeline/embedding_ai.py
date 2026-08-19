from sentence_transformers import SentenceTransformer

# Load the model once globally so it doesn't reload on every request
# all-MiniLM-L6-v2 produces 384-dimensional embeddings (matching our pgvector column)
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text: str) -> list[float]:
    """
    Generates a 384-dimensional embedding for the given text.
    """
    if not text.strip():
        return [0.0] * 384
        
    embedding = model.encode(text)
    return embedding.tolist()
