from sentence_transformers import SentenceTransformer
import numpy as np

def load_embedding_model():
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('embeddings.model')
    #model.save('embeddings.model')
    return model

def compute_embedding(model, text):
    return model.encode([text])[0]

load_embedding_model()