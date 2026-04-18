import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
import os

# Save data to disk so it survives restarts
client = chromadb.PersistentClient(path="/data/chroma_db")

# Connect to Ollama's embedding model to convert text into vectors
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url=OLLAMA_URL
)

# Create (or reuse) a collection - like a table in a database
collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,  # Tells ChromaDB how to convert text to vectors
)

# Load the profile text and add it to the collection
with open('/app/profile.txt', 'r') as f:
    profile_text = f.read()


if collection.count() == 0:
    collection.add(
    documents=[profile_text],
    ids=["personal_profile"]
    )

print("Knowledge base built successfully. Profile data added to ChromaDB.")
