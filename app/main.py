from fastapi import FastAPI, HTTPException
import ollama
import chromadb
from pydantic import BaseModel
import os
import time

app = FastAPI()

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Create Ollama client
ollama_client = ollama.Client(host=OLLAMA_URL)

# ChromaDB setup
client = chromadb.PersistentClient(path="/data/chroma_db")

ef = chromadb.utils.embedding_functions.ollama_embedding_function.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url=OLLAMA_URL
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef
)

# -----------------------------
# MODELS
# -----------------------------
class DocumentSubmission(BaseModel):
    user_name: str
    content: str

# -----------------------------
# HEALTH CHECK (VERY IMPORTANT)
# -----------------------------
@app.get("/")
def root():
    return {"message": "RAG API is running 🚀"}

# -----------------------------
# ADD DOCUMENT
# -----------------------------
@app.post("/documents")
def add_document(submission: DocumentSubmission):
    try:
        chunks = [
            chunk.strip()
            for chunk in submission.content.split("\n\n")
            if chunk.strip()
        ]

        if not chunks:
            raise HTTPException(status_code=400, detail="No valid content provided")

        collection.add(
            ids=[f"{submission.user_name}-chunk{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[
                {
                    "source": "profile",
                    "user_name": submission.user_name,
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ],
        )

        return {
            "message": f"Added {len(chunks)} chunks for user '{submission.user_name}'.",
            "user_name": submission.user_name,
            "chunks_added": len(chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# ASK QUESTION
# -----------------------------
@app.get("/ask")
def ask(question: str, user: str = None):
    try:
        # RETRIEVE
        query_params = {}
        if user:
            query_params["where"] = {"user_name": user}

        results = collection.query(
            query_texts=[question],
            n_results=2,
            **query_params
        )

        documents = results.get("documents", [[]])

        if not documents or not documents[0]:
            return {
                "question": question,
                "answer": "No relevant information found.",
                "context_used": [],
                "filtered_by_user": user,
            }

        context = "\n\n".join(documents[0])

        # AUGMENT
        augmented_prompt = f"""Use the following context to answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}"""

        # Small retry (important for cloud cold starts)
        for attempt in range(3):
            try:
                response = ollama_client.chat(
                    model="qwen2.5:0.5b",
                    messages=[{"role": "user", "content": augmented_prompt}],
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(2)

        return {
            "question": question,
            "answer": response["message"]["content"],
            "context_used": documents[0],
            "filtered_by_user": user,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))