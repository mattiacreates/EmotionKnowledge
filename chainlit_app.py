import os
import chainlit as cl
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import chromadb
from chromadb.utils import embedding_functions

MODEL_NAME = os.getenv("HF_MODEL", "meta-llama/Llama-2-7b-chat-hf")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
DB_PATH = os.getenv("CHROMADB_PATH", "segment_db")
COLLECTION_NAME = os.getenv("CHROMADB_COLLECTION", "segments")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Load LLM from HuggingFace model hub
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
_llm = pipeline("text-generation", model=_model, tokenizer=_tokenizer, max_new_tokens=256)

# Connect to ChromaDB with diarized transcripts
_client = chromadb.PersistentClient(path=DB_PATH)
_collection = _client.get_collection(COLLECTION_NAME)
_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

@cl.on_message
async def main(message: str):
    """Respond to a user message with context from diarized transcripts."""
    query_embedding = _embedder(message)
    results = _collection.query(query_embeddings=[query_embedding], n_results=3)
    context = "\n".join(results["documents"][0])
    prompt = f"{SYSTEM_PROMPT}\n{context}\nUser: {message}\nAssistant:"
    output = _llm(prompt)[0]["generated_text"][len(prompt):]
    await cl.Message(content=output).send()
