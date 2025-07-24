import json
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def embed_chunks(chunk_docs, vectorstore_dir):
    """
    chunk_docs: list of {"page_content": ..., "metadata": ...}
    vectorstore_dir: directory path for persistent Chroma DB

    Returns: collection (Chroma/ChromaDB object or None)
    """
    # Prepare documents
    texts = [doc["page_content"] for doc in chunk_docs]
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in chunk_docs]

    # Load embedder model
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Embed
    embeddings = embedder.embed_documents(texts)

    # Create ChromaDB persistent collection
    client = chromadb.PersistentClient(path=vectorstore_dir)
    collection = client.get_or_create_collection(name="rag_docs", metadata={"hnsw:space": "cosine"})
    ids = [f"doc-{i}" for i in range(len(texts))]

    # Add to collection
    for id_, emb, doc in zip(ids, embeddings, docs):
        collection.add(ids=[id_], embeddings=[emb], documents=[doc.page_content], metadatas=[doc.metadata])

    return collection  # or return vectorstore_dir for later querying
