import json
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = "GROQ_API_KEY"

def enrich_with_metadata(chunks):
    from langchain.schema import Document
    return [Document(page_content=chunk["page_content"], metadata=chunk["metadata"]) for chunk in chunks]

def run_rag_query(vectorstore_dir, query, chunks_json="chunks.json"):
    with open(chunks_json, encoding="utf8") as f:
        chunk_docs = json.load(f)
    documents = enrich_with_metadata(chunk_docs)
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(collection_name="rag_docs", embedding_function=embedding_fn, persist_directory=vectorstore_dir)
    bm25_retriever = BM25Retriever.from_documents(documents, k=10)
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.4, 0.6])
    memory = ConversationBufferMemory(k=10, return_messages=True)

    # Compose message
    def rag_chain(user_query, memory, retriever):
        all_docs = retriever.invoke(user_query)
        context = "\n\n".join([doc.page_content for doc in all_docs[:5]])
        messages = []
        for m in memory.load_memory_variables({})['history']:
            if isinstance(m, dict):
                messages.append(m)
            elif isinstance(m, tuple) and len(m) == 2:
                messages.append({"role": m[0], "content": m[1]})
            elif hasattr(m, "role") and hasattr(m, "content"):
                messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"})
        llm = ChatGroq(model_name=LLM_MODEL, temperature=0.0, api_key=GROQ_API_KEY)
        answer = llm.invoke(messages)
        memory.save_context({"input": user_query}, {"output": answer.content if hasattr(answer, "content") else answer})
        return answer.content if hasattr(answer, "content") else answer

    return rag_chain(query, memory, ensemble_retriever)
