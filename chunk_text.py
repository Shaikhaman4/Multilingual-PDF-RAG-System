from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=800, overlap=200):
    """
    Splits text into LangChain Document chunks.
    Returns a list of dicts: {"page_content": ..., "metadata": {"chunk_id": ...}}
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.create_documents([text])
    return [
        {"page_content": chunk.page_content, "metadata": {"chunk_id": i}}
        for i, chunk in enumerate(chunks)
    ]
