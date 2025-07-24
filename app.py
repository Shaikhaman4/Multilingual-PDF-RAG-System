import streamlit as st
import os
import tempfile
import shutil
import json

# Import your pipeline functions as refactored:
from prepare_pdf import extract_text_multilang
from chunk_text import chunk_text
from embed_db import embed_chunks
from query import run_rag_query

st.set_page_config(
    page_title="Document Q&A (RAG)",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Document Q&A – RAG Chatbot")
st.info(
    "Upload a PDF, ask questions in any language, and get instant answers using RAG technology."
)
st.markdown("---")

# Sidebar: Language and Settings
st.sidebar.header("App Settings")
answer_lang = st.sidebar.selectbox(
    "Preferred answer language",
    ["Auto Detect", "English", "বাংলা (Bengali)", "اردو (Urdu)"],
    index=0,
)
# 🟢 language mapping for OCR/extraction
lang_map = {
    "English": "eng",
    "বাংলা (Bengali)": "ben",
    "اردو (Urdu)": "urd"
}
st.sidebar.markdown("---")
with st.sidebar.expander("How it works"):
    st.write(
        "- Upload a PDF (digital or scanned)\n"
        "- Enter your question\n"
        "- The app extracts, chunks, embeds, and answers with RAG."
    )

# Main UI: File Upload and Query Input
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
example_questions = [
    "Summarize the main topic of this document.",
    "List official events and activities described.",
    "How are schools involved in the program?",
    "প্রধান বৈজ্ঞানিক ধাপগুলো কী কী?",  # Bengali
    "۱۲ ربیع الاول کے لئے کون سی سرگرمیاں ہیں؟",  # Urdu
]
question = st.text_input(
    "Enter your question:",
    help="You may ask in any language supported by your pipeline.",
    placeholder="E.g. What are the official Eid activities in this document?",
)
if st.button("Show Example Questions"):
    st.markdown("#### Examples:")
    for q in example_questions:
        st.write(f"- {q}")

st.markdown("---")

if uploaded_file and question:
    with st.spinner("🔄 Processing your PDF..."):
        # Create a true temp directory per session
        temp_dir = tempfile.mkdtemp()

        # Save PDF to disk
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # 🟢 Use correct language for text extraction
        lang_code = lang_map.get(answer_lang, "eng")  # Default to English if unspecified

        # 1. Extract text (digital or OCR)
        raw_text = extract_text_multilang(pdf_path, lang=lang_code)
        st.success("Extracted text from PDF.")

        # Preview extracted text
        with st.expander("Show Extracted Text (Preview)", expanded=False):
            st.text_area(
                "Text Preview",
                value=raw_text[:2000] + "..." if len(raw_text) > 2000 else raw_text,
                height=180,
            )

        # 2. Chunking
        st.info("Splitting text into retrievable segments...")
        chunks = chunk_text(raw_text)
        st.success(f"Chunked into {len(chunks)} segments.")

        # 3. Build a fresh vectorstore in a unique temp folder
        VECTORSTORE_DIR = os.path.join(temp_dir, "chroma_db")
        embed_chunks(chunks, VECTORSTORE_DIR)

        # 4. Save chunks to disk for query stage
        CHUNKS_FILE = os.path.join(temp_dir, "chunks.json")
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        st.success("Knowledge base prepared.")

    # Prompt for answer language
    prompt = question
    if answer_lang != "Auto Detect":
        lang_instructions = {
            "English": "Please answer in English.",
            "বাংলা (Bengali)": "অনুগ্রহ করে উত্তরটি বাংলায় দিন।",
            "اردو (Urdu)": "براہ کرم جواب اردو میں دیں۔",
        }
        prompt = f"{question} {lang_instructions.get(answer_lang, '')}"

    with st.spinner("💡 Generating answer with RAG pipeline..."):
        try:
            answer = run_rag_query(VECTORSTORE_DIR, prompt, CHUNKS_FILE)
            st.markdown("### 📢 Answer")
            st.markdown(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Clean up everything (chroma_db, chunks, temp file)
    shutil.rmtree(temp_dir, ignore_errors=True)

elif uploaded_file:
    st.warning("Please enter a question to get started!")
elif question:
    st.warning("Please upload a PDF file.")
else:
    st.info("Upload a PDF and enter your question to begin.")

st.markdown("---")
st.caption("Powered by Streamlit and RAG • Supports English, Bengali, Urdu, and more.")
