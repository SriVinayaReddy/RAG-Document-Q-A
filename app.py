import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# -----------------------------
# Load Environment Variables
# -----------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# -----------------------------
# Streamlit Page Configuration
# -----------------------------

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Q&A System")

st.write("Upload a PDF and ask questions about your document.")

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

# -----------------------------
# Process Uploaded PDF
# -----------------------------

vectorstore = None

if uploaded_file is not None:

    os.makedirs("data", exist_ok=True)

    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF Uploaded Successfully")

    # -----------------------------
    # Load PDF
    # -----------------------------

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    st.info(f"📄 Number of Pages : {len(documents)}")

    # -----------------------------
    # Split into Chunks
    # -----------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    st.info(f"✂ Number of Chunks : {len(chunks)}")

    # -----------------------------
    # Create Embeddings
    # -----------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.success("✅ Embedding Model Loaded")

    # -----------------------------
    # Create FAISS Vector Store
    # -----------------------------

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    st.success("✅ FAISS Vector Database Created")

# -----------------------------
# Ask Question
# -----------------------------

question = st.text_input(
    "💬 Ask a question from your PDF"
)

# -----------------------------
# Ask Button
# -----------------------------

if st.button("Ask Question"):

    if uploaded_file is None:

        st.warning("Please upload a PDF first.")

    elif question.strip() == "":

        st.warning("Please enter a question.")

    else:

        # -----------------------------
        # Retrieve Similar Chunks
        # -----------------------------

        retrieved_docs = vectorstore.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        # -----------------------------
        # Create Prompt
        # -----------------------------

        prompt = f"""
You are an AI Assistant.

Answer ONLY using the context below.

If the answer is not available in the context,
say

"I couldn't find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

        # -----------------------------
        # Load Groq Model
        # -----------------------------

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )

        # -----------------------------
        # Generate Response
        # -----------------------------

        response = llm.invoke(prompt)

        st.subheader("📖 Answer")

        st.write(response.content)

        # -----------------------------
        # Display Retrieved Context
        # -----------------------------

        with st.expander("🔍 Retrieved Context"):

            for i, doc in enumerate(retrieved_docs):

                st.markdown(f"### Chunk {i+1}")

                st.write(doc.page_content)

                st.write("-------------------------------------")