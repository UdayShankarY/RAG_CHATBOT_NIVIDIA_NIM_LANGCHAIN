import os
import time
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error("NVIDIA_API_KEY not found in .env")
    st.stop()

os.environ["NVIDIA_API_KEY"] = api_key

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="NVIDIA RAG Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 NVIDIA NIM RAG Demo")

st.markdown("Upload one or more PDF files and ask questions about them.")

# -------------------------------------------------------
# Load LLM
# -------------------------------------------------------

llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    temperature=0
)

# -------------------------------------------------------
# Prompt Template
# -------------------------------------------------------

prompt = ChatPromptTemplate.from_template(
"""
You are an AI assistant.

Answer the user's question ONLY using the provided context.

If the answer is not present in the context, reply:

"I don't know based on the provided documents."

<context>
{context}
</context>

Question:
{input}
"""
)

# -------------------------------------------------------
# File Upload
# -------------------------------------------------------

uploaded_files = st.file_uploader(
    "📄 Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# -------------------------------------------------------
# Vector Store Creation
# -------------------------------------------------------

def vector_embedding():

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        return

    with st.spinner("Processing PDF(s)..."):

        embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5"
        )

        docs = []

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            loader = PyPDFLoader(temp_path)

            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=50
        )

        final_documents = splitter.split_documents(docs)

        vectors = FAISS.from_documents(
            final_documents,
            embeddings
        )

        st.session_state.embeddings = embeddings
        st.session_state.docs = docs
        st.session_state.final_documents = final_documents
        st.session_state.vectors = vectors

# -------------------------------------------------------
# Create Vector Store Button
# -------------------------------------------------------

if uploaded_files:

    st.success(f"{len(uploaded_files)} PDF(s) uploaded.")

    if st.button("🚀 Create Vector Store"):

        vector_embedding()

        st.success("Vector Store Created Successfully!")

# -------------------------------------------------------
# User Question
# -------------------------------------------------------

user_question = st.text_input(
    "Ask a question from the uploaded PDFs"
)

# -------------------------------------------------------
# RAG Pipeline
# -------------------------------------------------------

if user_question:

    if "vectors" not in st.session_state:

        st.warning("Please upload PDF(s) and create the vector store first.")

        st.stop()

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.time()

    response = retrieval_chain.invoke(
        {
            "input": user_question
        }
    )

    end = time.time()

    st.subheader("💡 Answer")

    st.write(response["answer"])

    st.success(
        f"Response Time: {end-start:.2f} seconds"
    )

    with st.expander("📚 Retrieved Documents"):

        for i, doc in enumerate(response["context"]):

            st.markdown(f"### Chunk {i+1}")

            st.write(doc.page_content)

            st.divider()