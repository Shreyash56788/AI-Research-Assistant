import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.title("📚 AI Research Assistant")

# Load PDF safely
try:
    loader = PyPDFLoader("paper.pdf")
    documents = loader.load()
except:
    st.error("paper.pdf file not found. Please add it to the project folder.")
    st.stop()

# Split text into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Check if document is empty
if len(docs) == 0:
    st.error("No readable text found in the PDF.")
    st.stop()

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
db = FAISS.from_documents(docs, embeddings)

# User question input
question = st.text_input("Ask a question about the research paper")

if question:

    results = db.similarity_search(question, k=3)

    if len(results) == 0:
        st.write("No relevant information found.")
    else:
        st.subheader("📖 Answer")

        for i, result in enumerate(results):
            st.write(f"Result {i+1}:")
            st.write(result.page_content)
            st.write("---")