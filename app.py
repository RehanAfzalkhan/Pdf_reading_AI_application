import os
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import streamlit as st
from tempfile import NamedTemporaryFile

# Initialize Groq client
# client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
client = Groq(api_key=os.getenv("Groq_api_key"))

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file_path):
    pdf_reader = PdfReader(pdf_file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# Function to create embeddings and store them in FAISS
def create_embeddings_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_db

# Function to query the vector database and interact with Groq
def query_vector_db(query, vector_db):
    # Retrieve relevant documents
    docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Interact with Groq API
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"Use the following context:\n{context}"},
            {"role": "user", "content": query},
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Streamlit app
st.title("Pdf reading AI Application")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    # st.write("PDF Text Extracted Successfully!")

    # Chunk text
    chunks = chunk_text(text)
    # st.write("Text Chunked Successfully!")

    # Generate embeddings and store in FAISS
    vector_db = create_embeddings_and_store(chunks)
    # st.write("Embeddings Generated and Stored Successfully!")

    # Interactive chat section
    st.write("### Interactive Chat Section")

    # State management for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User query input
    user_query = st.text_input("Enter your query:", key="user_query")

    if st.button("Submit Query"):
        if user_query:
            # Get response from the model
            response = query_vector_db(user_query, vector_db)

            # Append the query and response to the chat history
            st.session_state.chat_history.append({"query": user_query, "response": response})

    # Display chat history
    for chat in st.session_state.chat_history:
        st.write(f"**User Query:** {chat['query']}")
        st.write(f"**Response:** {chat['response']}")
        st.write("---")
