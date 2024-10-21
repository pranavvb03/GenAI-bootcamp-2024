import streamlit as st
import os
from groq import Groq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# import speech_recognition as sr
# import pyttsx3

# Initialize Groq API client with environment variable

groq_api_key = st.secrets["YOUR_GROQ_API_KEY"]
client = Groq(api_key=groq_api_key)

# Function to interact with the Groq API for question answering
def query_groq_api(question, context):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"Context: {context}. Question: {question}"}
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Voice recognition setup
# recognizer = sr.Recognizer()
# engine = pyttsx3.init()

# Load and process document
def load_and_preprocess_document(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

# Create FAISS vector store
def create_faiss_vector_store(texts):
    embeddings = OpenAIEmbeddings()  # Replace with appropriate embeddings
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Query the document
def query_document(question, vector_store):
    docs = vector_store.similarity_search(question)
    context = " ".join([doc.page_content for doc in docs[:3]])  # Taking top 3 relevant docs
    return context

# Voice input function
# def voice_input():
#     with sr.Microphone() as source:
#         st.write("Listening for your query...")
#         audio = recognizer.listen(source)
#         try:
#             query = recognizer.recognize_google(audio)
#             st.write(f"You said: {query}")
#             return query
#         except sr.UnknownValueError:
#             st.write("Could not understand audio")
#             return None

# # Voice output function
# def voice_output(text):
#     engine.say(text)
#     engine.runAndWait()

# Streamlit Interface
st.title("SmartDocMate: AI-powered Document Assistant")

# File upload
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
if uploaded_file:
    pdf_path = f"{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Processing document...")
    texts = load_and_preprocess_document(pdf_path)
    vector_store = create_faiss_vector_store(texts)
    st.write("Document processed successfully.")

    # Voice query or text query
    query_option = st.selectbox("Select input method:", ("Text", "Voice"))
    
    if query_option == "Text":
        question = st.text_input("Enter your query")
        if st.button("Submit"):
            context = query_document(question, vector_store)
            answer = query_groq_api(question, context)
            st.write(f"Answer: {answer}")
            # voice_output(answer)

    # elif query_option == "Voice":
    #     if st.button("Record Query"):
    #         question = voice_input()
    #         if question:
    #             context = query_document(question, vector_store)
    #             answer = query_groq_api(question, context)
    #             st.write(f"Answer: {answer}")
    #             voice_output(answer)
