import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local("faiss_index")
    return vector_store
    
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    return vector_store
    
def get_conversational_chain():
    prompt_template = """
    Answer the question in as detailed manner as possible from the provided context, make sure to provide all the details, if the answer is not in the provided
    context then just say, "I do not have an answer to your question.", do not provide the wrong answer\n\n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
"""
    model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash-latest",temperature = 0.3)

    prompt = PromptTemplate(template= prompt_template,input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type = "stuff",prompt = prompt)
    return chain

def user_input(user_question,vector_store):
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    return response["output_text"]

def main():
    st.set_page_config(page_title="Chat PDF", page_icon=":file_pdf:")  # Set title and icon

    # Colored header with center alignment
    st.markdown("<h1 style='color: #3498db; text-align: center;'>A complete end-to-end PDF RAG chat application using Gemini</h1>",unsafe_allow_html = True)

    user_question = st.text_input("Ask a Question from the PDF Files")  # Plain text input without styling

    with st.sidebar:
        # Green colored sidebar title
        st.markdown("<h3 style='color: #2ecc71;'>Menu:</h3>", unsafe_allow_html=True)

        pdf_docs = st.file_uploader("Please upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        # Styled button using markdown and HTML
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    create_vector_store(text_chunks)
                st.success("Processing completed successfully!")
            else:
                st.error("Please upload PDF files before submitting.")

    if user_question:
        try:
            vector_store = load_vector_store()
            response = user_input(user_question, vector_store)
            st.write("Reply: ", response)
        except ValueError as e:
            st.error(f"Error loading FAISS index: {e}. Please process the PDF files first.")
        
if __name__ == "__main__":
    main()
