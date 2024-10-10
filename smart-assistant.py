import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
import time

# Initialize model and vector store
model = SentenceTransformer('all-MiniLM-L6-v2')

class DocumentStore:
    def __init__(self, vector_size=384):
        self.vector_size = vector_size
        self.index = faiss.IndexFlatL2(vector_size)
        self.documents = []

    def add_document(self, doc_text):
        embedding = model.encode([doc_text])
        self.index.add(embedding)
        self.documents.append(doc_text)

    def search(self, query, top_k=5):
        query_embedding = model.encode([query])
        D, I = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in I[0]]

huggingface_api_key = st.secrets["YOUR_API_KEY"]
# Load the language model
llm = HuggingFaceHub(repo_id="google/flan-t5-small", "huggingfacehub_api_token": huggingface_api_key)

# Define the prompt for the assistant
template = """
You are a smart assistant that helps with answering questions and document retrieval. 
If a user asks a question related to their uploaded documents, first search the documents for an answer. 
Otherwise, try to provide an answer based on your knowledge.

Question: {question}

Answer:
"""

prompt = PromptTemplate(input_variables=["question"], template=template)

class SmartAssistant:
    def __init__(self, document_store):
        self.chain = LLMChain(llm=llm, prompt=prompt)
        self.document_store = document_store

    def ask_question(self, question):
        # First search in the uploaded documents
        results = self.document_store.search(question)
        if results:
            return f"I found something related to your question in your documents: {results[0]}"
        else:
            # If no document matches, ask the LLM
            return self.chain.run(question)

# Tool classes
class Tool:
    def __init__(self, name, action):
        self.name = name
        self.action = action

# Reminder tool
def reminder_tool(time_in_seconds, message):
    time.sleep(time_in_seconds)
    return f"Reminder: {message}"

# Summarization tool
def summarize_text_tool(text):
    prompt = f"Summarize this: {text}"
    return llm(prompt)

# Initialize document store and assistant
doc_store = DocumentStore()
assistant = SmartAssistant(doc_store)

# Streamlit app
st.title("Smart Assistant with LangChain and VectorStore")

# Document Upload Section
st.header("Upload Your Documents")
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    # Process uploaded file
    document_text = uploaded_file.read().decode("utf-8")
    doc_store.add_document(document_text)
    st.success("Document uploaded and indexed successfully!")

# Question Answering Section
st.header("Ask a Question")
user_question = st.text_input("Enter your question")

if st.button("Ask"):
    if user_question:
        response = assistant.ask_question(user_question)
        st.write(response)

# Tool Section
st.header("Use a Tool")

# Reminder Tool
if st.checkbox("Set a Reminder"):
    reminder_time = st.number_input("Set reminder time (in seconds)", min_value=1, step=1)
    reminder_message = st.text_input("Enter reminder message")
    
    if st.button("Start Reminder"):
        reminder_response = reminder_tool(reminder_time, reminder_message)
        st.success(reminder_response)

# Summarization Tool
if st.checkbox("Summarize Text"):
    text_to_summarize = st.text_area("Enter text to summarize")
    
    if st.button("Summarize"):
        if text_to_summarize:
            summary = summarize_text_tool(text_to_summarize)
            st.write(summary)
