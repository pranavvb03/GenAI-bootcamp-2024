import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlparse
from groq import Groq

# Set up Groq API client
groq_api_key = st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=groq_api_key)
# groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Load a smaller language model compatible with CPU
# You can replace with a smaller OpenAI GPT-2 model for faster performance
model_name = "distilgpt2"  # Smaller model for CPU usage
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prompt template for generating SQL queries with dynamic table definitions
base_prompt = """### Task
Generate SQL queries based on user questions and dynamically generated table definitions.

### Instructions
1. Dynamically generate `CREATE TABLE` statements based on context provided by the user.
2. Use these table definitions to formulate accurate SQL queries.
3. If the question cannot be answered, respond with "I do not know."

### Contextual Table Definitions
{table_definitions}

### Question
[QUESTION] {question} [/QUESTION]

### SQL Answer
[SQL]
"""

# Define functions for dynamic table generation and SQL query generation
def generate_table_definitions(context):
    """Use Groq API to dynamically generate table definitions based on context."""
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": f"Create SQL tables based on the following context: {context}"}],
        model="llama3-8b-8192",  # Adjust model as necessary
    )
    table_definitions = response.choices[0].message.content
    return table_definitions

def generate_sql_query(question, table_definitions):
    """Generate SQL query based on user's question and generated table definitions."""
    prompt = base_prompt.format(question=question, table_definitions=table_definitions)
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate SQL query on CPU
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        num_beams=2,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    outputs = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    sql_query = sqlparse.format(outputs.split("[SQL]")[-1], reindent=True)
    
    return sql_query

# Streamlit app interface
st.title("Text2SQL Chatbot (CPU-Compatible)")
st.write("This chatbot generates SQL queries based on your questions and context.")

# Display a chat-like interface
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Input form for user message
with st.form(key="user_input_form"):
    user_input = st.text_input("You:", "")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    # Append user message to conversation
    st.session_state.conversation.append(("user", user_input))
    
    # Check if user is greeting or asking a question
    if "hello" in user_input.lower() or "hi" in user_input.lower():
        bot_response = "Hello! I'm here to help you generate SQL queries. Please provide a context or ask a question."
    else:
        # Generate dynamic table definitions based on context
        table_definitions = generate_table_definitions(user_input)
        
        # Generate SQL query based on question and table definitions
        bot_response = generate_sql_query(user_input, table_definitions)

    # Append bot response to conversation
    st.session_state.conversation.append(("bot", bot_response))

# Display the conversation
for sender, message in st.session_state.conversation:
    if sender == "user":
        st.write(f"**You:** {message}")
    else:
        st.write(f"**Bot:** {message}")
