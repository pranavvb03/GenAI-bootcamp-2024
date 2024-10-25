import streamlit as st
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sqlparse
import pdfplumber

# Load the LLM model (SQLCoder)
model_name = "defog/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=True,
)

# Streamlit app interface
st.title("Text2SQL using Defog SQLCoder")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Upload PDF File with table definitions
uploaded_file = st.file_uploader("Upload a PDF with table definitions", type="pdf")

if uploaded_file:
    # Extract and display table schema from PDF
    schema_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Database Schema")
    st.write(schema_text)

    # Question input from the user
    question = st.text_input("Enter your question related to the database")

    if st.button("Generate SQL Query"):
        if question:
            # Dynamically create the prompt using extracted schema
            prompt = f"""
            ### Task
            Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

            ### Instructions
            - If you cannot answer the question with the available database schema, return 'I do not know'
            - Remember that profit is revenue minus cost
            - Remember that revenue is sale_price multiplied by quantity_sold
            - Remember that cost is purchase_price multiplied by quantity_sold
            ### Database Schema
            This query will run on a database whose schema is represented in this string:
            {schema_text}
            ### Answer
            Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
            [SQL]
            """

            # Tokenize and generate SQL query
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            generated_ids = model.generate(
                **inputs,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=400,
                do_sample=False,
                num_beams=2,
            )

            # Decode and format the SQL query
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            sql_query = sqlparse.format(outputs[0].split("[SQL]")[-1], reindent=True)
            
            # Display the generated SQL query
            st.subheader("Generated SQL Query")
            st.code(sql_query)
        else:
            st.warning("Please enter a question.")

