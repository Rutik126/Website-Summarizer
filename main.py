import streamlit as st
import requests
from newspaper import Article
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from llama_index.core.prompts import PromptTemplate
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
def initialize_llm(model_type):
    return Groq(model=model_type, api_key=api_key)

# Function to scrape webpage text
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error fetching the article: {e}"

# Summarization function
def summarize_text(llm, text, summary_type):
    prompts = {
        "Long Summary": "Summarize the following webpage in detail:\n{text}",
        "Short Summary": "Summarize the following webpage in 100 words:\n{text}",
        "Creative Summary": "Provide a creative summary of the following webpage:\n{text}",
        "Bullet Point Summary": "Summarize the following webpage in 3 bullet points:\n{text}"
    }
    text = text[:5000]  # Limit input
    formatted_prompt = prompts[summary_type].format(text=text)
    response = llm.complete(formatted_prompt)
    return response.text

# Question Answering function
def ask_question(llm, text, question):
    prompt = f"Based on the following webpage content, answer the question:\n\n{text}\n\nQuestion: {question}"
    response = llm.complete(prompt)
    return response.text

# Streamlit App
st.title("🌐 AI-Powered Webpage Summarizer & Q&A 🤖")

# URL input
url = st.text_input("Enter Website URL:")

# Summary Type Selection
summary_type = st.selectbox("Select Summary Type", ("Long Summary", "Short Summary", "Creative Summary", "Bullet Point Summary"))

# Model Type Selection
model_type = st.selectbox("Select Model Type", ("qwen-2.5-32b", "llama3-70b-8192", "deepseek-r1-distill-qwen-32b"))

# Initialize model
llm = initialize_llm(model_type)

# Fetch & summarize webpage
if st.button("Summarize Webpage"):
    if url:
        with st.spinner("Extracting and summarizing content... ⏳"):
            extracted_text = extract_text_from_url(url)
            if extracted_text.startswith("Error"):
                st.error(extracted_text)
            else:
                summary = summarize_text(llm, extracted_text, summary_type)
                st.subheader(f"📜 {summary_type} using {model_type}")
                st.write(summary)
                st.session_state["webpage_text"] = extracted_text  # Store extracted text for Q&A
    else:
        st.warning("⚠️ Please enter a valid URL.")

# Q&A Section
st.markdown("## ❓ Ask Questions")
question = st.text_input("Ask a question based on the webpage:")

if st.button("Get Answer"):
    if "webpage_text" in st.session_state and question:
        with st.spinner("Generating answer... ⏳"):
            answer = ask_question(llm, st.session_state["webpage_text"], question)
            st.subheader("🤖 Answer:")
            st.write(answer)
    else:
        st.warning("⚠️ Please summarize the webpage first.")

# Footer
st.markdown("---")
st.markdown("🚀 Made by **Rutik Kumbhar**")

