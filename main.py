import streamlit as st
import trafilatura
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Default LLM model
DEFAULT_MODEL = "llama3-70b-8192"

# Initialize LLM
def initialize_llm():
    return Groq(model=DEFAULT_MODEL, api_key=api_key)

# Function to scrape webpage content
def extract_text_from_url(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        return trafilatura.extract(downloaded)
    return "Error extracting content from the URL."

# Summarization function
def summarize_text(llm, text):
    prompt = f"Summarize the following webpage content in a clear and concise manner:\n\n{text}"
    response = llm.complete(prompt)
    return response.text

# Question Answering function
def ask_question(llm, text, question):
    prompt = f"Based on the following webpage content, answer the question:\n\n{text}\n\nQuestion: {question}"
    response = llm.complete(prompt)
    return response.text

# Streamlit UI
st.title("🌐 AI Webpage Summarizer & Q&A 🤖")

# URL Input
url = st.text_input("Enter Website URL:")

# Initialize model
llm = initialize_llm()

# Summarize Button
if st.button("Summarize Webpage"):
    if url:
        with st.spinner("Extracting and summarizing content... ⏳"):
            extracted_text = extract_text_from_url(url)
            if extracted_text.startswith("Error"):
                st.error(extracted_text)
            else:
                summary = summarize_text(llm, extracted_text)
                st.subheader(f"📜 Webpage Summary")
                st.write(summary)
                st.session_state["webpage_text"] = extracted_text  # Store text for Q&A
    else:
        st.warning("⚠️ Please enter a valid URL.")

# Q&A Section
st.markdown("## ❓ Ask a Question")
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
