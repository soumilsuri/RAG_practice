import streamlit as st
import os
from chat_llama3 import chat_with_llama3
from chat_chemical import chat_with_chemical
from chat_website import chat_with_website

st.title("Chat with AI")

# Initialize session state variables if they don't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "urls" not in st.session_state:
    st.session_state.urls = []
if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = []
if "pdf_vectors" not in st.session_state:
    st.session_state.pdf_vectors = None
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""

# Sidebar with buttons for different interfaces
st.sidebar.title("Select Interface")

st.sidebar.subheader("Enter API Keys")
st.session_state.groq_api_key = st.sidebar.text_input("GROQ API Key", type="password")
st.session_state.google_api_key = st.sidebar.text_input("Google API Key", type="password")

interface_option = st.sidebar.radio(
    "Choose an interface:",
    ("Chat with llama3", "Chat with Chemical", "Chat with Website")
)

# Main content changes based on selected interface
if interface_option == "Chat with llama3":
    chat_with_llama3(st.session_state.groq_api_key)
elif interface_option == "Chat with Chemical":
    chat_with_chemical(st.session_state.groq_api_key)
elif interface_option == "Chat with Website":
    chat_with_website(st.session_state.groq_api_key, st.session_state.google_api_key)
