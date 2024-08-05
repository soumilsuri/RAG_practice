from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.url_selenium import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import cv2

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']
google_api_key = os.environ['GOOGLE_API_KEY']

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image_for_tesseract(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    return gray

def extract_text_from_pdf(pdf_document):
    documents = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = preprocess_image_for_tesseract(np.array(img))
        text = pytesseract.image_to_string(img)
        documents.append(Document(page_content=text))
    return documents


# Prompt templates
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a very helpful assistant which provides answers with the source of that answer. You provide all the answers only looking at credible and reliable sources."),
        ("user", "Question:{question}")
    ]
)

property_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a very helpful assistant which provides answers with the source of that answer. You provide all the answers only looking at credible and reliable sources."),
        ("user", "Provide the following properties for {chemical_name} with their sources: CAS No, IUPAC name, Chemical formula, HS code.")
    ]
)

product_info_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a very helpful assistant which provides answers with the source of that answer. You provide all the answers only looking at credible and reliable sources."),
        ("user", "Provide the following product information for {chemical_name} with their sources: Product Description, Feedstock, Synonyms, Purity(for commercial purpose), Storage Container, Packing Type, Origin(Country).")
    ]
)

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

# Sidebar with buttons for different interfaces
st.sidebar.title("Select Interface")

interface_option = st.sidebar.radio(
    "Choose an interface:",
    ("Chat with llama3", "Chat with Chemical", "Chat with Website", "Chat with PDF")
)

# Main content changes based on selected interface
if interface_option == "Chat with llama3":

    input_txt = st.text_input("Message llama3")

    if input_txt:
        chat_chain = chat_prompt | ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192") | StrOutputParser()
        chat_response = chat_chain.invoke({"question": input_txt})
        st.write(chat_response)

elif interface_option == "Chat with Chemical":
    st.title("Chat with Chemical")

    with st.sidebar.expander("Select a Category:"):
        category = st.selectbox("Choose a category:", ["Properties", "Product Information", "Market Data", "Trade Data", "Regulatory Information", "Documentation", "Buyer Information", "Producer Information"])

        if category == "Properties":
            st.write("""
            - CAS No
            - IUPAC Name
            - Chemical Formula
            - HS Code
            """)
        elif category == "Product Information":
            st.write("""
            - Product Description
            - Feedstock
            - Synonyms
            - Purity
            - Storage Container
            - Packing Type
            - Origin
            """)
        elif category == "Market Data":
            st.write("""
            - Industy Use
            - Industry Use Demand with Volume
            - Price Trend
            - Current Price in USD
            - All Time High Price in USD
            - All Time Low Price in USD
            - Average Price Benchmark in USD
            - Product Market Overview
            - Total Market Size (Volume) in India
            - Future Outlook
            """)
        elif category == "Trade Data":
            st.write("""
            - Custom Duty Rate
            - Anti Dumping Rate per MT
            - FTA Benefits
            - Major Importers in India
            - Major Importers Volume
            - Major Exporters in India
            - Major Players in Global Production
            - Major Players in Global Production Volume
            """)
        elif category == "Regulatory Information":
            st.write("""
            - Regulatory / Compliance Requirement
            """)
        elif category == "Documentation":
            st.write("""
            - COA (Certificate of Analysis)
            - MSDS (Material Safety Data Sheet)
            """)
        elif category == "Buyer Information":
            st.write("""
            - Buyer's in India
            - Buyer's Yearly Volume
            """)
        elif category == "Producer Information":
            st.write("""
            - Top Producers in India
            - Top Producers Volume
            """)

    input_chemical = st.text_input("Enter the name of a chemical:")

    if st.button("Get Properties"):
        property_chain = property_prompt | ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192") | StrOutputParser()
        property_response = property_chain.invoke({"chemical_name": input_chemical})
        st.write(property_response)

    if st.button("Get Product Information"):
        product_info_chain = product_info_prompt | ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192") | StrOutputParser()
        product_info_response = product_info_chain.invoke({"chemical_name": input_chemical})
        st.write(product_info_response)

elif interface_option == "Chat with Website":
    st.title("Chat with Website Content")
    st.write("Enter the URLs of the websites you want to chat with")

    if "vector" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def add_url():
        st.session_state.urls.append("")

    if st.button("Add URL"):
        add_url()

    for i, url in enumerate(st.session_state.urls):
        st.session_state.urls[i] = st.text_input(f"Website URL {i+1}", value=st.session_state.urls[i])

    if st.button("Submit"):
        if "loader" not in st.session_state:
            loaders = []
            for url in st.session_state.urls:
                loaders.append(WebBaseLoader(url))
                loaders.append(SeleniumURLLoader(urls=[url], browser="chrome"))

            all_docs = []
            for loader in loaders:
                all_docs.extend(loader.load())

            st.session_state.text_splitter = RecursiveCharacterTextSplitter()
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    if "vectors" in st.session_state:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
        prompt_template = ChatPromptTemplate.from_template(
            """
            Answer every question only from the provided context and give the source for it.
            Please provide the most accurate response based on the question.
            Do not provide any additional information that is not asked for.
            <context>
            {context}
            </context>
            Question: {input}
            """
        )

        chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, chain)

        prompt = st.text_input("Input your prompt here")

        if prompt:
            start = time.process_time()
            response = retrieval_chain.invoke({"input": prompt})
            st.session_state.chat_history.append({"prompt": prompt, "response": response["answer"]})

            st.write("Response time:", time.process_time() - start)

    for chat in st.session_state.chat_history:
        st.write(f"**Prompt:** {chat['prompt']}")
        st.write(f"**Response:** {chat['response']}")
        st.write("---")

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("--------------------------------")

elif interface_option == "Chat with PDF":
    st.title("Chat with PDF Content")

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            documents = extract_text_from_pdf(pdf_document)
            st.session_state.pdf_texts.extend(documents)  # Extend instead of append

        if st.session_state.pdf_texts:
            text_splitter = RecursiveCharacterTextSplitter()
            split_documents = text_splitter.split_documents(st.session_state.pdf_texts)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.pdf_vectors = FAISS.from_documents(split_documents, embeddings)

    if st.session_state.pdf_vectors:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
        prompt_template = ChatPromptTemplate.from_template(
            """
            Answer every question only from the provided context and give the source for it.
            Please provide the most accurate response based on the question.
            Do not provide any additional information that is not asked for.
            <context>
            {context}
            </context>
            Question: {input}
            """
        )

        chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = st.session_state.pdf_vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, chain)

        prompt = st.text_input("Ask a question about the PDFs")

        if prompt:
            start = time.process_time()
            response = retrieval_chain.invoke({"input": prompt})
            st.write("Response time:", time.process_time() - start)
            st.write(response["answer"])

    for chat in st.session_state.chat_history:
        st.write(f"**Prompt:** {chat['prompt']}")
        st.write(f"**Response:** {chat['response']}")
        st.write("---")


