import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import cv2
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import time

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

def chat_with_pdf(groq_api_key, google_api_key):
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
