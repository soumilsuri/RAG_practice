from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.url_selenium import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import time

def chat_with_website(groq_api_key, google_api_key):
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
                loaders.append(SeleniumURLLoader(url, browser="chrome"))

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
