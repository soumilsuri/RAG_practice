from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a very helpful assistant which provides answers with the source of that answer. You provide all the answers only looking at credible and reliable sources."),
        ("user", "Question:{question}")
    ]
)

def chat_with_llama3(groq_api_key):
    input_txt = st.text_input("Message llama3")
    if input_txt:
        chat_chain = chat_prompt | ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192") | StrOutputParser()
        chat_response = chat_chain.invoke({"question": input_txt})
        st.write(chat_response)
