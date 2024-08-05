import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.document_loaders.url_selenium import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Define prompts for the required fields
properties_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a very helpful assistant which provides answers with the source(url) of that answer. You provide all the answers only looking at credible and reliable sources."),
        ("user", "Provide the CAS No, IUPAC name, chemical formula, and HS code for {chemical_name} with the source(url).")
    ]
)

product_information_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a very helpful assistant which provides answers with the source(url) of that answer. You provide all the answers only looking at credible and reliable sources."),
        ("user", "Provide the product description, feedstock, synonyms, purity, storage container, packing type, and origin country for {chemical_name} with the source(url).")
    ]
)

custom_duty_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a very helpful assistant which provides answers with the source(url) of that answer. You provide all the answers only looking at credible and reliable sources."),
        ("user", "Based on the content embeddings: {content_embeddings}, provide the custom duty rate for the HS code {hs_code}.")
    ]
)

def scrape_and_embed_content(url):
    try:
        # Initialize SeleniumURLLoader with the URL
        loader = SeleniumURLLoader(urls=[url])
        documents = loader.load()

        # Check if documents are loaded
        if not documents:
            st.error("No documents found at the provided URL.")
            return []

        # Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # Generate embeddings for the text chunks
        embeddings_generator = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Ensure this is correctly configured
        embeddings = []
        for text in texts:
            try:
                embedding = embeddings_generator.embed_documents([text.page_content])  # Extract the text content
                embeddings.append(embedding)
            except Exception as e:
                st.error(f"Error generating embedding for text chunk: {e}")

        return embeddings

    except Exception as e:
        st.error(f"Error scraping and embedding content: {e}")
        return []

def chat_with_chemical(groq_api_key):
    st.sidebar.title("Chemical Information")
    input_chemical = st.sidebar.text_input("Enter the name of a chemical:")

    with st.expander("Get Properties"):
        if st.button("Get Properties"):
            try:
                property_chain = properties_prompt | ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", max_tokens=1000) | StrOutputParser()
                property_response = property_chain.invoke({"chemical_name": input_chemical})
                st.write(property_response)
            except Exception as e:
                st.error(f"Error fetching properties: {e}")

    with st.expander("Get Product Information"):
        if st.button("Get Product Information"):
            try:
                product_info_chain = product_information_prompt | ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", max_tokens=1000) | StrOutputParser()
                product_info_response = product_info_chain.invoke({"chemical_name": input_chemical})
                st.write(product_info_response)
            except Exception as e:
                st.error(f"Error fetching product information: {e}")

    with st.expander("Trade Data"):
        hs_code = st.text_input("Enter HS Code:", key="hs_code")  # Ensure a unique key for text input
        if st.button("Custom Duty Rate"):
            if hs_code:
                url = f"https://www.cybex.in/indian-custom-duty/hs-{hs_code}.aspx"
                try:
                    content_embeddings = scrape_and_embed_content(url)
                    if content_embeddings:
                        # Split content embeddings to stay within token limit
                        trade_data_responses = []
                        for embeddings_chunk in content_embeddings:
                            trade_data_chain = custom_duty_prompt | ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", max_tokens=1000) | StrOutputParser()
                            trade_data_response = trade_data_chain.invoke({"content_embeddings": embeddings_chunk, "hs_code": hs_code})
                            trade_data_responses.append(trade_data_response)

                        # Combine responses
                        st.write(" ".join(trade_data_responses))
                    else:
                        st.error("No embeddings generated from content.")
                except Exception as e:
                    st.error(f"Error fetching custom duty rate: {e}")
            else:
                st.error("Please enter a valid HS Code.")

# Run the Streamlit app
if __name__ == "__main__":
    groq_api_key = st.secrets["GROQ_API_KEY"]
    chat_with_chemical(groq_api_key)
