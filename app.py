import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Bedrock
import boto3


# AWS Bedrock setup
bedrock = boto3.client(service_name='bedrock-runtime')

# Set up the Claude LLM using Bedrock
def get_claude_llm():
    llm = Bedrock(
        model_id="anthropic.claude-v2:1",
        client=bedrock,
        model_kwargs={
            'max_tokens_to_sample': 512,
            'temperature': 0.5,
            'top_p': 1,
            'top_k': 250
        }
    )
    return llm

# Data ingestion (PDF processing)
def data_ingestion():
    pdf_folder_path = "data"
    docs = []
    for file_name in os.listdir(pdf_folder_path):
        if file_name.endswith(".pdf"):
            pdf_loader = PyPDFLoader(os.path.join(pdf_folder_path, file_name))
            docs.extend(pdf_loader.load())
    return docs

# Create vector store
def get_vector_store(docs):
    bedrock_embeddings = BedrockEmbeddings(client=bedrock)
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss-index")
    return vectorstore_faiss

# Load vector store if available
def load_vector_store():
    try:
        bedrock_embeddings = BedrockEmbeddings(client=bedrock)
        vectorstore_faiss = FAISS.load_local("faiss-index", bedrock_embeddings)
        return vectorstore_faiss
    except ValueError:
        st.warning("Could not load FAISS index. Regenerating...")
        docs = data_ingestion()
        return get_vector_store(docs)

# Get response from the LLM using vector store for retrieval
def get_response_llm(llm, vector_store, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    answer = qa({"query": query})
    return answer['result']

# Streamlit UI
def main():
    st.title("Chat with PDF using AWS Bedrock")

    # Load or regenerate FAISS index
    with st.spinner("Loading vector store..."):
        faiss_index = load_vector_store()
        st.success("Vector store loaded.")

    # Claude LLM setup
    llm = get_claude_llm()

    # User input for question
    user_question = st.text_input("Ask a Question from the PDF files:")
    
    if st.button("Submit"):
        with st.spinner("Generating response..."):
            try:
                response = get_response_llm(llm, faiss_index, user_question)
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
