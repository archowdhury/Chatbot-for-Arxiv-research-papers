import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document

import chromadb

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings


import streamlit as st



load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Load the PDF files to the vector store
# --------------------------------------

def load_documents(doc_path=None):

    # Read all the documents from the specified path
    #-----------------------------------------------

    documents = []

    for idx, file in enumerate(os.listdir(doc_path)):
        
        if idx > 70:
            break

        try:
            
            pdf_path = os.path.join(doc_path, file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            documents.extend(pages)
            
            print(f"Loading PDF {idx + 1}: Pages : {len(pages)}   {file} - SUCCESSFUL")
        except:
            print(f"Loading PDF : {file} - FAILED")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)


    # Load the documents to the vector store
    #---------------------------------------

    client = chromadb.Client()
    
    embeddings = OpenAIEmbeddings()
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory='arxiv_vector_store.db'
    )
    
    return vectordb


# Create the agent chain
#-----------------------

def create_chat_agent():
    
    model_name = 'gpt-3.5-turbo'
    model = ChatOpenAI(model_name=model_name)

    chain = load_qa_chain(model, chain_type='stuff')
    
    return chain


def create_googlet_agent():
    
    model_name = 'gemini-pro'
    model = ChatOpenAI(model_name=model_name)

    chain = load_qa_chain(model, chain_type='stuff')
    
    return chain



# Get the answer to the query
#----------------------------
def get_answer(query, vectordb, chain):
    
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    
    return answer


# Load all the documents

vectordb = load_documents(doc_path='llm_papers')

# if os.path.exists('arxiv_vector_store.db'):
#     print("PDF documents already loaded")
# else:
#     print("Loading the PDF files to the vector store")
#     vectordb = load_documents(doc_path='llm_papers')


    


# # Create the Streamlit chat app
# #------------------------------

# st.set_page_config(page_title="Arxiv research paper searcher")
# st.header("Query Arxiv papers")

# # Load the documents and initialize the chat agent
# vectordb = load_documents(doc_path='llm_papers')
# chain = create_chat_agent()

# query = st.text_input('Enter search query :')

# if query:
#     answer = get_answer(query, vectordb, chain)
#     st.write(answer)
    
