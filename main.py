import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from pypdf import PdfReader

import chromadb

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings


import streamlit as st

# Load the API keys from the .env file
load_dotenv()


def process_documents(doc_path):

    # Read all the documents from the specified path and chunk them
    #--------------------------------------------------------------

    documents = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)

    for idx, file in enumerate(os.listdir(doc_path)):
        
        # if idx > 2:
        #     break

        try:
            
            pdf_path = os.path.join(doc_path, file)
            loader = PyPDFLoader(pdf_path)
            data = loader.load_and_split(text_splitter=splitter)
            documents.extend(data)
            
            print(f"Loading PDF {idx + 1}: Pages : {len(data)}   {file} - SUCCESSFUL")
        except:
            print(f"Loading PDF : {file} - FAILED")


    # Generate the embeddings
    #------------------------

    print("\nGenerating embeddings and storing to the vector database.....\n")

    embeddings = OpenAIEmbeddings()
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory='arxiv_vector_store.db'
    )
    
    return vectordb


def create_chat_agent(persist_directory=None):
    
    chat_llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings()
    
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


    retriever = db.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={
            "k": 5,
            "score_threshold": 0.5
            }
        )

    chain = RetrievalQA.from_chain_type(
        llm=chat_llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    
    return chain

   


if __name__ == '__main__':
    
    if os.path.exists('arxiv_vector_store.db'):
        pass
    else:
        print("Loading the PDF files to the vector store")
        vectordb = process_documents(doc_path='llm_papers')

    # Create the search agent
    search_agent = create_chat_agent(persist_directory="arxiv_vector_store.db")

    no_match = ["don't", "do not"]
    
    # query = "What is CEO of Infosys"
    
    # answer = search_agent.invoke(query)
    
    # print(f"\nQuestion : {answer['query']} \nAnswer : {answer['result']}\n") 
    
    # if any([m in answer['result'][:15] for m in no_match]):
    #     pass
    # else:
    #     relevant_sources = []
    #     for source in answer['source_documents']:
    #         ref = source.metadata['source']
    #         ref = ref.split('\\')[1]
    #         relevant_sources.append(ref)
    #     relevant_sources = list(set(relevant_sources))
    
    #     print("You can refer to these Arxiv papers for more information :")
    #     for r in relevant_sources:
    #         print(r)   



    #----------------
    # Streamlit app
    #----------------

    app_name = "Arxiv Research Papers - Search Page"
    st.set_page_config(page_title=app_name)

    st.header(app_name)

    query = st.text_input("Enter your search query :", key="query")

    # If the user has entered some URL
    if query:
            
        with st.spinner('Getting the results...'):
            answer = search_agent.invoke(query)
        
        st.subheader("The answer is :") 
        st.write(answer['result']) 

        if any([m in answer['result'][:15] for m in no_match]):        
            pass
        else:
            reference_sources = []
            for source in answer['source_documents']:
                ref = source.metadata['source']
                ref = ref.split('\\')[1]
                reference_sources.append(ref)
            reference_sources = list(set(reference_sources))
            
            st.subheader("You can refer to these Arxiv papers for more information")
            for r in reference_sources:
                st.write(r)
                     
    
