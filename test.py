import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import streamlit as st
import PyPDF2


# Load the API keys from the .env file
#-------------------------------------
load_dotenv()


doc_path = "llm_papers"

def get_pdf_title(pdf_file_path):
    with open(pdf_file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        document_info = pdf_reader.metadata
        print(document_info)
        title = document_info.title
        return title

# Iterate through each Arxiv document in the folder and chunk it
for idx, file in enumerate(os.listdir(doc_path)):
    
    if idx > 2:
        break
        
    pdf_path = os.path.join(doc_path, file)
    title = get_pdf_title(pdf_path)
    print(title)
