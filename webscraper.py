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


embeddings = OpenAIEmbeddings()

persist_directory="arxiv_vector_store.db"

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

query = "what is a large language model. Explain in 100 words"

answer = db.similarity_search(query)
print(answer)


