import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import streamlit as st


# Load the API keys from the .env file
#-------------------------------------
load_dotenv()


# Function to read all the documents from the specified path,chunk them, and load to the vector store
#----------------------------------------------------------------------------------------------------
def process_documents(doc_path):

    documents = []

    # Define the splitter used for chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)

    # Iterate through each Arxiv document in the folder and chunk it
    for idx, file in enumerate(os.listdir(doc_path)):
        
        try:
            
            pdf_path = os.path.join(doc_path, file)
            loader = PyPDFLoader(pdf_path)
            data = loader.load_and_split(text_splitter=splitter)
            documents.extend(data)
            
            print(f"Loading PDF {idx + 1}: Pages : {len(data)}   {file} - SUCCESSFUL")
        except:
            print(f"Loading PDF : {file} - FAILED")


    # Create embeddings for each document and add it to the vector store
    print("\nGenerating embeddings and storing to the vector database.....\n")

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory='arxiv_vector_store.db'
    )
    
    return vectordb


# Function to create the chat agent. This would be used to retrieve the relevant embeddings  
# from the vector store and then call the LLM to getthe answer to the user query
#------------------------------------------------------------------------------------------
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

   
   
   

#======================================================================
# START OF THE MAIN PROGRAM
#======================================================================

if __name__ == '__main__':
    
 
    #--------------------------
    # Create the Streamlit app
    #--------------------------

    # Start the Streamlit app
    app_name = "Search Page - Arxiv Research Papers"
    st.set_page_config(page_title=app_name)
    st.header(app_name)

    
    # Load all the Arxiv research papers to the vector store (no need to do this if the vector store already exists)
    if os.path.exists('arxiv_vector_store.db'):
        pass
    else:
        print("Loading the PDF files to the vector store")
        with st.spinner('Loading all the Arxiv papers to the vector store. This is a one-time activity and might take a while. \nGo and enjoy a cup of tea and come back :)'):
            vectordb = process_documents(doc_path='llm_papers')


    # Create the search agent
    search_agent = create_chat_agent(persist_directory="arxiv_vector_store.db")


    # Get the user query
    query = st.text_input("Enter your search query :", key="query")


    # Get the search results
    if query:

        # Display the results of the search            
        with st.spinner('Getting the results...'):
            answer = search_agent.invoke(query)
        
        st.subheader("The answer is :") 
        st.write(answer['result']) 


        # Show relevant documenets the user could refer to for more information
        # If the search results did not return any results then suppress this output
        no_match = ["don't", "do not"]
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
                     
    
