# Project Overview

This project demonstrates how to take create a search agent to query across multiple PDF files and return the answers.

ChromaDB is the vector database used to store all the document embeddings.

Streamlit has ben used to create the UI.

### Project Setup
1) Clone the repository
2) Create a virtual environment
3) If using VSCode
4) 		You can use Ctrl + Shift + P to create the environment. Specify the requirements.txt file and it will also install all the required packages
   		You can also use the command line to create the environment. Type python3 -m venv evn_name. Activate the environment. Then run pip install -r requirements.txt to install all the required packages
   		You can use either conda or a native virtual environment. I used the native environment.

### Running the code
1) Open the terminal
2) Type streamlit run main.py
3) The first time it would take a while as all the PDF documents would be read, chunked up, embedded and stored in the vector database. From next time onwards it would run much faster.
4) Once the UI search box shows up type your query and get the answers.

### Some things to mention
1) The code fetches the best 5 Arxiv papers to get the answer to the query. It shows other relevant documents for further reference.
2) If the answer is not found it would say that it could not get the answer.
3) I had tried to display the PDF titles along with the document names, but unfortunately these PDF metadata don't have the title mentioned. There could be other options we could explore, but kept the project simple as of now

