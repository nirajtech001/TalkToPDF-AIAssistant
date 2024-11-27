# TalkToPDF: AI-Powered PDF Chat Assistant  

**TalkToPDF** is a custom-built application that allows users to upload PDF documents and interact with their content using **Retrieval-Augmented Generation (RAG)**. It provides context-aware responses by leveraging chat history, embeddings, and a large language model.

---

## Features  
- Upload multiple PDFs and query their content.  
- Context-aware answers using chat history.  
- Uses Hugging Face embeddings and FAISS for efficient retrieval.  
- Allows switching between developer and user API keys for flexibility.  
- Session management to maintain context across multiple questions.  

---

## Technologies Used  
- **Python 3.8+**  
- **Streamlit**  
- **LangChain**  
- **FAISS**  
- **Hugging Face Embeddings**  
- **Groq API**  
- **PyPDFLoader**  

---

## Building the Application  

### Step 1: Install Dependencies  
Create a virtual environment and install the required packages:  
```bash  
python -m venv venv  
source venv/bin/activate   # For Linux/macOS  
venv\Scripts\activate      # For Windows  

pip install streamlit langchain langchain-community langchain-groq \
            langchain-core langchain-huggingface langchain-text-splitters \
            faiss-cpu pypdf python-dotenv  
```


### Step 2: Create the Application Script (Detailed Breakdown)  

In this step, we'll build the `app.py` script piece by piece, explaining each component and its role in the application.

---

### **2.1: Importing Required Libraries**  
We need libraries for document processing, embeddings, AI model interaction, and the user interface.  

```python  
import streamlit as st  
from langchain.vectorstores import FAISS  
from langchain.chains import create_history_aware_retriever, create_retrieval_chain  
from langchain.chains.combine_documents import create_stuff_documents_chain  
from langchain_community.document_loaders import PyPDFLoader  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain_groq import ChatGroq  
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from dotenv import load_dotenv  
import os  
```

- **Streamlit**: For the web interface.  
- **FAISS**: For vector storage of document embeddings.  
- **LangChain**: For chaining prompts and building history-aware retrieval.  
- **PyPDFLoader**: For loading and extracting text from PDFs.  
- **dotenv**: To load environment variables securely.  

---

### **2.2: Loading Environment Variables**  

We load the environment variables for API keys and embeddings from a `.env` file or Streamlit’s `secrets`.  

```python  
load_dotenv()  
os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"]  
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]  
os.environ["LANGCHAIN_TRACING_V2"] = "true"  
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]  
```

- **HF_TOKEN**: Token for Hugging Face embeddings.  
- **LANGCHAIN_API_KEY**: API key for Langchain tracking and projects.  

---

### **2.3: Setting Up the Streamlit Interface**  

```python  
st.title("TalkToPDF: AI Assistant")  
st.write("Generative AI-based RAG with PDF Uploads and Chat History")  
```

- **st.title**: Adds a title to the app.  
- **st.write**: Adds descriptive text explaining the app’s functionality.  

---

### **2.4: Handling API Key Usage and Question Limits**  

```python  
if 'question_count' not in st.session_state:  
    st.session_state.question_count = 0  

api_key = st.secrets["GROQ_API_KEY"] if st.session_state.question_count < 3 else st.text_input("Enter your Groq API key:", type="password")  
```

- **Session State**: Keeps track of the number of questions asked.  
- **API Key Input**: Switches to user-provided API key after three questions.

---

### **2.5: Initializing the LLM (Large Language Model)**  

```python  
if api_key:  
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")  
```

- **ChatGroq**: Initializes the Groq API with the selected API key and model (`Gemma2-9b-It`).  

---

### **2.6: Managing Session History**  

```python  
if 'store' not in st.session_state:  
    st.session_state.store = {}  

session_id = st.text_input("Session ID", value="default_session")  
```

- **Session Store**: Stores chat history for different sessions using a unique session ID.  

---

### **2.7: Uploading and Processing PDF Files**  

```python  
uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)  

if uploaded_files:  
    documents = []  
    for uploaded_file in uploaded_files:  
        with open("temp.pdf", "wb") as f:  
            f.write(uploaded_file.getvalue())  
        loader = PyPDFLoader("temp.pdf")  
        docs = loader.load()  
        documents.extend(docs)  
```

- **File Uploader**: Allows users to upload one or more PDF files.  
- **File Handling**: Saves the uploaded file temporarily and loads its content using **PyPDFLoader**.  

---

### **2.8: Splitting and Embedding Documents**  

```python  
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)  
splits = text_splitter.split_documents(documents)  
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)  
retriever = vectorstore.as_retriever()  
```

- **Text Splitter**: Breaks the document into chunks for efficient embedding.  
- **FAISS**: Creates a vector store from the document chunks.  
- **Retriever**: Converts the vector store into a retriever for querying.  

---

### **2.9: Setting Up RAG (Retrieval-Augmented Generation) Chains**  

```python  
contextual_prompt = ChatPromptTemplate.from_messages([("system", "Contextualize questions using chat history."), MessagesPlaceholder("chat_history"), ("human", "{input}")])  
retriever_chain = create_history_aware_retriever(llm, retriever, contextual_prompt)  

qa_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_messages([("system", "{context}"), MessagesPlaceholder("chat_history"), ("human", "{input}")]))  
rag_chain = create_retrieval_chain(retriever_chain, qa_chain)  
```

- **Contextual Prompt**: Helps the model understand and maintain context from previous questions.  
- **Retriever Chain**: Connects the retriever to the language model for history-aware question reformulation.  
- **QA Chain**: Connects document context to the model for generating answers.  
- **RAG Chain**: Combines retriever and QA chains to form the complete pipeline.  

---

### **2.10: Handling User Input and Displaying Responses**  

```python  
user_question = st.text_input("Your question:")  

if user_question:  
    response = rag_chain.invoke({"input": user_question, "chat_history": st.session_state.store[session_id].messages})  
    st.write("Assistant:", response['answer'])  

    st.session_state.question_count += 1  
```

- **Text Input**: Allows users to enter questions.  
- **Invoke RAG Chain**: Queries the model with the input and chat history.  
- **Display Response**: Outputs the assistant’s answer.  

---

### **2.11: Handling Missing API Keys**  

```python  
else:  
    st.warning("Please enter a Groq API key.")  
```

- **Warning**: Alerts the user if an API key is missing or invalid.  

---

## Step 3: Configure Secrets  
Create a file called `secrets.toml` in the `.streamlit/` directory:  
```toml  
[secrets]  
HF_TOKEN = "your_huggingface_token"  
LANGCHAIN_API_KEY = "your_langchain_key"  
LANGCHAIN_PROJECT = "your_project_name"  
GROQ_API_KEY = "your_groq_key"  
```  

---

## Running the Application  
Run the app with the following command:  
```bash  
streamlit run app.py  
```  

---

## Usage  
- **Upload PDFs**: Use the file uploader in the app.  
- **Ask Questions**: Enter your query and receive AI-generated answers.  
- **Session Management**: Use the session ID to maintain chat history across interactions.  

---
