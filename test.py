# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.llms import HuggingFaceHub
# from langchain.prompts import PromptTemplate
# import os
# from dotenv import load_dotenv
# load_dotenv()
# import warnings
# warnings.filterwarnings('ignore')

# # Step 1: Load the PDF
# # pdf_loader = PyPDFLoader("budget_speech.pdf")
# # documents = pdf_loader.load()

# def pdf_loader(directory):
#     document = PyPDFDirectoryLoader(directory)
#     document = document.load()
#     return document

# document = pdf_loader('documents')

# # Step 2: Chunking/Splitting the document into manageable pieces
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# texts = text_splitter.split_documents(document)

# # Step 3: Initialize HuggingFace Embeddings
# # Choose a HuggingFace model for embeddings (e.g., 'sentence-transformers/all-mpnet-base-v2')
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# print(embeddings)

# # Step 4: Set up Pinecone for Vector Search
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index_name = "myindex"

# if index_name not in pc.list_indexes().names():
#     pc.create_index(name=index_name, dimension=1536, spec='ServerlessSpec')

# index = pc.Index(index_name)

# # Step 5: Store embeddings in Pinecone
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# # Step 6: Define the HuggingFace model for generating answers
# llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0}, 
#                      huggingfacehub_api_token= "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")

# # Step 7: Set up the Retrieval-Augmented Generation pipeline
# retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Use semantic similarity search
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# # Step 8: User query and response generation
# query = "What is the main topic of the PDF?"
# response = qa_chain.run(query)

# print("Response:", response)



'''

'''

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
import faiss
import os
from dotenv import load_dotenv
import warnings

load_dotenv()
warnings.filterwarnings('ignore')

# Step 1: Load the PDF
def pdf_loader(directory):
    document = PyPDFDirectoryLoader(directory)
    document = document.load()
    return document

document = pdf_loader('documents')

# Step 2: Chunking/Splitting the document into manageable pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(document)

# Step 3: Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Step 4: Use FAISS for Vector Search
# Convert the document texts into vectors using embeddings
text_vectors = embeddings.embed_documents([text.page_content for text in texts])

# Initialize FAISS index
dimension = 1536  # Embedding dimension from the model
index = faiss.IndexFlatL2(dimension)

# Add vectors to the FAISS index
index.add(text_vectors)

# Step 5: Store embeddings in FAISS using LangChain's FAISS wrapper
vector_store = FAISS(embedding=embeddings, index=index, documents=texts)

# Step 6: Define the HuggingFace model for generating answers
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0}, 
                     huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"))

# Step 7: Set up the Retrieval-Augmented Generation pipeline
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Use semantic similarity search
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 8: User query and response generation
query = "What is the main topic of the PDF?"
response = qa_chain.run(query)

print("Response:", response)
