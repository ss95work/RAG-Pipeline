''' extracting content from website '''

import requests
from bs4 import BeautifulSoup
import nltk
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings
warnings.filterwarnings('ignore')


''' Step 1: Content Parsing '''

def fetch_website_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text content (assuming it's within <p> tags for simplicity)
    paragraphs = soup.find_all('p')
    content = "\n".join([p.get_text() for p in paragraphs])
    return content

# Example usage
url = 'https://en.wikipedia.org/wiki/Natural_language_processing'  # Replace with any website
website_content = fetch_website_content(url)
# print(website_content[:500])  # Preview first 500 characters


''' step 2: chunking '''

def chunk_text(text, max_chunk_size=500):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = []

    for sentence in sentences:
        if len(" ".join(chunk)) + len(sentence) <= max_chunk_size:
            chunk.append(sentence)
        else:
            chunks.append(" ".join(chunk))
            chunk = [sentence]

    if chunk:  # Append any remaining sentences
        chunks.append(" ".join(chunk))

    return chunks

# Example usage
chunks = chunk_text(website_content)
# print(f"Number of chunks: {len(chunks)}")


''' Step 3: Embedding '''

# Load pre-trained sentence embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Convert chunks to embeddings
chunk_embeddings = model.encode(chunks)

# Create a FAISS index for fast retrieval
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])  # L2 distance index
index.add(np.array(chunk_embeddings))

# Example usage: Retrieve the most similar chunk for a query
def retrieve_chunk(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [(chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

# Test retrieval with a query
query = "What is natural language processing?"
retrieved_chunks = retrieve_chunk(query)
# for chunk, distance in retrieved_chunks:
#     print(f"Chunk: {chunk}\nDistance: {distance}\n")


''' Step 4: Response Generation '''

from transformers import pipeline

# Load pre-trained text generation model (like GPT-2 or T5)
generator = pipeline('text-generation', model='t5-small', max_length=512)

def generate_answer(query, context):
    # Combine query and context
    input_text = f"Question: {query} Context: {context}"
    
    # Generate response
    response = generator(input_text, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

# Example usage: Generating answer based on retrieved chunks
context = " ".join([chunk for chunk, _ in retrieved_chunks])
answer = generate_answer(query, context)
# print(f"Answer: {answer}")

def qa_bot(query, url, top_k=3):
    # Step 1: Fetch website content
    website_content = fetch_website_content(url)
    
    # Step 2: Chunk the content
    chunks = chunk_text(website_content)
    
    # Step 3: Embed and index the chunks
    chunk_embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(np.array(chunk_embeddings))
    
    # Step 4: Retrieve the most relevant chunks
    retrieved_chunks = retrieve_chunk(query, top_k=top_k)
    context = " ".join([chunk for chunk, _ in retrieved_chunks])
    
    # Step 5: Generate and return the answer
    return generate_answer(query, context)

# Example usage
url = 'https://en.wikipedia.org/wiki/Natural_language_processing'  # Use the URL you want
query = "What is the application of NLP?"
answer = qa_bot(query, url)
print('Query: ',query)
print('Reponse: ',answer)