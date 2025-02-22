# from qdrant_client import QdrantClient
# from langchain.vectorstores import Qdrant
# from langchain.embeddings.openai import OpenAIEmbeddings
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # Set API keys
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "false"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# # Initialize Qdrant Client
# QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # Change if using cloud
# client = QdrantClient(url=QDRANT_URL)

# # Initialize embeddings
# embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# # Define collection name
# index_name = "upes_scraped_data"

# # Initialize Qdrant Vector Store
# vectordb = Qdrant(client=client, collection_name=index_name, embeddings=embeddings_model)

# def get_qa_chain():
#     from langchain.chat_models import ChatOpenAI
#     from langchain.chains import RetrievalQA
#     from langchain.prompts import PromptTemplate
#     from langchain.callbacks import StdOutCallbackHandler

#     # Initialize LLM
#     llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

#     # Adjust retriever settings
#     retriever = vectordb.as_retriever(score_threshold=0.8, top_k=3)

#     # Define Prompt
#     prompt_template = """You are UPESBuddy, a virtual assistant created specifically for students of the University of Petroleum and Energy Studies (UPES), Dehradun. Your primary role is to answer questions related to UPES using only the context provided to you, which includes information pertaining to academic programs, admission processes, campus facilities, event schedules, and university policies.

# Please adhere strictly to the following guidelines:

# 1. Respond accurately to user inquiries by referencing only the available context related to UPES. Avoid referencing external information or creating answers based on general knowledge.
# 2. If the context does not contain the necessary information to answer a user's question, do not attempt to infer an answer or provide speculative responses. Instead, kindly inform the user with the following response:
#    "I'm sorry, but I currently do not have the information required to answer your question. I encourage you to consult the official UPES resources or provide additional context."
# 3. Maintain a courteous and professional tone at all times to ensure a positive and respectful interaction with users.
# 4. Read the CONTEXT given here carefully to form an answer for the QUESTION given after it.
# 5. If the query is about a person do mention their contact details along with other details.
# 6. If the query is about a person and context has information about multiple people with the same name, write a short summary with their contact details.
# ----------------------

#     - CONTEXT: 
#     {context}

# --------------------------

#     - QUESTION: 
#     {question}"""

#     PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#     handler = StdOutCallbackHandler()

#     chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         input_key="query",
#         callbacks=[handler],
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT},
#     )

#     return chain

# if __name__ == "__main__":
#     chain = get_qa_chain()
#     print("QA Chain initialized successfully!")



from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant, FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import StdOutCallbackHandler
import faiss
import numpy as np
import os
from dotenv import load_dotenv
import pickle
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# Set API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize Qdrant Client
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
client = QdrantClient(url=QDRANT_URL)

# Initialize Embeddings Model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Define collection name
index_name = "upes_scraped_data"

# Initialize Qdrant Vector Store
vectordb = Qdrant(client=client, collection_name=index_name, embeddings=embeddings_model)

# FAISS Cache Initialization
faiss_index_file = "faiss_cache.index"
cache_store_file = "faiss_cache.pkl"

# Load FAISS index if it exists
try:
    faiss_index = faiss.read_index(faiss_index_file)
    with open(cache_store_file, "rb") as f:
        query_cache = pickle.load(f)  # Dictionary {index: (query, response)}
except Exception as e:
    faiss_index = faiss.IndexFlatL2(1536)  # Adjust dimensions for your embedding model
    query_cache = {}

# Create a thread pool executor for async disk I/O
executor = ThreadPoolExecutor(max_workers=2)

def save_faiss_index_and_cache():
    """Function to save FAISS index and cache to disk."""
    faiss.write_index(faiss_index, faiss_index_file)
    with open(cache_store_file, "wb") as f:
        pickle.dump(query_cache, f)

def add_to_faiss(query, response):
    """Store the query embedding and response in FAISS asynchronously."""
    embedding = np.array(embeddings_model.embed_query(query)).reshape(1, -1)
    faiss_index.add(embedding)
    query_cache[faiss_index.ntotal - 1] = (query, response)
    
    # Offload disk write operations to a background thread
    executor.submit(save_faiss_index_and_cache)

def search_faiss(query):
    """Search FAISS for a similar query."""
    if faiss_index.ntotal == 0:
        return None  # No stored queries yet

    embedding = np.array(embeddings_model.embed_query(query)).reshape(1, -1)
    distances, indices = faiss_index.search(embedding, k=1)

    threshold = 0.001  # Adjust based on similarity needs
    if distances[0][0] < threshold:
        cached_response = query_cache.get(indices[0][0], (None, None))[1]
        return cached_response

    return None  # No good match found

def get_qa_chain():
    """Creates the QA chain with FAISS caching."""
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    retriever = vectordb.as_retriever(score_threshold=0.8, top_k=3)

    # Define Prompt
    prompt_template = """You are UPESBuddy, a virtual assistant for UPES students. Answer questions **only using the provided CONTEXT** on academics, admissions, campus, events, and policies.  

### Guidelines:  
1. **Use Only Context:** No external knowledge or assumptions. If info is missing, suggest checking UPES [website](https://www.upes.ac.in/).  
2. **Be Accurate & Clear:** Responses must be factual and precise. Clarify if needed.  
3. **Professional Tone:** Keep it respectful, concise, and informative.  
4. **Queries About People:**  
   - If found, include **full name, contact (if available), and role**.  
   - If multiple matches exist, summarize all with contact details.  
5. **Answering Approach:**  
   - **Read CONTEXT carefully** before responding.  
   - If info is **partial**, mention that more may be available elsewhere.  
6. **Missing Information:**  
   - If no relevant details exist, respond with: 
    *"I couldn’t find the requested information in the provided context. You may check the official UPES website for the most updated details." Provide UPES website URL:- https://www.upes.ac.in/*  

--------------------------

    - CONTEXT: 
    {context}

--------------------------

    - QUESTION: 
    {question}
    
"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    handler = StdOutCallbackHandler()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        callbacks=[handler],
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain

if __name__ == "__main__":
    chain = get_qa_chain()
    print("QA Chain initialized successfully!")

    while True:
        query = input("Ask a question: ")
        
        # 1️⃣ Check FAISS cache first
        cached_response = search_faiss(query)
        if cached_response:
            print("Cached Response:", cached_response)
            continue

        # 2️⃣ If not in FAISS, fetch from Qdrant & generate response
        response = chain.invoke({"query": query})
        print("Generated Response:", response["result"])

        # 3️⃣ Store the new response in FAISS for future queries asynchronously
        add_to_faiss(query, response["result"])