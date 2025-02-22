# import os
# import time
# import logging
# from datetime import datetime
# from fastapi import FastAPI, Request
# from pymongo import MongoClient
# from telegram import Bot, Update
# from telegram.error import TelegramError
# from langchain_redis import RedisSemanticCache
# from langchain_openai import OpenAIEmbeddings
# from langchain.globals import set_llm_cache
# from model import get_qa_chain
# from langchain_community.callbacks import get_openai_callback

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# # Environment variables
# BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# REDIS_URL = os.getenv("REDIS_URL", "redis://redis-stack:6379")


# # Initialize LangChain components
# chain = get_qa_chain()
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# semantic_cache = RedisSemanticCache(redis_url=REDIS_URL, embeddings=embeddings, distance_threshold=0.01)
# set_llm_cache(semantic_cache)

# # Initialize Telegram Bot, MongoDB client, and FastAPI app
# bot = Bot(BOT_TOKEN)
# app = FastAPI()
# client = MongoClient(MONGO_URI)
# db = client.telegram_bot
# users_collection = db.users
# chats_collection = db.chats

# # Set up logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Helper function to calculate latency
# def calculate_latency(start_time, end_time):
#     return round(end_time - start_time, 2)

# @app.post("/webhook")
# async def webhook(request: Request):
#     try:
#         # Parse the incoming update
#         data = await request.json()
#         update = Update.de_json(data, bot)

#         if update.message and update.message.text:
#             start_time = time.perf_counter()

#             # Extract user and message details
#             user = update.message.from_user
#             message = update.message

#             user_id = user.id
#             username = user.username if user.username else "Unknown"
#             first_name = user.first_name if user.first_name else "Unknown"
#             last_name = user.last_name if user.last_name else "Unknown"

#             with get_openai_callback() as cb:
#                 response = chain.invoke({"query": message.text})  # Call the retrieval chain
#                 response_text = response["result"]
#                 source_documents = response["source_documents"]

#             # Log retrieved documents
#             logging.info("Retrieved Documents:")
#             for doc in source_documents:
#                 logging.info(f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}\n")

#             # Send response to the user
#             await bot.send_message(chat_id=message.chat.id, text=response_text)
#             end_time = time.perf_counter()
#             logging.info("Message sent successfully!")

#             # Store user data in MongoDB
#             users_collection.update_one(
#                 {"userid": user_id},
#                 {"$set": {"userid": user_id, "username": username, "first_name": first_name, "last_name": last_name}},
#                 upsert=True
#             )

#             # Store chat data
#             chat_data = {
#                 "message_id": message.message_id,
#                 "user_id": user_id,
#                 "message_date": datetime.utcnow().strftime("%Y-%m-%d"),
#                 "message_time": datetime.utcnow().strftime("%H:%M:%S"),
#                 "message_text": message.text,
#                 "response_text": response_text,
#                 "latency": calculate_latency(start_time, end_time),
#                 "tokens_used": cb.total_tokens,
#                 "retrieved_sources": [doc.metadata.get("source", "Unknown") for doc in source_documents]
#             }
#             chats_collection.insert_one(chat_data)

#     except TelegramError as te:
#         logging.error(f"Telegram error: {te}")
#     except Exception as e:
#         logging.error(f"Error: {e}")

#     return {"status": "ok"}

# # Run the FastAPI server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# import os
# import time
# import logging
# from datetime import datetime
# from fastapi import FastAPI, Request
# from pymongo import MongoClient
# from telegram import Bot, Update
# from telegram.error import TelegramError
# from langchain_openai import OpenAIEmbeddings
# from model import get_qa_chain
# from langchain_community.callbacks import get_openai_callback
# from qdrant_client import QdrantClient
# from langchain.vectorstores import Qdrant

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# # Environment variables
# BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN1")
# MONGO_URI = "mongodb://localhost:27017"
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # Use Cloud URL if needed

# # Initialize Qdrant Client
# client = QdrantClient(url=QDRANT_URL)

# # Initialize LangChain Components
# chain = get_qa_chain()
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# # Use Qdrant for retrieval
# vectordb = Qdrant(client=client, collection_name="upes_scraped_data", embeddings=embeddings)
# retriever = vectordb.as_retriever(score_threshold=0.8, top_k=3)

# # Initialize Telegram Bot, MongoDB client, and FastAPI app
# bot = Bot(BOT_TOKEN)
# app = FastAPI()
# client = MongoClient(MONGO_URI)
# db = client.telegram_bot
# users_collection = db.users
# chats_collection = db.chats

# # Set up logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Helper function to calculate latency
# def calculate_latency(start_time, end_time):
#     return round(end_time - start_time, 2)

# @app.post("/webhook")
# async def webhook(request: Request):
#     try:
#         # Parse the incoming update
#         data = await request.json()
#         update = Update.de_json(data, bot)

#         if update.message and update.message.text:
#             start_time = time.perf_counter()

#             # Extract user and message details
#             user = update.message.from_user
#             message = update.message

#             user_id = user.id
#             username = user.username if user.username else "Unknown"
#             first_name = user.first_name if user.first_name else "Unknown"
#             last_name = user.last_name if user.last_name else "Unknown"

#             with get_openai_callback() as cb:
#                 response = chain.invoke({"query": message.text})  # Call the retrieval chain
#                 response_text = response["result"]
#                 source_documents = response["source_documents"]

#             # Log retrieved documents
#             logging.info("Retrieved Documents:")
#             for doc in source_documents:
#                 logging.info(f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}\n")

#             # Send response to the user
#             await bot.send_message(chat_id=message.chat.id, text=response_text)
#             end_time = time.perf_counter()
#             logging.info("Message sent successfully!")

#             # Store user data in MongoDB
#             users_collection.update_one(
#                 {"userid": user_id},
#                 {"$set": {"userid": user_id, "username": username, "first_name": first_name, "last_name": last_name}},
#                 upsert=True
#             )

#             # Store chat data
#             chat_data = {
#                 "message_id": message.message_id,
#                 "user_id": user_id,
#                 "message_date": datetime.utcnow().strftime("%Y-%m-%d"),
#                 "message_time": datetime.utcnow().strftime("%H:%M:%S"),
#                 "message_text": message.text,
#                 "response_text": response_text,
#                 "latency": calculate_latency(start_time, end_time),
#                 "tokens_used": cb.total_tokens,
#                 "retrieved_sources": [doc.metadata.get("source", "Unknown") for doc in source_documents]
#             }
#             chats_collection.insert_one(chat_data)

#     except TelegramError as te:
#         logging.error(f"Telegram error: {te}")
#     except Exception as e:
#         logging.error(f"Error: {e}")

#     return {"status": "ok"}

# # Run the FastAPI server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)




import os
import time
import logging
import pickle
import faiss
import numpy as np
import asyncio
from datetime import datetime
from pymongo import MongoClient
from telegram import Update
from telegram.ext import Application, MessageHandler, filters
from langchain_openai import OpenAIEmbeddings
from model import get_qa_chain
from langchain_community.callbacks import get_openai_callback
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from dotenv import load_dotenv

load_dotenv()

# Environment variables
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN1")
MONGO_URI = "mongodb://localhost:27017"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Initialize components
client = QdrantClient(url=QDRANT_URL)
chain = get_qa_chain()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Qdrant(client=client, collection_name="upes_scraped_data", embeddings=embeddings)
retriever = vectordb.as_retriever(score_threshold=0.8, top_k=3)

mongo_client = MongoClient(MONGO_URI)
db = mongo_client.telegram_bot
users_collection = db.users
chats_collection = db.chats

# FAISS Cache Paths
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)
FAISS_INDEX_FILE = os.path.join(CACHE_DIR, "faiss_cache.index")
CACHE_STORE_FILE = os.path.join(CACHE_DIR, "faiss_cache.pkl")

# Set up logging (logs both to file and console)
LOG_FILE = os.path.join(CACHE_DIR, "bot.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Load FAISS Index & Cache Store
try:
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CACHE_STORE_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(CACHE_STORE_FILE, "rb") as f:
            cache_store = pickle.load(f)
        logging.info("FAISS index and cache loaded successfully.")
    else:
        raise FileNotFoundError("Cache files not found, initializing new FAISS index.")
except Exception as e:
    logging.warning(f"Error loading FAISS cache: {e}")
    index = faiss.IndexFlatL2(1536)  # OpenAI embedding dimension
    cache_store = {}

# Helper function to calculate latency
def calculate_latency(start_time, end_time):
    return round(end_time - start_time, 2)

# Async helper to update user data in MongoDB
async def update_user_data(user):
    await asyncio.to_thread(
        users_collection.update_one,
        {"userid": user.id},
        {"$set": {
            "userid": user.id,
            "username": user.username or "Unknown",
            "first_name": user.first_name or "Unknown",
            "last_name": user.last_name or "Unknown"
        }},
        upsert=True
    )

# Async helper to insert chat data into MongoDB
async def save_chat_data(chat_data):
    await asyncio.to_thread(chats_collection.insert_one, chat_data)

# Helper function for synchronous FAISS index and cache saving
def save_faiss_index_and_cache():
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(CACHE_STORE_FILE, "wb") as f:
        pickle.dump(cache_store, f)

# Async wrapper to offload FAISS disk I/O
async def async_save_index():
    await asyncio.to_thread(save_faiss_index_and_cache)

# Handle incoming messages
async def handle_message(update: Update, context):
    try:
        start_time = time.perf_counter()
        user = update.message.from_user
        message = update.message
        user_id = user.id
        query_text = message.text.strip()
        query_vector = np.array(embeddings.embed_query(query_text)).reshape(1, -1)

        # Check FAISS cache first
        distances, indices = index.search(query_vector, 1)
        if indices[0][0] != -1:
            logging.info(f"FAISS Distance: {distances[0][0]}")

        # Determine if cache hit based on similarity threshold
        if indices[0][0] != -1 and indices[0][0] in cache_store and distances[0][0] < 0.001:
            response_text = cache_store[indices[0][0]]
            logging.info("Cache hit - retrieved from FAISS.")
        else:
            with get_openai_callback() as cb:
                response = chain.invoke({"query": query_text})
                response_text = response["result"]
                source_documents = response["source_documents"]

            # Add new query-response pair to FAISS & cache
            index.add(query_vector)
            cache_store[len(cache_store)] = response_text

            # Offload FAISS disk writes asynchronously
            asyncio.create_task(async_save_index())

            logging.info("Retrieved Documents:")
            for doc in source_documents:
                logging.info(f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}\n")
        
        # Send response to user
        await context.bot.send_message(chat_id=message.chat.id, text=response_text)
        end_time = time.perf_counter()
        logging.info("Message sent successfully!")

        # Prepare chat data for logging
        chat_data = {
            "message_id": message.message_id,
            "user_id": user_id,
            "message_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "message_time": datetime.utcnow().strftime("%H:%M:%S"),
            "message_text": query_text,
            "response_text": response_text,
            "faiss_distance": float(distances[0][0]),
            "latency": calculate_latency(start_time, end_time),
            "tokens_used": int(cb.total_tokens) if 'cb' in locals() else 0
        }

        # Offload MongoDB updates asynchronously
        asyncio.create_task(update_user_data(user))
        asyncio.create_task(save_chat_data(chat_data))
        logging.info("MongoDB update tasks scheduled.")

    except Exception as e:
        logging.error(f"Error handling message: {e}")

# Set up Telegram bot application
def main():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()

if __name__ == "__main__":
    main()
