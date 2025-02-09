import os
import time
from datetime import datetime
from fastapi import FastAPI, Request
from pymongo import MongoClient
from telegram import Bot, Update
from telegram.error import TelegramError
from langchain_redis import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_llm_cache
from model import get_qa_chain
from langchain_community.callbacks import get_openai_callback

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
chain = get_qa_chain()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
semantic_cache = RedisSemanticCache(redis_url=REDIS_URL, embeddings=embeddings, distance_threshold=0.2)
set_llm_cache(semantic_cache)

# Initialize Telegram Bot, MongoDB client, and FastAPI app
bot = Bot(BOT_TOKEN)
app = FastAPI()
client = MongoClient(MONGO_URI)
db = client.telegram_bot
users_collection = db.users
chats_collection = db.chats


# Helper function to calculate latency
def calculate_latency(message_time,response_time):
    return round(response_time - message_time, 2)


@app.post("/webhook")
async def webhook(request: Request):
    try:
        # Parse the incoming update
        data = await request.json()
        update = Update.de_json(data, bot)

        if update.message and update.message.text:
            message_time = time.time()

            # Extract user and message details
            user = update.message.from_user
            message = update.message

            with get_openai_callback() as cb:
                response_text = chain(message.text)['result']  # Replace with your chain logic

            # Send response to the user
            await bot.send_message(chat_id=message.chat_id, text=response_text)
            response_time = time.time()
            print("Message sent")


            user_data = {
                "userid": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
            }

            # Store user data (upsert operation)
            users_collection.update_one({"userid": user.id}, {"$set": user_data}, upsert=True)

            # Store chat data
            chat_data = {
                "message_id": message.message_id,
                "user_id": user.id,
                "message_date": message.date.strftime("%Y-%m-%d"),
                "message_time": message.date.strftime("%H:%M:%S"),
                "message_text": message.text,
                "response_text": response_text,
                "latency": calculate_latency(message_time,response_time),
                "tokens_used": cb.total_tokens,
            }
            chats_collection.insert_one(chat_data)


    except TelegramError as te:
        print(f"Telegram error: {te}")
    except Exception as e:
        print(f"Error: {e}")

    return {"status": "ok"}


# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
