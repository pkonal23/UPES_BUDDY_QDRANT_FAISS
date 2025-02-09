import os
from pymongo import MongoClient

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client.telegram_bot

# Create collections and set indexes
def create_collections():
    # Users collection
    users_collection = db.users
    users_collection.create_index("userid", unique=True)

    # Chats collection
    chats_collection = db.chats
    chats_collection.create_index("message_id", unique=True)

    print("Collections and indexes created successfully!")

if __name__ == "__main__":
    create_collections()
