from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
print(client.list_database_names())  # Check if MongoDB is accessible
