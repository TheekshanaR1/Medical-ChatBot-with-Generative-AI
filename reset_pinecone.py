from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

print("🗑️  Deleting existing Pinecone index...")
try:
    pc.delete_index(index_name)
    print(f"✅ Index '{index_name}' deleted successfully!")
    print("\nNow run: python store_index.py")
    print("This will upload all ~40,000 chunks fresh.")
except Exception as e:
    print(f"❌ Error: {e}")
