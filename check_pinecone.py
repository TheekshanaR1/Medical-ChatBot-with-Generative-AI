from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

print("Checking Pinecone indexes...")
print("="*60)

pc = Pinecone(api_key=PINECONE_API_KEY)

# List all indexes
indexes = pc.list_indexes()
print(f"\n📋 All Pinecone indexes:")
if indexes:
    for idx in indexes:
        print(f"  - {idx.name}")
else:
    print("  ❌ No indexes found!")

# Check for medicalbot specifically
index_name = "medicalbot"
existing_indexes = [idx.name for idx in indexes]

if index_name in existing_indexes:
    print(f"\n✅ Index '{index_name}' EXISTS!")
    
    # Get index stats
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    
    print(f"\n📊 Index Statistics:")
    print(f"  - Total vectors: {stats.get('total_vector_count', 0)}")
    print(f"  - Dimension: {stats.get('dimension', 'N/A')}")
    print(f"  - Namespaces: {stats.get('namespaces', {})}")
    
    # Get index details
    details = pc.describe_index(index_name)
    print(f"\n🔧 Index Details:")
    print(f"  - Status: {details.status}")
    print(f"  - Host: {details.host}")
    print(f"  - Metric: {details.metric}")
else:
    print(f"\n❌ Index '{index_name}' DOES NOT EXIST!")
    print("\nRun 'python store_index.py' to create it.")

print("="*60)
