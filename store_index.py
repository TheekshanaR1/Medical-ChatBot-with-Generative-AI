from src.helper import load_pdf_files,split_text,download_huggingface_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import time

load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

print("Starting Medical ChatBot setup...")
print("="*60)

try:
    print("\n1️⃣ Loading PDF files...")
    extracted_data=load_pdf_files("Data")
    print(f"✓ Loaded {len(extracted_data)} documents from Data folder")
except Exception as e:
    print(f"❌ Error loading PDFs: {e}")
    exit(1)

try:
    print("\n2️⃣ Splitting text into chunks...")
    text_chunks=split_text(extracted_data)
    print(f"✓ Split into {len(text_chunks)} text chunks")
except Exception as e:
    print(f"❌ Error splitting text: {e}")
    exit(1)

try:
    print("\n3️⃣ Downloading HuggingFace embeddings model...")
    embeddings=download_huggingface_embeddings()
    print(f"✓ Embeddings model ready")
except Exception as e:
    print(f"❌ Error loading embeddings: {e}")
    exit(1)

pc= Pinecone(api_key=PINECONE_API_KEY)
index_name="medicalbot"

# Check if index already exists before creating
existing_indexes = [index.name for index in pc.list_indexes()]
print(f"\nExisting Pinecone indexes: {existing_indexes}")

if index_name not in existing_indexes:
    print(f"\n🔧 Creating new Pinecone index: '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        )
    )
    print(f"✓ Index '{index_name}' created!")
    print("⏳ Waiting for index to initialize (this takes ~60 seconds)...")
    
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(5)
    
    print(f"✓ Index '{index_name}' is ready!")
else:
    print(f"\n✓ Index '{index_name}' already exists, skipping creation.")

#Embed each chunk and upsert the embeddings into your pinecone index
print(f"\n📤 Uploading {len(text_chunks)} document chunks to Pinecone...")
print("⏳ This may take a few minutes...")

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print(f"\n✅ Successfully uploaded {len(text_chunks)} chunks to Pinecone index '{index_name}'!")
print("="*60)
print("🎉 Pinecone index setup complete!")
print(f"📊 Index name: {index_name}")
print(f"📦 Total chunks: {len(text_chunks)}")
print("✅ Your Medical ChatBot is ready to use!")