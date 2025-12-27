import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Use offline mode
os.environ['HF_DATASETS_OFFLINE'] = '1'  # Use offline mode for datasets

from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import *

app= Flask(__name__)
load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

print("Loading Medical ChatBot...")
print("1. Loading embeddings model (this may take a minute)...")
embeddings=download_huggingface_embeddings()
print("✓ Embeddings loaded")

index_name="medicalbot"

print("2. Connecting to Pinecone...")
#Load existing Pinecone index
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})
print("✓ Connected to Pinecone index")
print("✓ Medical ChatBot ready!")
print("="*60)

@app.route("/")
def index():
    return render_template("chat.html")



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User question:", msg)
    
    # Handle greetings and simple queries
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    if msg.lower().strip() in greetings:
        return "Hello! I'm your Medical ChatBot. I can help answer medical questions based on thousands of medical documents. Ask me anything about symptoms, diseases, treatments, or medical conditions!"
    
    # Handle very short queries (less than 3 characters)
    if len(msg.strip()) < 3:
        return "Please ask a complete medical question. For example: 'What is diabetes?' or 'What are symptoms of flu?'"
    
    # Retrieve relevant documents
    docs = retriever.invoke(msg)
    
    # Check if results are relevant (basic check)
    if not docs or len(docs) == 0:
        return "I couldn't find relevant information about that. Please try rephrasing your question or ask about a specific medical condition, symptom, or treatment."
    
    # Format response from retrieved documents
    response_parts = []
    for i, doc in enumerate(docs):
        content = doc.page_content.strip()
        # Limit content length for better readability
        if len(content) > 500:
            content = content[:500] + "..."
        response_parts.append(f"<strong>Source {i+1}:</strong><br>{content}")
    
    response = f"<p><strong>Based on the medical documents:</strong></p><br>" + "<br><br>".join(response_parts)
    
    print("Response sent")
    return str(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)