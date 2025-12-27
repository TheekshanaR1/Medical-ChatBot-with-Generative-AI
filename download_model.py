import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Downloading HuggingFace model...")
print("This is a one-time download (~100MB)")
print("Please wait without interrupting...")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("\n✓ Model downloaded successfully!")
print("You can now run app.py")
