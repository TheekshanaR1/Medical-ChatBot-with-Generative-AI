from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#Extract Data from PDF files
def load_pdf_files(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

#Split Data into chunks
def split_text(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

#Download the embeddings from Huggingface
def download_huggingface_embeddings():
    # Use local files only to avoid network issues
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder=None,  # Use default cache
        multi_process=False
    )
    return embeddings