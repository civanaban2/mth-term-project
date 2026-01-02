from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def embed_data():

	load_dotenv()

	loader = PyPDFLoader("data/sonkartus.pdf")
	data = loader.load()

	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
	docs = text_splitter.split_documents(data)

	embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
	Chroma.from_documents(documents=docs,embedding=embeddings, persist_directory="./chroma_db")

if __name__ == "__main__":
    embed_data()