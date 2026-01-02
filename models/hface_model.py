from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
import os

load_dotenv()

def load_hface_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

def get_hface_response(query):
    retriever = load_hface_vectorstore()

    llm =  HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.3, 
        max_new_tokens=100
    )

    system_prompt = (
        "Sen, sadece 'Son Kartuş' oyunu hakkında konuşan, bilge ve zarif bir sanat duayenisin."
        "Oyun dışındaki tüm konuları kibarca reddederek sohbeti her zaman bu eserin metnine ve sanatsal derinliğine yönlendirmelisin."
        "Kullanıcılara yardımcı olurken bir tiyatro üstadının nezaketiyle hareket etmeli ve sadece sana sağlanan oyun verilerine sadık kalmalısın."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("human", system_prompt + "Question: {input}\nAnswer:")
    ])

    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(retriever,question_answer_chain)
    response = rag_chain.invoke({"input": query})
    
    return response["answer"]
