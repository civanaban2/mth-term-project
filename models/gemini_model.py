from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def load_gemini_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

def get_gemini_response(query):
    retriever = load_gemini_vectorstore()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.3,
        max_tokens=500
    )

    system_prompt = (
        "Sen, sadece 'Son Kartuş' oyunu hakkında konuşan, bilge ve zarif bir sanat duayenisin."
        "Oyun dışındaki tüm konuları kibarca reddederek sohbeti her zaman bu eserin metnine ve sanatsal derinliğine yönlendirmelisin."
        "Kullanıcılara yardımcı olurken bir tiyatro üstadının nezaketiyle hareket etmeli ve sadece sana sağlanan oyun verilerine sadık kalmalısın."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            ("human","{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(retriever,question_answer_chain)
    response = rag_chain.invoke({"input": query})

    return response["answer"]
