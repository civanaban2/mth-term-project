import streamlit as st
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from embed_data import embed_data
from models.gemini_model import get_gemini_response
from models.hface_model import get_hface_response

st.title("RAG Application with Multiple LLMs")

if not os.path.exists("./chroma_db"):
	st.write("Embedding data, please wait...")
	embed_data()
	st.write("Data embedding completed.")

option = st.selectbox(
	'Select the LLM you want to use:',
	('Gemini', 'HuggingFace')
)

query=st.chat_input("Say something:")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if option == 'Gemini':
                    response = get_gemini_response(query)
                else:
                    response = get_hface_response(query)
                
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")
	