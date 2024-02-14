import uuid
from pathlib import Path

import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from src.documents import VectorDB

def initialize_session_state():
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None

    if "openai_api_model" not in st.session_state:
        st.session_state.openai_api_model = None 

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "base_path" not in st.session_state:
        st.session_state.base_path = f'resources/{st.session_state.session_id}'
    
    upload_path = f'{st.session_state.base_path}/uploads'
    Path(upload_path).mkdir(parents=True, exist_ok=True)

def set_openai_api_key(api_key):
    st.session_state["openai_api_key"] = api_key

def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"
            "2. Upload a Documents 📄\n"
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",
            value=st.session_state.get("OPENAI_API_KEY", ""),
        )

        if api_key_input:
            set_openai_api_key(api_key_input)
        
        openai_model = ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4", "gpt-4-0613"] 
        st.session_state.openai_api_model = st.sidebar.radio(
                    "Choose OpenAI model:",
                    openai_model,
                    index=1,
                )

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "This tool allows you to chat with your "
            "PDFs, DOCXs, Markdowns and more. "
        )
        st.markdown(
            "This tool is a work in progress. "
            "You can contribute to the project on [GitHub](https://github.com/bkhanal-11/DocChat) "
            "with your feedback and suggestions💡."
        )
        st.markdown("Made by [Bishwash](https://github.com/bkhanal-11)")
        st.markdown("---")


st.set_page_config(
    page_title="DocChat",
    page_icon="🦜",
    layout="wide",
)

st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)  # Removes Page Navigation

st.title("DocChat 📄")
sidebar()
initialize_session_state()

# Page 1 - Upload PDF
st.header("Upload your Documents")
uploaded_file = st.file_uploader("Choose a file(s)", type=["pdf", "docx", "md", "txt"], accept_multiple_files=True)

if uploaded_file:
    vectordb = VectorDB(uploaded_file, st.session_state.session_id)

    if st.session_state.openai_api_key is None:
        st.error("Please enter your OpenAI API key in the sidebar to continue.")

    else:
        st.success("OpenAI API key set successfully!")

        with st.spinner("Processing PDF File...This may take a while⏳"):
            st.session_state.vector_store = vectordb.create_vectors(st.session_state.base_path)

        st.success("PDF uploaded successfully!")
        switch_page("chat")