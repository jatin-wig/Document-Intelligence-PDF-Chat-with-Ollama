import streamlit as st
import os
import shutil
import time

from utils.loader import load_pdf
from utils.splitter import split_documents
from utils.vectorstore import create_vector_store
from utils.rag_chain import get_rag_chain

st.set_page_config(page_title="Document Intelligence", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #FAFAFA;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
    [data-testid="stChatInput"] {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    .main-header {
        text-align: center;
        color: #111827;
        font-weight: 600;
        margin-top: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #6B7280;
        font-weight: 400;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False


@st.cache_resource
def load_chain():
    return get_rag_chain()


def reset_workspace():
    st.session_state.messages = []
    st.session_state.doc_loaded = False
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    st.cache_resource.clear()


with st.sidebar:
    st.title("Document Intelligence")
    
    st.divider()
    
    if not st.session_state.doc_loaded:
        st.subheader("Data Ingestion")
        uploaded_file = st.file_uploader("Select PDF for analysis", type="pdf", label_visibility="collapsed")

        if uploaded_file:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.status("Initializing document processing pipeline...", expanded=True) as status:
                st.write("Extracting text contents...")
                documents = load_pdf("temp.pdf")
                
                st.write("Segmenting text into semantic chunks...")
                chunks = split_documents(documents)
                
                st.write("Generating vector embeddings...")
                create_vector_store(chunks)
                
                status.update(label="Document indexed successfully.", state="complete", expanded=False)

            st.session_state.doc_loaded = True
            st.session_state.messages = [{"role": "ai", "content": "Document processing complete. The system is ready for queries."}]
            st.rerun()
    else:
        st.success("System Status: Active and Indexed")
        if st.button("Reset Workspace", type="primary", use_container_width=True):
            reset_workspace()
            st.rerun()


if not st.session_state.doc_loaded:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='main-header'>Document Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Awaiting document ingestion. Please upload a file via the sidebar to initialize the system.</h3>", unsafe_allow_html=True)
        
        st.info("System capabilities include semantic search, summarization, and contextual question-answering based on the provided corpus.")

else:
    chain = load_chain()

    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "ai"
        with st.chat_message(role):
            if role == "user":
                st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ai-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

    user_prompt = st.chat_input("Input query...")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(f"<div class='user-bubble'>{user_prompt}</div>", unsafe_allow_html=True)

        with st.chat_message("ai"):
            with st.spinner("Analyzing context..."):
                response = chain.invoke(user_prompt)
                st.markdown(f"<div class='ai-bubble'>{response}</div>", unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "ai", "content": response})