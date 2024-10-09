import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

model = ChatGroq(groq_api_key=groq_api_key, model='Llama3-8b-8192')

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the context only.
Provide the answer accurately and briefly to the question
<context>
{context}
<context>
Question:{input}
"""
)

st.set_page_config(page_title = 'Simple RAG', page_icon = '⛓️', initial_sidebar_state = 'collapsed')

st.sidebar.header('About')
st.sidebar.markdown(
"""
Embeddings: Craig/paraphrase-MiniLM-L6-v2  
VectorDB: FAISS  
LLM: Llama3-8b-8192
"""
)

st.title('Simple RAG Application')

st.warning('This is a simple RAG demonstration application. It uses open-source models for embeddings and \
inference. So it can be slow and ineffecient.', icon='⚠️')


def create_vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name='Craig/paraphrase-MiniLM-L6-v2')
        st.session_state.loader = PyPDFDirectoryLoader('documents')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.rerun()

if 'vectors' not in st.session_state:
    st.write('The vector store database is not yet ready')
    if st.button('Create'):
        with st.spinner('Working...'):
            create_vector_embedding()

if 'vectors' in st.session_state:
    user_prompt = st.text_input('Enter your query here')
    if user_prompt:
        document_chain = create_stuff_documents_chain(model, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(response['answer'])

        with st.expander('Context'):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('\n\n')
