import os, tempfile
import glob

import pinecone
from pathlib import Path
import random

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain_openai import ChatOpenAI

from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone as Pinecone1
from pinecone import ServerlessSpec

from langchain_community.document_loaders import UnstructuredPowerPointLoader

import streamlit as st

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')
embeddings = None

st.set_page_config(page_title="PPT RAG")
st.title("PPT Retrieval Augmented Generation Engine")


def load_documents():
    #loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.ppt*')
    files1 = glob.glob(TMP_DIR.joinpath("tmp").with_name("*.ppt*").as_posix())
    #st.warning("Jai Bhalla " + str(files1))
    #print(files1)

    documents = list()
    for file1 in files1:
        loader = UnstructuredPowerPointLoader(file1)
        documents = documents + loader.load()

    return documents


def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts


def embeddings_on_local_vectordb(texts):
    randstr = str(random.randint(1, 10000))
    dir1 = LOCAL_VECTOR_STORE_DIR.as_posix() + randstr
    print(dir1)
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key),
                                     persist_directory=dir1)
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever


def embeddings_on_pinecone(texts):
    #pinecone.init(api_key=st.session_state.pinecone_api_key, environment=st.session_state.pinecone_env)
    #embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)

    pc = Pinecone1(api_key=st.session_state.pinecone_api_key)
    if st.session_state.pinecone_index in pc.list_indexes().names():
        #print(f"Index '{st.session_state.pinecone_index}' already exists.")
        st.warning(
            f"Index : {st.session_state.pinecone_index} allready exists, reusing existing index and not submitting ")
        vectordb = PineconeVectorStore.from_existing_index(index_name=st.session_state.pinecone_index, embedding=embeddings)
    else:
        #print(f"Index '{st.session_state.pinecone_index}' does not exist.")
        st.warning(f"Index '{st.session_state.pinecone_index}' does not exist. Creating a fresh index")
        pc.create_index(
            name=st.session_state.pinecone_index,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    vectordb = PineconeVectorStore.from_documents(texts, embeddings, index_name=st.session_state.pinecone_index)
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever


def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        #llm=OpenAIChat(openai_api_key=st.session_state.openai_api_key),
        #llm=OpenAIChat(),
        llm=ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
            # base_url="...",
            # organization="...",
            # other params...
        ),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result


def input_fields():
    #
    with st.sidebar:
        #
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")
        #
        if "pinecone_api_key" in st.secrets:
            st.session_state.pinecone_api_key = st.secrets.pinecone_api_key
        else:
            st.session_state.pinecone_api_key = st.text_input("Pinecone API key", type="password")
        #
        if "pinecone_env" in st.secrets:
            st.session_state.pinecone_env = st.secrets.pinecone_env
        else:
            st.session_state.pinecone_env = st.text_input("Pinecone environment")
        #
        if "pinecone_index" in st.secrets:
            st.session_state.pinecone_index = st.secrets.pinecone_index
        else:
            st.session_state.pinecone_index = st.text_input("Pinecone index name")
    #
    st.session_state.pinecone_db = st.toggle('Use Pinecone Vector DB')
    #
    st.button("Reuse Existing Index w/o upload", on_click=reuse_index)
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pptx", accept_multiple_files=True)
    #


def reuse_index():
    pc = Pinecone1(api_key=st.session_state.pinecone_api_key)
    if st.session_state.pinecone_index in pc.list_indexes().names():
        # print(f"Index '{st.session_state.pinecone_index}' already exists.")
        st.warning(
            f"Index : {st.session_state.pinecone_index} allready exists, reusing existing index")
        vectordb = PineconeVectorStore.from_existing_index(index_name=st.session_state.pinecone_index, embedding=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={'k': 7})
        st.session_state.retriever = retriever

    else:
        # print(f"Index '{st.session_state.pinecone_index}' does not exist.")
        st.warning(f"Index '{st.session_state.pinecone_index}' does not exist. Submit a pdf to create a new index")


def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
        return
    if st.session_state.pinecone_db and (not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index):
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        #st.warning(f"xxxx"+str(st.session_state.source_docs))
        try:
            for source_doc in st.session_state.source_docs:
                #
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pptx') as tmp_file:
                    #st.warning(f"xxxx"+str(tmp_file.name))
                    #st.warning(f"xxxx"+str(TMP_DIR.as_posix()))
                    tmp_file.write(source_doc.read())
                    tmp_file.close()
                #
                documents = load_documents()
                #
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                #
                texts = split_documents(documents)
                #
                if not st.session_state.pinecone_db:
                    st.session_state.retriever = embeddings_on_local_vectordb(texts)
                else:
                    st.session_state.retriever = embeddings_on_pinecone(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")


def boot():
    global embeddings
    #
    input_fields()
    #

    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    st.button("Submit Documents", on_click=process_documents)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])
        #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)


if __name__ == '__main__':
    #
    boot()
