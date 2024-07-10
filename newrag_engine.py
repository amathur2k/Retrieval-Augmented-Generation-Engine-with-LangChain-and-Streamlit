import os, tempfile
import pinecone
from pathlib import Path
import random
import traceback

import signal
import sys


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

import streamlit as st
from pinecone import Pinecone as Pinecone1
from pinecone import ServerlessSpec

import hashlib

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate



def generate_short_id(content: str) -> str:
    """
    Generate a short ID based on the content using SHA-256 hash.

    Args:
    - content (str): The content for which the ID is generated.

    Returns:
    - short_id (str): The generated short ID.
    """
    hash_obj = hashlib.sha256()
    hash_obj.update(content.encode("utf-8"))
    return hash_obj.hexdigest()


def combine_vector_and_text(
    documents: list[any], doc_embeddings: list[list[float]]
) -> list[dict[str, any]]:
    """
    Process a list of documents along with their embeddings.

    Args:
    - documents (List[Any]): A list of documents (strings or other types).
    - doc_embeddings (List[List[float]]): A list of embeddings corresponding to the documents.

    Returns:
    - data_with_metadata (List[Dict[str, Any]]): A list of dictionaries, each containing an ID, embedding values, and metadata.
    """
    data_with_metadata = []

    for doc_text, embedding in zip(documents, doc_embeddings):
        # Convert doc_text to string if it's not already a string
        if not isinstance(doc_text, str):
            doc_text = str(doc_text)

        # Generate a unique ID based on the text content
        doc_id = generate_short_id(doc_text)

        # Create a data item dictionary
        data_item = {
            "id": doc_id,
            "values": embedding[0],
            "metadata": {"text": doc_text},  # Include the text as metadata
        }

        # Append the data item to the list
        data_with_metadata.append(data_item)

    return data_with_metadata


embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")


def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    randstr = str(random.randint(1,10000))
    dir1 = LOCAL_VECTOR_STORE_DIR.as_posix()+randstr
    print (dir1)
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(),
                                     persist_directory=dir1)
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def embeddings_on_pinecone(texts):


    #pinecone.init(api_key=st.session_state.pinecone_api_key, environment=st.session_state.pinecone_env)

    #api_key = os.environ.get("PINECONE_API_KEY")

    embedded = [embeddings.embed_documents(doc) for doc in texts]

    print(len(embedded))

    data_with_meta_data = combine_vector_and_text(documents=texts,
                                                  doc_embeddings=embedded)

    # configure client
    pc = Pinecone1(api_key=st.session_state.pinecone_api_key)

    #vectordb = Pinecone.from_documents(texts, embeddings, index_name=st.session_state.pinecone_index)
    #retriever = vectordb.as_retriever()
    cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
    region = os.environ.get('PINECONE_REGION') or 'us-east-1'

    spec = ServerlessSpec(cloud=cloud, region=region)

    index = pc.Index('index2')

    index.upsert(vectors=data_with_meta_data)


    return index

def get_query_embeddings(query: str) -> list[float]:
    """This function returns a list of the embeddings for a given query

    Args:
        query (str): The actual query/question

    Returns:
        list[float]: The embeddings for the given query
    """
    query_embeddings = embeddings.embed_query(query)
    return query_embeddings

def query_pinecone_index(
    query_embeddings: list, index,  top_k: int = 2, include_metadata: bool = True
) -> dict[str, any]:
    """
    Query a Pinecone index.

    Args:
    - index (Any): The Pinecone index object to query.
    - vectors (List[List[float]]): List of query vectors.
    - top_k (int): Number of nearest neighbors to retrieve (default: 2).
    - include_metadata (bool): Whether to include metadata in the query response (default: True).

    Returns:
    - query_response (Dict[str, Any]): Query response containing nearest neighbors.
    """
    query_response = index.query(
        vector=query_embeddings, top_k=top_k, include_metadata=include_metadata
    )
    return query_response
def query_llm2(index, query):
    query_embeddings = get_query_embeddings(query=query)
    answers = query_pinecone_index(query_embeddings=query_embeddings, index=index)
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
        # base_url="...",
        # organization="...",
        # other params...
    )
    # Extract only the text from the dictionary before passing it to the LLM
    text_answer = " ".join([doc['metadata']['text'] for doc in answers['matches']])

    prompt = f"{text_answer} Using the provided information, give me a better and summarized answer"

    prompt = ChatPromptTemplate.from_template(prompt)

    final_answer = llm(prompt)

    return final_answer
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
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    #


def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        #st.warning(f"xxxx"+str(st.session_state.source_docs))
        try:
            for source_doc in st.session_state.source_docs:
                #
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
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
            st.error(f"Jai Bhalla An error occurred: {e}")

            st.error(''.join(traceback.format_exception(None, e, e.__traceback__)))

def boot():
    #
    input_fields()
    #
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
        #response = query_llm(st.session_state.retriever, query)
        response = query_llm(st.session_state.retriever, query)

        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()
    
