# upload_and_save.py
import streamlit as st
import faiss
import os
import pickle
import datetime
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from my_modules import modelName_embedding_small
from langchain_community.callbacks import get_openai_callback

TOPICS_DIR = "faiss_indices"

def save_faiss_index(vector, topic, api_key, model_name, metadata):
    if not os.path.exists(TOPICS_DIR):
        os.makedirs(TOPICS_DIR)
    faiss.write_index(vector.index, os.path.join(TOPICS_DIR, f"{topic}.index"))

    # Extract docstore data manually
    docstore_data = {k: v.__dict__ for k, v in vector.docstore._dict.items()}
    index_to_docstore_id_data = vector.index_to_docstore_id

    with open(os.path.join(TOPICS_DIR, f"{topic}_meta.pkl"), "wb") as f:
        pickle.dump((api_key, model_name, docstore_data, index_to_docstore_id_data, metadata), f)

def vectorize_documents(api_key, documents):
    embedding_model_name = modelName_embedding_small()
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=embedding_model_name)

    text_splitter = RecursiveCharacterTextSplitter()
    split_documents = text_splitter.split_documents(documents)
    with get_openai_callback() as cb:
        vector = FAISS.from_documents(split_documents, embeddings)

    return vector, embedding_model_name

def main():
    st.title('Upload and Save to FAISS')
    
    api_key = st.text_input("Please input your OpenAI API Key:", type="password")
    uploaded_files = st.file_uploader("Upload text files", type="txt", accept_multiple_files=True)
    topic = st.text_input("Please enter the topic for the uploaded files:")

    if st.button("Save to FAISS") and api_key and uploaded_files and topic:
        documents = []
        metadata = {
            "creation_date": str(datetime.datetime.now()),
            "content_size": 0,
            "file_names": [uploaded_file.name for uploaded_file in uploaded_files]
        }
        for uploaded_file in uploaded_files:
            content = uploaded_file.read().decode('utf-8')
            document = Document(page_content=content)
            documents.append(document)
            metadata["content_size"] += len(content)

        vector, model_name = vectorize_documents(api_key, documents)
        save_faiss_index(vector, topic, api_key, model_name, metadata)
        st.success(f"Data for topic '{topic}' saved successfully.")

if __name__ == "__main__":
    main()
