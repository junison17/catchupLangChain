import streamlit as st
import faiss
import os
import pickle
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from my_modules import modelName, modelName4o
from langchain_community.callbacks import get_openai_callback

TOPICS_DIR = "faiss_indices"

def load_faiss_index(topic):
    index_file = os.path.join(TOPICS_DIR, f"{topic}.index")
    meta_file = os.path.join(TOPICS_DIR, f"{topic}_meta.pkl")
    if os.path.exists(index_file) and os.path.exists(meta_file):
        index = faiss.read_index(index_file)
        with open(meta_file, "rb") as f:
            api_key, model_name, docstore_data, index_to_docstore_id_data, metadata = pickle.load(f)
        
        embedding_function = OpenAIEmbeddings(openai_api_key=api_key, model=model_name)
        
        # Manually reconstruct the docstore
        docstore = InMemoryDocstore({k: Document(**v) for k, v in docstore_data.items()})
        
        return FAISS(index=index, embedding_function=embedding_function, docstore=docstore, index_to_docstore_id=index_to_docstore_id_data), metadata
    return None, None

def generate_text(vector, question, model_name, openai_api_key, document_only):
    if document_only:
        prompt_template = """You are responsible for providing information to users regarding the data on the Topic. please indicate the URL in the reference materials at the very end of your reply if possible.
        If there is anything that is not in the context, please reply that you cannot answer because the uploaded document does not contain that content. :
        <context>
        {context}
        </context>
        Question: {input}"""
    else:
        prompt_template = """Answer the following question.
        Also, please indicate the URL in the reference materials at the very end of your reply if possible :
        <context>
        {context}
        </context>
        Question: {input}"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    with get_openai_callback() as cb:
        generated_text = retrieval_chain.invoke({"input": question})

    return generated_text["answer"]

def main():
    st.title('Select and Use FAISS Topic')
    
    api_key = st.text_input("Please input your OpenAI API Key:", type="password")
    topics = [f.split('.')[0] for f in os.listdir(TOPICS_DIR) if f.endswith('.index')]
    selected_topic = st.selectbox("Select a topic:", topics)
    select_model = st.radio("Please choose the Model you'd like to use.", ["Cheapest", "GPT 4o"]) 

    document_only = st.radio("Answer from document only:", ["Yes", "No"]) == "Yes"

    topic_key_prefix = f"topic_{selected_topic}_"

    st.write("**Note :** *If you change the above settings, you will need to reload the topic for the changes to take effect.*")

    if st.button("Load Topic"):
        if not api_key:
            st.warning("Please input your OpenAI API Key.")
        elif not selected_topic:
            st.warning("Please select a topic.")
        else:
            model_name = modelName() if select_model == "Cheapest" else modelName4o()
            vector, metadata = load_faiss_index(selected_topic)
            if not vector:
                st.warning(f"Could not load FAISS index for topic '{selected_topic}'.")
                return

            st.session_state[f"{topic_key_prefix}vector"] = vector
            st.session_state[f"{topic_key_prefix}model_name"] = model_name
            st.session_state[f"{topic_key_prefix}api_key"] = api_key
            st.session_state[f"{topic_key_prefix}document_only"] = document_only
            st.session_state[f"{topic_key_prefix}messages"] = []

            st.write(f"You are using '{select_model}' model now.")
            st.write(f"From document only : '{document_only}'.")

    if f"{topic_key_prefix}vector" in st.session_state:
        st.write(f"Topic '{selected_topic}' loaded. Please ask a question related to this topic.")

        for message in st.session_state[f"{topic_key_prefix}messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("What is your question?"):
            vector = st.session_state[f"{topic_key_prefix}vector"]
            model_name = st.session_state[f"{topic_key_prefix}model_name"]
            api_key = st.session_state[f"{topic_key_prefix}api_key"]
            document_only = st.session_state[f"{topic_key_prefix}document_only"]

            st.session_state[f"{topic_key_prefix}messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = generate_text(vector, prompt, model_name, api_key, document_only)
            
            st.session_state[f"{topic_key_prefix}messages"].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
