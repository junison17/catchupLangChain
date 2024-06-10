import streamlit as st
import openai
from langchain_community.callbacks import get_openai_callback
from my_modules import view_sourcecode, modelName, modelName_embedding_small, modelName4o

def generate_text(vector, question, model_name, selected_language, openai_api_key):
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain

        prompt_template = """Answer the following question based only on the provided context in English and Translate the answer in """ + selected_language + """ :
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
    
    except openai.OpenAIError as e:
        st.warning("generate_text: Incorrect API key provided or OpenAI API error.")
        st.error(str(e))

def crawl_vectorize(api_key, urlFromUser):
    try:
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        loader = WebBaseLoader(urlFromUser)
        docs = loader.load()

        embedding_model_name = modelName_embedding_small()
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=embedding_model_name)

        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        with get_openai_callback() as cb:
            vector = FAISS.from_documents(documents, embeddings)

        return vector

    except openai.OpenAIError as e:
        st.warning("crawl_vectorize: Incorrect API key provided or OpenAI API error.")
        st.error(str(e))

def set_SessionState(api_key, urlFromUser, select_model, selected_language):
    if not api_key or not urlFromUser or not select_model or not selected_language:
        st.warning("To get started, please enter your OpenAI API Key and the URL of the web page to fetch data from, then press Enter.")
        return

    model_name = modelName() if select_model == "Cheapest" else modelName4o()

    st.write(f"Hello! Please ask a question related to the content of {urlFromUser}.") 
    vector = crawl_vectorize(api_key, urlFromUser)

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = model_name

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = generate_text(vector, prompt, model_name, selected_language, api_key)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.title('RAG - Fetch Data from given URL.')

    api_key = st.text_input("Please input your OpenAI API Key:", type="password")
    urlFromUser = st.text_input("Please enter the URL of the web page to fetch data from. e.g. https://www.tecace.com/about")
    select_model = st.radio("Please choose the Model you'd like to use.", ["Cheapest", "GPT 4o"]) 

    available_languages = ["Korean", "Spanish", "French", "German", "Chinese", "Japanese"]
    selected_language = st.selectbox("Select a language:", available_languages)

    set_SessionState(api_key, urlFromUser, select_model, selected_language)

if __name__ == "__main__":
    main()
