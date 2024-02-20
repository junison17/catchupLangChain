import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAIError
from my_modules import view_sourcecode, modelName, modelName_embedding_small
import os
from langchain_community.callbacks import get_openai_callback
from langchain_sidebar_content import LC_QuickStart_02

# Function to interact with OpenAI API
def generate_text(api_key, language, question):
    try: 
        openai_api_key = api_key
        embedding_model_name = modelName_embedding_small()
        model_name = modelName()

        # 1. Get Data
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader("https://docs.smith.langchain.com")
        docs = loader.load()

        # 2. Set Embedding model
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model_name)

        # 3. Store vector into vector storage
        from langchain_community.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)

        # 4. create documents chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.chains.combine_documents import create_stuff_documents_chain

        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context in English and Translate the answer in """ + language + """ :

        <context>
        {context}
        </context>

        Question: {input}""")

        llm = ChatOpenAI(openai_api_key=openai_api_key,model_name=model_name )
        document_chain = create_stuff_documents_chain(llm, prompt)

        # 5. Create Retrieval Chain
        from langchain.chains import create_retrieval_chain

        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with get_openai_callback() as cb:
            generated_text = retrieval_chain.invoke({"input": question})
            st.write(cb)

        # LangSmith offers several features that can help with testing:...

        vector.delete([vector.index_to_docstore_id[0]])
        # Is now missing
        0 in vector.index_to_docstore_id

        return generated_text
    except OpenAIError as e:
        st.warning("Incorrect API key provided or OpenAI API error.")
        st.warning(e)

def main():
    st.title('LangChain Quickstart 02 - Retrieval Chain')

    # Get user input for OpenAI API key
    api_key = st.text_input("Please input your OpenAI API Key:", type="password")
    st.write("Fetching this Web Page Contents : https://docs.smith.langchain.com")

    # List of Questions
    quastions = ["How can langsmith help with testing?", 
                "Please tell me the additional Resources.", 
                "What are the Next Steps?",
                "Please summarize this context."]

    # User-selected question
    selected_question = st.selectbox("Select a question:", quastions)    

    st.write("*Answers will be in English and the language of your choice.* ")  

    # List of languages available for ChatGPT
    available_languages = ["Korean", "Spanish", "French", "German", "Chinese", "Japanese"]

    # User-selected language
    selected_language = st.selectbox("Select a language:", available_languages)  

    # Button to trigger text generation
    if st.button("Submit."):
        if api_key:
            with st.spinner('Wait for it...'):
                # When an API key is provided, display the generated text
                generated_text = generate_text(api_key,  selected_language, selected_question)
                st.write(generated_text)
                st.write("**: Answer Only**")
                st.write(generated_text["answer"])
        else:
            st.warning("Please insert your OpenAI API key.")

    current_file_name = os.path.basename(__file__)
    view_sourcecode(current_file_name)

if __name__ == "__main__":
    main()

LC_QuickStart_02()