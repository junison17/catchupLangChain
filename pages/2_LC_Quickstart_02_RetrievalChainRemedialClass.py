import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAIError
import os
from langchain_community.callbacks import get_openai_callback
from my_modules import view_sourcecode, modelName, modelName_embedding_small, modelName4o
from langchain_sidebar_content import LC_QuickStart_02_RemedialClass

# https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846
# https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/vectorstore/
# https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html

# Function to interact with OpenAI API
def generate_text(api_key, language, question, select_model, selected_chunk_size, selected_chunk_overlap, selected_search_type, selected_chunk_num):
    try: 
        openai_api_key = api_key
        embedding_model_name = modelName_embedding_small()
        if(select_model == "Cheapest"):
            model_name = modelName()
        else:
            model_name = modelName4o()

        st.write("*** Work Process ***")

        # 1. Get Data
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
        docs = loader.load()
        st.write("1. Get data from the Webpage.")

        # 2. Set Embedding model
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model_name)
        st.write("2. Set Embedding Model. (Model Name is " + embedding_model_name + ")")

        # 3. Store vector into vector storage
        from langchain_community.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = selected_chunk_size,
            chunk_overlap = selected_chunk_overlap,
            separators=["\n\n", "\n", " "]
        )

        documents = text_splitter.split_documents(docs)
        st.write("Number of Chunks : ", len(documents))
        vector = FAISS.from_documents(documents, embeddings)
 
        st.write("3. Split text and store as vector using FAISS.")

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
        st.write("4. Create Document chain. -create_stuff_documents_chain(llm, prompt)-")

        # 5. Create Retrieval Chain
        from langchain.chains import create_retrieval_chain
        retriever = vector.as_retriever(search_type=selected_search_type, search_kwargs={"score_threshold": 0.2, "k" : selected_chunk_num})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        st.write("5. Create Retrieval Chain. -create_retrieval_chain(retriever, document_chain)-")

        with get_openai_callback() as cb:
            generated_text = retrieval_chain.invoke({"input": question})
            st.write(cb)

        vector.delete([vector.index_to_docstore_id[0]])
        # Is now missing
        # 0 in vector.index_to_docstore_id

        return generated_text
    except OpenAIError as e:
        st.warning("Incorrect API key provided or OpenAI API error.")
        st.warning(e)

def main():
    st.title('LangChain Quickstart 02 - Remedial Class')

    # Get user input for OpenAI API key
    api_key = st.text_input("Please input your OpenAI API Key:", type="password")
    st.write("Fetching this Web Page Contents : https://docs.smith.langchain.com/user_guide")

    select_model = st.radio(
    "Please choose the Model you'd like to use.",
    ["Cheapest", "gpt-4o-2024-05-13"])    

    # List of Questions
    quastions = ["How can langsmith help with testing?", 
                "Please summarize on Prototyping.", 
                "Please tell me about Beta Testing.",
                "Please explain about Production.",
                "Please summarize whole LangSmith User Guide."]

    # User-selected question
    selected_question = st.selectbox("Select a question:", quastions)    

    st.write("*Answers will be in English and the language of your choice.* ")  

    # List of languages available for ChatGPT
    available_languages = ["Korean", "Spanish", "French", "German", "Chinese", "Japanese"]

    # User-selected language
    selected_language = st.selectbox("Select a language:", available_languages)  

    st.write("**- Params for RecursiveCharacterTextSplitter**")

    # List of chunk size for RecursiveCharacterTextSplitter
    available_chunk_size = [5, 10, 50, 200, 500, 1000]    

    # User-selected chunk size
    selected_chunk_size = st.selectbox("Chunk Size for RecursiveCharacterTextSplitter():", available_chunk_size) 

    # List of chunk overlap for RecursiveCharacterTextSplitter
    available_chunk_overlap = [1, 5, 10, 20, 30]    

    # User-selected chunk overlap
    selected_chunk_overlap = st.selectbox("Chunk Overlaps for RecursiveCharacterTextSplitter():", available_chunk_overlap)    

    st.write("**- Params for as_retriever**") 

    # List of search type for as_retriever()
    available_search_type = ["similarity", "mmr", "similarity_score_threshold"]

    # User-selected search type
    selected_search_type = st.selectbox("Select a searcy type for as_retriever():", available_search_type) 

    # List of number of chunks for as_retriever()
    available_chunk_num = [1, 2, 3,4]

    # User-selected language
    selected_chunk_num = st.selectbox("Select a number of chunks for as_retriever():", available_chunk_num)     

    # Button to trigger text generation
    if st.button("Submit."):
        if api_key:
            with st.spinner('Wait for it...'):
                # When an API key is provided, display the generated text
                generated_text = generate_text(api_key,  selected_language, selected_question, select_model, selected_chunk_size, selected_chunk_overlap, selected_search_type, selected_chunk_num)
                st.write(generated_text)
                st.write("**: Answer Only**")
                st.write(generated_text["answer"])
        else:
            st.warning("Please insert your OpenAI API key.")

    current_file_name = os.path.basename(__file__)
    view_sourcecode(current_file_name)

if __name__ == "__main__":
    main()

LC_QuickStart_02_RemedialClass()