import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAIError
from my_modules import view_sourcecode, modelName, modelName_embedding_small
import os
from langchain_community.callbacks import get_openai_callback
from langchain_sidebar_content import LC_QuickStart_03

# Function to interact with OpenAI API
def generate_text(api_key,  selected_3rd_question, selected_4th_question):
    try: 
        openai_api_key = api_key
        embedding_model_name = modelName_embedding_small()
        model_name = modelName()

        st.write("*** Work Process ***")

        # 1. Get Data
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader("https://docs.smith.langchain.com")
        docs = loader.load()
        st.write("1. Get data from the Webpage.")

        # 2. Set Embedding model
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model_name)
        st.write("2. Set Embedding Model. (Model Name is " + embedding_model_name + ")")

        # 3. Store vector into vector storage
        from langchain_community.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        st.write("3. Split text and store as vector using FAISS.")

        # 4. Take most recent input and chat_history
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_community.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains import create_history_aware_retriever
        from langchain_core.prompts import MessagesPlaceholder

        llm = ChatOpenAI(openai_api_key=openai_api_key,model_name=model_name )
        retriever = vector.as_retriever()

        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        st.write("4. Take most recent input and chat_history and Create a chain. (create_history_aware_retriever(llm, retriever, prompt))")

        # 5. test this out by passing in an instance where the user is asking a follow up question.
        from langchain_core.messages import HumanMessage, AIMessage

        chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
        test_follwUpQ = retriever_chain.invoke({
            "chat_history": chat_history,
            "input": "Tell me how"
        })
        st.write("5. Result value (Documents) received by invoking create_history_aware_retriever(llm, retriever, prompt).")
        st.write(test_follwUpQ)

        # 6. creating a new chain to continue the conversation with these retrieved documents in mind
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(llm, prompt)

        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
        st.write("6. created a new chain to continue the conversation with these retrieved documents in mind. \ncreate_retrieval_chain(retriever_chain, document_chain)")

        # 7. test this out end-to-end
        chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
        generated_text_2nd = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": "Tell me how"
        })
        st.write("7. Invoke above chain (2nd Question) and receive the result from LLM.")
        st.write("chat_history is ",  generated_text_2nd["chat_history"])
        st.write("question is :blue[", generated_text_2nd["input"],"]")
        st.write("answer is :blue[",  generated_text_2nd["answer"],"]")

        chat_history = chat_history + [HumanMessage(content=generated_text_2nd["input"]), AIMessage(content=generated_text_2nd["answer"])]
        generated_text_3rd = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": selected_3rd_question
        })
        st.write("8. Invoke above chain (3rd Question) and receive the result from LLM.")
        st.write("chat_history is " , generated_text_3rd["chat_history"])
        st.write("question is :blue[" , generated_text_3rd["input"],"]")
        st.write("answer is :blue[" , generated_text_3rd["answer"],"]")

        chat_history = chat_history + [HumanMessage(content=generated_text_3rd["input"]), AIMessage(content=generated_text_3rd["answer"])]
        question_4th = "Is " + selected_4th_question + " one of them?"
        generated_text = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": question_4th
        })
        st.write("8. Invoke above chain (4th Question) and receive the result from LLM.")

        # 8. Delete the vector
        vector.delete([vector.index_to_docstore_id[0]])
        # Is now missing
        0 in vector.index_to_docstore_id

        return generated_text
    except OpenAIError as e:
        st.warning("Incorrect API key provided or OpenAI API error.")
        st.warning(e)

def main():
    st.title('LangChain Quickstart 03 - :blue[Conversation Retrieval Chain]')

    # Get user input for OpenAI API key
    api_key = st.text_input("Please input your OpenAI API Key:", type="password")
    st.write("Fetching this Web Page Contents : https://docs.smith.langchain.com")

    st.subheader("Scenario")
    st.write("1. The first question and answer are set in advance as follows.")
    st.caption("Question : ***:blue[Can LangSmith help test my LLM applications?]*** Answer : ***:green[Yes!]***")

    st.write("2. The second one asks a question that can be answered only if LLM knows above first Q&A.")
    st.caption("Question : ***:blue[Tell me how?]***, ***:green[Check the answer after Submit all the scenarios]***")

    # List of Questions
    quastions_3rd = ["How many items in Next Steps?", 
                "How many items in Additional Resources?"]

    st.write("3. Thirdly, the question selected by you from the dropdown below will be delivered.")
    # User-selected question
    selected_3rd_question = st.selectbox("Select a question: ", quastions_3rd)   

    st.write("4. The 4th one asks whether the item selected by you belongs to the 3rd question category.")
    # List of Questions
    quastions_4th = ["User Guide", "Setup", "Pricing", "Self-Hosting", "Tracing", "Evaluation", "Prompt Hub", 
                     "LangSmith Cookbook", "LangChain Python", "LangChain Python API Reference", "LangChain JS", "Discord" ]

    # User-selected question
    selected_4th_question = st.selectbox("select an item (*:blue[LLM should know the 3rd question and the answer above to answer it.]*):", quastions_4th)   

    # Button to trigger text generation
    if st.button("Submit."):
        if api_key:
            with st.spinner('Wait for it...'):
                # When an API key is provided, display the generated text
                generated_text = generate_text(api_key,  selected_3rd_question, selected_4th_question)
                st.write(generated_text)
                st.write("chat_history is " , generated_text["chat_history"])
                st.write("question is :blue[" , generated_text["input"],"]")
                st.write("answer is :blue[" , generated_text["answer"],"]")
        else:
            st.warning("Please insert your OpenAI API key.")

    current_file_name = os.path.basename(__file__)
    view_sourcecode(current_file_name)

if __name__ == "__main__":
    main()

LC_QuickStart_03()