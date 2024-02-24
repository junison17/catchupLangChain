import streamlit as st
from openai import OpenAIError
from my_modules import view_sourcecode, modelName, modelName_embedding_small
import os
from langchain_community.callbacks import get_openai_callback
from langchain_sidebar_content import LC_QuickStart_04

# Function to interact with OpenAI API
def generate_text(openai_api_key, tavily_api_key,  selected_q, selected_language):
    try: 
        openai_api_key = openai_api_key
        embedding_model_name = modelName_embedding_small()
        model_name = modelName()

        st.write("*** Work Process ***")

        # 1. Get Data
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader("https://docs.smith.langchain.com")
        docs = loader.load()
        st.write("1. Get data from the Webpage. (Create retriever tool step 1)")

        # 2. Set Embedding model
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model_name)
        st.write("2. Set Embedding Model. (Create retriever tool step 2)")

        # 3. Store vector into vector storage
        from langchain_community.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        st.write("3. Split text and store as vector using FAISS. (Create retriever tool step 3)")

        # 4. Set the vector as retriever
        retriever = vector.as_retriever()
        st.write("4. Set the vector as retrieve. (Create retriever tool step 4)")

        # 5. Set up a tol for the retriever above
        from langchain.tools.retriever import create_retriever_tool

        retriever_tool = create_retriever_tool(
            retriever,
            "langsmith_search",
            "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
        )
        st.write("5. Set retriever tool. (Complete retriever tool set up.)")

        # 6. Prepare to use Tavily Search Engine
        os.environ['TAVILY_API_KEY'] = tavily_api_key
        from langchain_community.tools.tavily_search import TavilySearchResults
        search = TavilySearchResults()
        st.write("6. Set Tavily search tool.")

        # 7. Create a list of tools we want to work
        tools = [retriever_tool, search]
        st.write("7. Set both retriever_tool and search tool as tools")

        # 8. Create an Agent
        from langchain_openai import ChatOpenAI
        from langchain import hub
        from langchain.agents import create_openai_functions_agent
        from langchain.agents import AgentExecutor

        # 9. Set Agent and Agent_executor
        from langchain_community.callbacks import StreamlitCallbackHandler
        prompt = hub.pull("hwchase17/openai-functions-agent") #https://smith.langchain.com/hub/hwchase17/openai-functions-agent
        llm = ChatOpenAI(openai_api_key=openai_api_key,model_name=model_name, temperature=0 )
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        st.write("8. Set Agent and Agent_executor.")

        st_callback = StreamlitCallbackHandler(st.container())
        # streamlit callback management : https://python.langchain.com/docs/integrations/callbacks/streamlit

        # 10. invoke agent_executer to LLM with questions.
        with get_openai_callback() as cb:
            generated_text = agent_executor.invoke({"input": selected_q + " and please answer in English and translate it into " + selected_language + " as well."}, 
                                                   {"callbacks": [st_callback]})
            st.write(cb)
        st.write("9. invoke agent_executer to LLM with questions.")

        # 8. Delete the vector
        vector.delete([vector.index_to_docstore_id[0]])
        # Is now missing
        0 in vector.index_to_docstore_id

        return generated_text
    except OpenAIError as e:
        st.warning("Incorrect API key provided or OpenAI API error.")
        st.warning(e)

def main():
    st.title('LangChain Quickstart 04 - :blue[Agent]')

    st.subheader("The scenario on this page is:")
    st.write("This example shows how to use an Agent.")
    st.write("The first thing to do is to create tools for the Agent to use.")

    st.write("For this example, we'll use two tools: Retriever, which we created to answer questions about LangSmith that we used in the previous step, and the Tavily search tool, which is a new tool.")
    st.caption("To use Tavily, you need a Tavily API Key. Visit https://docs.tavily.com/ for more information.")

    st.write("When these two tools are registered with the Agent and then invoked, the Agent determines which tool should be used to answer the question, obtains the necessary information using the tool, and then invokes the question and the obtained information together with the LLM.")
    st.write("")
    st.subheader("***:blue[Enter the input values below and click Submit.]***")    

    # Get user input for OpenAI API key
    openai_api_key = st.text_input("Please input your OpenAI API Key:", type="password")
    st.write("Fetching this Web Page Contents : https://docs.smith.langchain.com")

    # Get user input for OpenAI API key
    tavily_api_key = st.text_input("Please input your TAVILY API Key:", type="password")  

    # List of Questions
    langsmith_question_list = ["Can LangSmith help test my LLM applications?", 
                               "How can langsmith help with testing?",
                               "How many items in Next Steps in LangSmith?",
                               "How many items in Additional Resources in LangSmith?",
                               "Please tell me the additional Resources in LangSmith.",
                               "What are the Next Steps in LangSmith?",
                               ]
    
    select_question = st.radio(
    "Please choose the question you'd like to ask.",
    ["about LangSmith", "about Daily life"])

    if select_question == 'about LangSmith':
        # User-selected question
        selected_q = st.selectbox("Select a question about LangSmith: ", langsmith_question_list)   
    else:
        # Daily life question.
        selected_q = st.text_input('Please ask questions related to your everyday life. e.g. What is the weather in Seattle?')

    st.write("*Answers will be in English and the language of your choice.* ")  

    # List of languages available for ChatGPT
    available_languages = ["Korean", "Spanish", "French", "German", "Chinese", "Japanese"]

    # User-selected language
    selected_language = st.selectbox("Select a language:", available_languages) 

    # Button to trigger text generation
    if st.button("Submit."):
        if openai_api_key and tavily_api_key:
            with st.spinner('Wait for it...'):
                # When an API key is provided, display the generated text
                generated_text = generate_text(openai_api_key, tavily_api_key,  selected_q, selected_language)
                st.write(generated_text)
        else:
            st.warning("Please insert your OpenAI API key or Tavily API key.")

    current_file_name = os.path.basename(__file__)
    view_sourcecode(current_file_name)

if __name__ == "__main__":
    main()

LC_QuickStart_04()