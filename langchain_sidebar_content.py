import streamlit as st

def LC_QuickStart_00():
    st.sidebar.header("AI Web App Development 🏠")
    st.sidebar.write('Tool : LangChain, Streamlit, OpenAI, Python, Visual Studio Code, Streamlit Cloud, Github')
    st.sidebar.write('This page is all about sharing what I have found while digging into the tools and methods I need to develop my AI Web App.')
    st.sidebar.header("Links where you can download the necessary tools.")
    st.sidebar.markdown(""" - [Python](https://www.python.org/downloads/) """)
    st.sidebar.markdown(""" - [OpenAI API Key](https://platform.openai.com/api-keys) """)
    st.sidebar.markdown(""" - [Visual Studio Code](https://code.visualstudio.com/download) """)
    st.sidebar.markdown(""" - [LangChain](https://python.langchain.com/docs/get_started/installation) """)
    st.sidebar.markdown(""" - [Streamlit](https://docs.streamlit.io/get-started/installation) """)
    st.sidebar.markdown(""" - [Streamlit Cloud](https://streamlit.io/cloud) """)
    st.sidebar.markdown(""" - [GitHub Docs](https://docs.github.com/en/get-started/start-your-journey/hello-world) """)    
    st.sidebar.markdown(""" - [Streamlit Multipage Template](https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app) """)
    st.sidebar.markdown(""" - [Streamlit API reference](https://docs.streamlit.io/library/api-reference) """)
    st.sidebar.markdown(""" - [Hugging Face](https://huggingface.co/) """)
    st.sidebar.markdown(""" - [Kaggle](https://www.kaggle.com/) """)

def LC_QuickStart_01():
    st.sidebar.header("LangChain QuickStart 01 🧑‍🎨")
    st.sidebar.write('Tool : ChatOpenAI, Langchain, Streamlit, ChatPromptTemplate, StrOutputParser, openai-OpenAIError')
    st.sidebar.write('On this page, you will learn how to build a simple application with LangChain and how to use the most basic and common components of LangChain: prompt templates, models, and output parsers.')
    st.sidebar.header("Items to study in this example")
    st.sidebar.markdown(""" - [LangChain Quickstart](https://python.langchain.com/docs/get_started/quickstart) """)
    st.sidebar.markdown(""" - [ChatOpenAI Python](https://python.langchain.com/docs/integrations/chat/openai) """)
    st.sidebar.markdown(""" - [ChatOpenAI JS](https://js.langchain.com/docs/integrations/chat/openai) """)
    st.sidebar.markdown(""" - [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html) """)
    st.sidebar.markdown(""" - [StrOutputParser API](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html) """)
    st.sidebar.markdown(""" - [OpenAI Error Types](https://help.openai.com/en/articles/6897213-openai-library-error-types-guidance) """)

def LC_QuickStart_02():
    st.sidebar.header("LangChain QuickStart 02 🧐")
    st.sidebar.write('Tools : beautifulsoup4, WebBaseLoader, OpenAIEmbeddings, FAISS, RecursiveCharacterTextSplitter, create_stuff_documents_chain, create_retrieval_chain')
    st.sidebar.write('Retrieval is useful when you have too much data to pass to the LLM directly. You can then use a retriever to fetch only the most relevant pieces and pass those in.')
    st.sidebar.write('In this process, we will look up relevant documents from a Retriever and then pass them into the prompt. A Retriever can be backed by anything - a SQL table, the internet, etc - but in this instance we will populate a vector store and use that as a retriever')
    st.sidebar.header("Items to study in this example")
    st.sidebar.markdown(""" - [LangChain Quickstart](https://python.langchain.com/docs/get_started/quickstart) """)    
    st.sidebar.markdown(""" - [Vector stores](https://python.langchain.com/docs/modules/data_connection/vectorstores) """)
    st.sidebar.markdown(""" - [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/) """)
    st.sidebar.markdown(""" - [WebBaseLoader](https://python.langchain.com/docs/integrations/document_loaders/web_base) """)
    st.sidebar.markdown(""" - [OpenAI Embedding](https://python.langchain.com/docs/integrations/text_embedding/openai) """)
    st.sidebar.markdown(""" - [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss) """)
    st.sidebar.markdown(""" - [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter) """)
    st.sidebar.markdown(""" - [RecursiveCharacterTextSplitter API](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html) """)
    st.sidebar.markdown(""" - [Chains](https://python.langchain.com/docs/modules/chains) """)
    st.sidebar.markdown(""" - [create_stuff_documents_chain API](https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html) """)
    st.sidebar.markdown(""" - [create_retrieval_chain API](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval.create_retrieval_chain.html#) """)

def LC_QuickStart_03():
    st.sidebar.header("LangChain QuickStart 03 🗣️")
    st.sidebar.write('Tools : create_history_aware_retriever, MessagesPlaceholder, HumanMessage, AIMessage')
    st.sidebar.write('The previous chain can only handle single questions, but to accommodate follow-up questions in applications like chat bots, modifications are needed.')
    st.sidebar.write('Two adjustments are crucial')
    st.sidebar.write('1) The retrieval method must consider the entire history, not just the latest input. ')
    st.sidebar.write('2) The final LLM chain should also incorporate the entire history.')
    st.sidebar.header("Items to study in this example")
    st.sidebar.markdown(""" - [LangChain Quickstart](https://python.langchain.com/docs/get_started/quickstart) """)    
    st.sidebar.markdown(""" - [create_history_aware_retriever](https://api.python.langchain.com/en/latest/chains/langchain.chains.history_aware_retriever.create_history_aware_retriever.html) """)
    st.sidebar.markdown(""" - [MessagesPlaceholder](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.MessagesPlaceholder.html) """)
    st.sidebar.markdown(""" - [Types of MessagePromptTemplate](https://python.langchain.com/docs/modules/model_io/prompts/message_prompts) """)
    st.sidebar.markdown(""" - [HumanMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.human.HumanMessage.html) """)
    st.sidebar.markdown(""" - [AIMessage](https://api.python.langchain.com/en/v0.0.339/schema/langchain.schema.messages.AIMessage.html) """)

def LC_QuickStart_04():
    st.sidebar.header("LangChain QuickStart 04 👨‍🔧")
    st.sidebar.write('Tools : hub, create_openai_functions_agent, AgentExecutor, StreamlitCallbackHandler, get_openai_callback')
    st.sidebar.write('Agents in LangChain are systems that use a language model to interact with other tools. They can be used for tasks such as grounded question/answering, interacting with APIs, or taking action. LangChain provides: A standard interface for agents.')
    st.sidebar.header("Items to study in this example")
    st.sidebar.markdown(""" - [LangChain Quickstart](https://python.langchain.com/docs/get_started/quickstart) """)    
    st.sidebar.markdown(""" - [LangChain Hub](https://docs.smith.langchain.com/cookbook/hub-examples) """)
    st.sidebar.markdown(""" - [LangChain Agents](https://python.langchain.com/docs/modules/agents/) """)
    st.sidebar.markdown(""" - [create_openai_functions_agent](https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent) """)
    st.sidebar.markdown(""" - [create_openai_functions_agent API](https://api.python.langchain.com/en/latest/agents/langchain.agents.openai_functions_agent.base.create_openai_functions_agent.html) """)
    st.sidebar.markdown(""" - [LangChain AgentExecutor API](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html) """)
    st.sidebar.markdown(""" - [StreamlitCallbackHandler API](https://api.python.langchain.com/en/latest/callbacks/langchain_community.callbacks.streamlit.streamlit_callback_handler.StreamlitCallbackHandler.html) """)
    st.sidebar.markdown(""" - [get_openai_callback](https://python.langchain.com/docs/modules/model_io/llms/token_usage_tracking) """)
    st.sidebar.markdown(""" - [get_openai_callback API](https://api.python.langchain.com/en/v0.0.341/callbacks/langchain.callbacks.manager.get_openai_callback.html) """)

def LC_QuickStart_08():
    st.sidebar.header("OpenAI Assistants API 🤵🏻‍♂️")
    st.sidebar.write('Tool : OpenAI Assistants API, Jocoding Youtube, TeddyNote blog')
    st.sidebar.write('This page contains a brief description of the beta version of OpenAI\'s Assistants API.')
    st.sidebar.header("Links where you can download the necessary tools.")
    st.sidebar.markdown(""" - [OpenAI Assistants API Overview](https://platform.openai.com/docs/assistants/overview?context=with-streaming) """)
    st.sidebar.markdown(""" - [OpenAI API Key](https://platform.openai.com/api-keys) """)
    st.sidebar.markdown(""" - [DevDay Announcement](https://coronasdk.tistory.com/1496) """)
    st.sidebar.markdown(""" - [Example (Galileo)](https://cdn.openai.com/new-models-and-developer-products-announced-at-devday/assistants-playground.mp4 ) """)
    st.sidebar.markdown(""" - [Assistants API Help Page](https://help.openai.com/en/articles/8550641-assistants-api) """)
    st.sidebar.markdown(""" - [Streamlit Cloud](https://platform.openai.com/docs/api-reference/assistants-streaming/events ) """)
    st.sidebar.markdown(""" - [Assistant stream events (status)](https://docs.github.com/en/get-started/start-your-journey/hello-world) """)    
    st.sidebar.markdown(""" - [Assistant Support Files](https://platform.openai.com/docs/assistants/tools/supported-files) """)
    st.sidebar.markdown(""" - [Assistants Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/Assistants_API_overview_python.ipynb) """)
    st.sidebar.markdown(""" - [TeddyNote Blog](https://teddylee777.github.io/openai/openai-assistant-tutorial/ ) """)
    st.sidebar.markdown(""" - [JoCoding Youtube](https://youtu.be/LdYb356GXeI?si=u_LpHRISWeCkkgWn) """)

def OpenAI_AssistantsAPI_Function():
    st.sidebar.header("OpenAI Assistants API Function 📢")
    st.sidebar.write('Tool : OpenAI Assistants API, Requests HTTP Library, JSON, freeCodeCamp.org, newsapi.org')
    st.sidebar.write('This page shows an example using the Functions tool from OpenAI\'s Assistants API. If you give a Topic, it retrieves related news from newsapi.org and provides a summary of the article.')
    st.sidebar.header("Links where you can download the necessary tools.")
    st.sidebar.markdown(""" - [OpenAI Assistants API Overview](https://platform.openai.com/docs/assistants/overview?context=with-streaming) """)
    st.sidebar.markdown(""" - [OpenAI API Key](https://platform.openai.com/api-keys) """)
    st.sidebar.markdown(""" - [newsapi.org](https://newsapi.org/) """)
    st.sidebar.markdown(""" - [Original Source code](https://github.com/pdichone/vincibits-news-aggregator/blob/main/app.py ) """)
    st.sidebar.markdown(""" - [Youtube Reference](https://youtu.be/qHPonmSX4Ms?si=GH99-vGzx-kJ6epq) """)
    st.sidebar.markdown(""" - [JSON module](https://docs.python.org/3/library/json.html) """)
    st.sidebar.markdown(""" - [Requests HTTP Library](https://pypi.org/project/requests/) """)    
