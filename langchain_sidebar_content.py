import streamlit as st

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





