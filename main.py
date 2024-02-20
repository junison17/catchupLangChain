import streamlit as st
from my_modules import adsense_ads

st.set_page_config(
    page_title="Catchup LangChain Tutorial",
    page_icon="👋",
)

st.write("# Catchup LangChain Tutorial!👋")

st.sidebar.success("Select a demo above.")
st.sidebar.markdown(""" - [LangChain Introduction](https://python.langchain.com/docs/get_started/introduction) """)
st.sidebar.markdown(""" - [LangChain Installation](https://python.langchain.com/docs/get_started/installation) """)

st.markdown(
    """
    LangChain is a framework for developing applications powered by language models. It enables applications that:
    \n**Are context-aware**: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
    \n**Reason**: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)
"""
)

st.markdown(
    """
    This framework consists of several parts.

    \n**LangChain Libraries**: The Python and JavaScript libraries. Contains interfaces and integrations for a myriad of components, a basic run time for combining these components into chains and agents, and off-the-shelf implementations of chains and agents.
    \n**LangChain Templates**: A collection of easily deployable reference architectures for a wide variety of tasks.
    \n**LangServe**: A library for deploying LangChain chains as a REST API.
    \n**LangSmith**: A developer platform that lets you debug, test, evaluate, and monitor chains built on any LLM framework and seamlessly integrates with LangChain.
"""
)

st.image('./images/langchain_structure.jpg', caption='LangChain Structure')

st.markdown(
    """
    - [Catchup AI Streamlit Web App](https://catchupai.streamlit.app/)
    - [Catchup AI Youtube Channel](https://www.youtube.com/@catchupai)
    - [Catchup AI Tistory Blog](https://coronasdk.tistory.com/)
    - [Deep Learning Fundamental PPT (Eng. Ver.)](https://docs.google.com/presentation/d/1F4qxSAv9g13de99rS8fcp4e1LCfrILq8QaahXCPx1Pw/edit?usp=sharing)
    - [Deep Learning Fundamental PPT (Kor. Ver.)](https://docs.google.com/presentation/d/15KNzGnSnJx_4ToSBM2MrHiC2q5MiVe0plOs7f3NJuWM/edit?usp=sharing)
    - [AI Web App Development 101 PPT](https://docs.google.com/presentation/d/18_6z05tmR_loTPWFHj8PCY3-uCNKuoy-IvE0g5ms8YM/edit?usp=sharing)
"""
)

adsense_ads()
