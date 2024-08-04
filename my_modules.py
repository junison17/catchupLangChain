import streamlit as st
import os
from my_summaries import get_summary

def modelName():
    modelName = 'gpt-4o-mini-2024-07-18'
    return modelName

def modelName3_5Turbo0125():
    modelName = 'gpt-3.5-turbo-0125'
    return modelName

def modelName4o():
    modelName = 'gpt-4o-2024-05-13'
    return modelName

def modelName_embedding_small():
    modelName = 'text-embedding-3-small'
    return modelName

def display_source_code(fileName):
    # Get the current file path
    current_file_path = os.path.abspath(__file__)

    # Get the path to aaa.py in the 'pages' folder
    pages_file_path = os.path.join(os.path.dirname(current_file_path), "pages", fileName)

    # Open aaa.py and read the source code
    with open(pages_file_path, "r", encoding="utf-8") as file:
        source_code = file.read()

    # Display the source code
    st.code(source_code, language="python")

def view_sourcecode(fileName):
    # Create a session state variable with the key "show_source_code" if it doesn't exist
    if "show_source_code" not in st.session_state:
        st.session_state.show_source_code = False

    # Create a button to toggle the source code display
    if st.button("Toggle Source Code"):
        st.session_state.show_source_code = not st.session_state.show_source_code

    # Display or hide the source code based on the session state
    if st.session_state.show_source_code:
        display_source_code(fileName)

def summary_eng(name):

    # Create a session state variable with the key "show_source_code" if it doesn't exist
    if "show_eng_summary" not in st.session_state:
        st.session_state.show_eng_summary = False

    # Create a button to toggle the source code display
    if st.button("Summary in English"):
        st.session_state.show_eng_summary = not st.session_state.show_eng_summary

    if st.session_state.show_eng_summary:
        st.write(get_summary(name))

def summary_kor(name):

    # Create a session state variable with the key "show_source_code" if it doesn't exist
    if "show_kor_summary" not in st.session_state:
        st.session_state.show_kor_summary = False

    # Create a button to toggle the source code display
    if st.button("Summary in Korean"):
        st.session_state.show_kor_summary = not st.session_state.show_kor_summary

    if st.session_state.show_kor_summary:
        st.write(get_summary(name))   

def simple_source(name):

    # Create a session state variable with the key "show_source_code" if it doesn't exist
    if "simple_source" not in st.session_state:
        st.session_state.simple_source = False

    # Create a button to toggle the source code display
    if st.button("Scripts for local run"):
        st.session_state.simple_source = not st.session_state.simple_source

    if st.session_state.simple_source:
        source_code = get_summary(name)
        st.code(source_code, language="python") 

def adsense_ads():
    # Google AdSense Javascript code
    adsense_script = """
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-3781188318552254"
        crossorigin="anonymous"></script>
    <!-- 2024New -->
    <ins class="adsbygoogle"
        style="display:block"
        data-ad-client="ca-pub-3781188318552254"
        data-ad-slot="6183304357"
        data-ad-format="auto"
        data-full-width-responsive="true"></ins>
    <script>
        (adsbygoogle = window.adsbygoogle || []).push({});
    </script>
    """

    # insert javascript in HTML
    st.markdown(adsense_script, unsafe_allow_html=True)