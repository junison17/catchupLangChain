import streamlit as st
import streamlit.components.v1 as components
from langchain_sidebar_content import LC_QuickStart_00
from my_modules import view_sourcecode
import os

st.title('AI Web App Structure and tools')
st.write('')
st.write('For bigger screen, click the link below.')
st.markdown(""" - [Google Slide Src](https://docs.google.com/presentation/d/18_6z05tmR_loTPWFHj8PCY3-uCNKuoy-IvE0g5ms8YM/edit?usp=sharing) """)
st.write('')

# embed streamlit docs in a streamlit app
components.iframe("https://docs.google.com/presentation/d/18_6z05tmR_loTPWFHj8PCY3-uCNKuoy-IvE0g5ms8YM/edit?usp=sharing", height =1000, width = 1500)

current_file_name = os.path.basename(__file__)
view_sourcecode(current_file_name)

LC_QuickStart_00()