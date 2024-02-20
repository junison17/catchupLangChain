import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAIError
from my_modules import view_sourcecode, modelName
import os
from langchain_community.callbacks import get_openai_callback
from langchain_sidebar_content import LC_QuickStart_01

def createChain(llm, output_parser):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a caring teacher who answers students' questions."),
    ("user", "{input}")
    ])
    if output_parser:
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser   
    else:
        chain = prompt | llm  

    return chain

# Function to interact with OpenAI API
def generate_text(api_key, input_text, whatToAsk, language):
    try: 
        model_name = modelName()
         # Initialize your OpenAI instance using the provided API key
        llm = ChatOpenAI(openai_api_key=api_key,model_name=model_name )
        if (whatToAsk == 'Basic'):
            st.write("- *Simpleast way to use LLM.*")
            generated_text = llm.invoke("Please answer for this question. in " + language + ". " + input_text )
        elif (whatToAsk == 'ChatPromptTemplate'):
            output_parser = False
            chain = createChain(llm, output_parser)
            st.write("- *Prompt templates are used to convert raw user input to a better input to the LLM.*")
            generated_text = chain.invoke({"input": "Please answer for this question. in " + language + ". " + input_text})
        else:
            output_parser = True
            chain = createChain(llm, output_parser)
            st.write("- *The output of a ChatModel (and therefore, of this chain) is a message. However, it's often much more convenient to work with strings.*")
            with get_openai_callback() as cb:
                generated_text = chain.invoke({"input": "Please answer for this question. in " + language + ". " + input_text})
                st.write(cb)

        return generated_text
    except OpenAIError as e:
        st.warning("Incorrect API key provided or OpenAI API error.")
        st.warning(e)

def main():
    st.title('LangChain Quickstart 01')

    # Get user input for OpenAI API key
    api_key = st.text_input("Please input your OpenAI API Key:", type="password")

    # Get user input for topic of the poem
    input_text = st.text_input('Throw a question, please!')
    st.write('The topic of the poem is ', input_text)

    whatToAsk = st.radio(
    "Please choose one way to ask an LLM question from the list below.",
    ["Basic", "ChatPromptTemplate", "StrOutputParser"],
    captions = ["Simplest way.", "Use ChatPromptTemplate with chain.", "Add StrOutputParser to the chain"])

    # List of languages available for ChatGPT
    available_languages = ["English", "Korean", "Spanish", "French", "German", "Chinese", "Japanese"]

    # User-selected language
    selected_language = st.selectbox("Select a language:", available_languages)  

    # Button to trigger text generation
    if st.button("Submit."):
        if api_key:
            with st.spinner('Wait for it...'):
                # When an API key is provided, display the generated text
                generated_text = generate_text(api_key, input_text, whatToAsk, selected_language)
                st.write(generated_text)
        else:
            st.warning("Please insert your OpenAI API key.")

    current_file_name = os.path.basename(__file__)
    view_sourcecode(current_file_name)

if __name__ == "__main__":
    main()

LC_QuickStart_01()