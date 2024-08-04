import streamlit as st
import os
import openai
import requests
import json
import time
from langchain_sidebar_content import OpenAI_AssistantsAPI_Function
from my_modules import view_sourcecode, modelName, modelName4o

def get_news(news_api_key, topic):
    url = (
        f"https://newsapi.org/v2/everything?q={topic}&apiKey={news_api_key}&pageSize=5"
    )
    try:
        response = requests.get(url)

        if response.status_code == 200:
            news = json.dumps(response.json(), indent=4)
            # Convert JSON string to a Python dictionary
            news_json = json.loads(news)
            data = news_json

            # Accessing individual fields
            status = data["status"]
            total_results = data["totalResults"]
            articles = data["articles"]

            final_news = []

            # Loop through articles
            for article in articles:
                source_name = article["source"]["name"]
                author = article["author"]
                title = article["title"]
                description = article["description"]
                url = article["url"]
                published_at = article["publishedAt"]
                content = article["content"]
                title_description = f"""Title: {title}, \n Author: {author}, \n Source: {source_name}, 
                 \n description: {description}  \n URL: {url}, \n Content: {content}"""

                final_news.append(title_description)

            return final_news
        else:
            return []

    except requests.exceptions.RequestException as e:
        print("Error occured during API Rquest:", e)


class AssistantManager:
    # Static variables to store the thread and assistant IDs

    thread_id = None
    assistant_id = None

    def __init__(self):
        self.client = None
        self.model = None
        self.assistant = None
        self.thread = None
        self.run = None

        # Add later for streamlit
        self.summary = (
            None  # Add an instance variable to store the summary for streamlit
        )
    def setAssistantThreadIDs(self,assistant_id,thread_id, openai_api_key, select_model):
        self.client = openai.OpenAI(api_key = openai_api_key)
        self.model = modelName() if select_model == "Cheapest" else modelName4o()        
        AssistantManager.assistant_id = assistant_id
        AssistantManager.thread_id = thread_id

        # Retrieve existing assistant and thread if IDs are already set
        if AssistantManager.assistant_id:
            self.assistant = self.client.beta.assistants.retrieve(
                AssistantManager.assistant_id
            )
        if AssistantManager.thread_id:
            self.thread = self.client.beta.threads.retrieve(AssistantManager.thread_id)
        st.write("3. Set Assistant ID and Thread ID")

    def create_assistant(self, openai_api_key, select_model, name, instructions, tools):
        self.client = openai.OpenAI(api_key = openai_api_key)
        self.model = modelName() if select_model == "Cheapest" else modelName4o()
        if not self.assistant:
            assistant_obj = self.client.beta.assistants.create(
                name=name, instructions=instructions, tools=tools, model=self.model
            )
            AssistantManager.assistant_id = assistant_obj.id
            self.assistant = assistant_obj
            st.write("1. Create Assistant")
            st.subheader(f":blue[AssisID: {self.assistant.id}]")

    def create_thread(self):
        if not self.thread:
            thread_obj = self.client.beta.threads.create()
            AssistantManager.thread_id = thread_obj.id
            self.thread = thread_obj
            st.write("2. Create Thread")
            st.subheader(f":blue[ThreadID: {self.thread.id}]")

    def add_message_to_thread(self, role, content):
        if self.thread:
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role=role,
                content=content,
            )
        st.write("4. Add Message to the Thread.")

    def run_assistant(self, instructions):
        if self.thread and self.assistant:
            self.run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                instructions=instructions,
            )
        st.write("5. Run the Assistant.")

    def process_messages(self):
        if self.thread:
            messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
            summary = []
            # just get the last message of the thread
            last_message = messages.data[0]
            role = last_message.role
            response = last_message.content[0].text.value
            print(f"SUMMARY: {role.capitalize()}: ==> {response}")
            summary.append(response)
            self.summary = "\n".join(summary)

            st.write("10. Message has been processed")

            # loop through all messages in this thread
            # for msg in messages.data:
            #     role = msg.role
            #     content = msg.content[0].text.value
            #     print(f"SUMMARY:: {role.capitalize()}: {content}")

    def wait_for_completion(self, newsapi_key):
        if self.thread and self.run:
            while True:
                time.sleep(3)
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=self.run.id,
                )

                print(f"RUN STATUS: {run_status.model_dump_json(indent=4)}")

                if run_status.status == "completed":
                    st.write("9. status is completed. process_messages() calling now....")
                    self.process_messages()
                    break
                elif run_status.status == "requires_action":
                    st.write("6. status is requires_action. Function calling now....")
                    self.call_required_functions(
                        run_status.required_action.submit_tool_outputs.model_dump(), newsapi_key
                    )
                else:
                    st.write("8. Waiting for the Assistant to process...")

    # for streamlit
    def get_summary(self):
        st.write('11. return self.summary for streamlit')
        return self.summary

    # Run the steps
    def run_steps(self):
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.thread.id, run_id=self.run.id
        )
        print(f"Run-Steps: {run_steps}")
        st.write("12. Display Run_steps.")
        return run_steps.data

    def call_required_functions(self, required_actions, newsapi_key):
        if not self.run:
            return

        tool_outputs = []

        for action in required_actions["tool_calls"]:
            func_name = action["function"]["name"]
            arguments = json.loads(action["function"]["arguments"])

            if func_name == "get_news":
                output = get_news(newsapi_key, topic=arguments["topic"])
                print(f"STUFF ===> {output}")
                final_str = ""
                for item in output:
                    final_str += "".join(item)

                tool_outputs.append({"tool_call_id": action["id"], "output": final_str})
            else:
                raise ValueError(f"Unknown function: {func_name}")

        st.write("7. Submitting outputs back to the Assistant... - call_required_functions")

        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=self.run.id,
            tool_outputs=tool_outputs,
        )


def main():
    manager = AssistantManager()

    # Streamlit interface
    st.title("News Summarizer")

    openai_api_key = st.text_input("Please input your OpenAI API Key:", type="password")
    st.markdown(""" - [Get OpenAI API Key](https://platform.openai.com/api-keys) """)

    # User-selected language
    select_model = st.radio("Please choose the Model you'd like to use.", ["Cheapest", "GPT 4o"]) 

    st.caption("First, create an Assistant and get the Assistant ID and Thread ID.")
    st.caption("Then use the Assistant to ask ChatGPT a question and get a response.")

    create_assistants = st.radio(
    "Please choose the task you want to proceed.",
    ["Create Assistant", "Use the Assistant"])

    if create_assistants == 'Create Assistant':
        create_assistant_btn = st.button(label="Click this button to Get Assistant ID and Thread ID")   
        if create_assistant_btn:
            if openai_api_key:
                with st.spinner('Wait for it...'):
                    # Create the assistant and thread if they don't exist
                    manager.create_assistant(
                        openai_api_key,
                        select_model,
                        name="News Summarizer",
                        instructions="You are a personal article summarizer Assistant who knows how to take a list of article's titles and descriptions and then write a short summary of all the news articles",
                        tools=[
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_news",
                                    "description": "Get the list of articles/news for the given topic",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "topic": {
                                                "type": "string",
                                                "description": "The topic for the news",
                                            }
                                        },
                                        "required": ["topic"],
                                    },
                                },
                            }
                        ],
                    )
                    manager.create_thread()
            else:
                st.warning("Please insert your OpenAI API key or Newsapi.org API key.")
    else:
        newsapi_key = st.text_input("Please input your newsapi.org API Key:", type="password")
        st.markdown(""" - [Get newsapi.org API Key](https://newsapi.org/) """)
        assistant_id = st.text_input("Enter Assistant ID:")
        thread_id = st.text_input("Enter Thread ID:")

        # Form for user input
        with st.form(key="user_input_form"):
            instructions = st.text_area("Enter topic:")
            submit_button = st.form_submit_button(label="Run Assistant")
        # Handling the button click
        if submit_button:
            if openai_api_key and select_model and newsapi_key and assistant_id and thread_id and instructions:

                manager.setAssistantThreadIDs(
                    assistant_id, thread_id, openai_api_key, select_model
                )
                # Add the message and run the assistant
                manager.add_message_to_thread(
                    role="user", content=f"Summarize the news about {instructions} and print it line by line. If possible, also indicate the URL link."
                )
                manager.run_assistant(instructions=f"Summarize the news about {instructions}")

                # Wait for completion and process messages
                manager.wait_for_completion(newsapi_key)

                summary = manager.get_summary()
                st.write(summary)

                st.text("Run Steps:")
                st.code(manager.run_steps(), line_numbers=True)
            else:
                st.write(st.warning("Please provide all the keys and values (OpenAI API Key, Newsapi key, LLM Model, Assistant ID, Thread ID and Topic)"))

if __name__ == "__main__":
    main()
    
current_file_name = os.path.basename(__file__)
view_sourcecode(current_file_name)
OpenAI_AssistantsAPI_Function()