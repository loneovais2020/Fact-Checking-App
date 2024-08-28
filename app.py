import streamlit as st
import os
import cohere
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate


class TavilySearchInput(BaseModel):
    query: str = Field(description="Internet query engine.")


# Function to generate a question based on user input
def generate_question(user_fact):
    co = cohere.Client()
    # co = cohere.Client(cohere_api_key=api_key)
    response = co.chat(
        message=f"""
    You are a bot that formulates a question based on user input. You will be given user input and you have to return the question that can be used on internet to retrieve info related to this user query.
    for example:- 
    user input: 'python is a programming language'
    resposne: 'what is python?'
    The user input is {user_fact}
    """,
    )
    return response.text

# Function to perform fact-checking
def fact_check(user_input, context):
    # co = cohere.Client(cohere_api_key  = api_key)
    co = cohere.Client()
    message = f"""
    You are a fact-checking assistant. Your task is to determine whether the user's input is factual or fake based on the provided web search results. Follow these steps:
    1. Carefully read the user's input.
    2. Analyze the web search results provided.
    3. Compare the user's input with the information from the web search.
    4. Determine if the user's input is factual or fake.
    5. If the input is factual, confirm this and provide supporting details from the web search.
    6. If the input is fake or misleading, explain why it's considered fake and provide the correct information from the web search.
    Remember to:
    - Be objective and rely solely on the provided web search results.
    - Cite specific parts of the web search text to support your conclusion.
    - If the web search doesn't provide enough information to make a determination, state that there's insufficient data to fact-check the claim.
    - Avoid using external knowledge not present in the user input or web search results.
    User input: {user_input}
    Web search results: {context}
    Based on the above, please analyze whether the user's input is factual or fake, and provide your reasoning.
    """
    response = co.chat(message=message)
    return response.text

# Initialize session state for API keys
if 'cohere_api_key' not in st.session_state:
    st.session_state['cohere_api_key'] = ''
if 'tavily_api_key' not in st.session_state:
    st.session_state['tavily_api_key'] = ''

# Streamlit app
def main():
    st.title("Fact-Checking App")

    # Sidebar for API key input
    st.sidebar.header("API Keys")
    cohere_api_key = st.sidebar.text_input("Enter Cohere API Key", type="password", value=st.session_state['cohere_api_key'])
    tavily_api_key = st.sidebar.text_input("Enter Tavily API Key", type="password", value=st.session_state['tavily_api_key'])

    # Submit button for API keys
    if st.sidebar.button("Save API Keys"):
        st.session_state['cohere_api_key'] = cohere_api_key
        st.session_state['tavily_api_key'] = tavily_api_key
        st.sidebar.success("API keys saved successfully!")
        os.environ["COHERE_API_KEY"] = st.session_state['cohere_api_key']
        os.environ["TAVILY_API_KEY"] = st.session_state['tavily_api_key']

    

    # Main app area
    user_input = st.text_area("Enter the text you want to fact-check:")
    
    if st.button("Fact Check"):
        if not st.session_state['cohere_api_key'] or not st.session_state['tavily_api_key']:
            st.error("Please enter and save both API keys in the sidebar first.")
        elif not user_input:
            st.error("Please enter some text to fact-check.")
        else:
            try:
                with st.spinner("Fact-checking in progress..."):
                    # Set up the tools and agent
                    chat = ChatCohere(model="command-r-plus", temperature=0.7, api_key=st.session_state['cohere_api_key'])
                    internet_search = TavilySearchResults(api_key=st.session_state['tavily_api_key'])
                    internet_search.name = "internet_search"
                    internet_search.description = "Returns a list of relevant documents from the internet."

                    internet_search.args_schema = TavilySearchInput

                    # Generate question and perform search
                    question = generate_question(user_input)

                    prompt = ChatPromptTemplate.from_template("{input}")
                    agent = create_cohere_react_agent(
                    llm=chat,
                    tools=[internet_search],
                    prompt=prompt,)

                    agent_executor = AgentExecutor(agent=agent, tools=[internet_search], verbose=True)

                    response = agent_executor.invoke({"input": question})

                    # Perform fact-checking
                    fact_check_response = fact_check(user_input, response["output"])

                    # Display results
                    st.subheader("Fact-Checking Results:")
                    st.write(fact_check_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check your API keys and try again.")

if __name__ == "__main__":
    main()