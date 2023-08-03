import os
import streamlit as st

from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

st.title("Sales Chatbot")

# Template for the chatbot's prompts
template = """
I want you to act as a sales person.
{history}
Human: {human_input}
Assistant:
"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

# Initialize the chatbot
chatgpt_chain = LLMChain(
    llm=ChatOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY")),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

# Text input for the user's message to the chatbot
user_input = st.text_input("Enter your message:")

if user_input:
    # Get the chatbot's response and print it
    chatbot_response = chatgpt_chain.predict(human_input=user_input)
    st.write(f"Bot: {chatbot_response}")
