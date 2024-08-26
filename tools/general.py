import streamlit as st
from llm import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

__chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert providing information about movies."),
        ("human", "{input}"),
    ]
)

general_chat = __chat_prompt | llm | StrOutputParser()