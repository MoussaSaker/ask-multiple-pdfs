import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'


llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

prompt = 'what team won the soccer world cup in {year}'

conversation_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt)
)

print(conversation_chain("2010"))