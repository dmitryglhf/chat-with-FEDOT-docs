import os
import streamlit as st
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings


st.set_page_config(
    page_title="FEDOT Docs",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

GOOGLE_API_KEY = st.secrets.genai_key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

genai.configure(api_key=st.secrets.genai_key)
st.title("Chat with the FEDOT docs")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about FEDOT AutoML Framework!",
        }
    ]

SYS_PROMPT = """You are an expert on 
        the FEDOT Python Framework and your 
        job is to answer technical questions. 
        Assume that all questions are related 
        to the FEDOT Python library. Keep 
        your answers technical and based on 
        facts – do not hallucinate features.
        It is preferable to present code examples."""


def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    Settings.llm = None
    Settings.llm = Gemini(
        model="models/gemini-1.5-flash",
        api_key=st.secrets.genai_key,
        system_prompt=SYS_PROMPT,
    )
    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        verbose=True,
        streaming=True,
        llm=Settings.llm,
        systemp_prompt=SYS_PROMPT
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
