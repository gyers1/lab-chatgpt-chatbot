from __future__ import annotations

from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()

st.set_page_config(page_title="Mini Chat", page_icon="💬")
st.title("Mini Chat 💬")

SYSTEM = "당신은 친절한 AI 어시스턴트입니다."
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    msgs = [SystemMessage(content=SYSTEM)]
    for message in st.session_state.messages:
        cls = HumanMessage if message["role"] == "user" else AIMessage
        msgs.append(cls(content=message["content"]))

    with st.chat_message("assistant"):
        answer = llm.invoke(msgs).content or ""
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
