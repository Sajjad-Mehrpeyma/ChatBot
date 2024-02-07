import streamlit as st
import time
from utils import QA


def UI():
    st.title("ChatBot")
    if "messages" not in st.session_state:
        st.session_state.messages = []


    if prompt := st.chat_input("Hi! Ask your question 8)"):
        # appending user question to show list
        st.session_state.messages.append({"role": "user", "content": prompt})
        # showing user questions and Bot answers
        for message in st.session_state.messages:
            role = message["role"]
            with st.chat_message(role):
                st.markdown(message["content"])

        with st.chat_message("assistant"):
            full_response = ""

            assistant_response = QA(prompt)
            answer = assistant_response[0]
            confidence = assistant_response[1]

            message_placeholder = st.empty()
            for chunk in answer.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            full_response += f"\n\n Confidence: {confidence}"
            message_placeholder.markdown(full_response)

        # appending Bot answer to show list
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})


if __name__ == '__main__':
    UI()
