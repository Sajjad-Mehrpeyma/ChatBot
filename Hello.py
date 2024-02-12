import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)


def intro():

    st.write("# Welcome to my ChatBot! 👋")
    st.sidebar.success("Select a task above.")

    st.markdown(
        """
        I develope and maintain this chatbot and its functionality.
        ### Want to see codes? 
        They are available open source at:
        - https://github.com/Sajjad-Mehrpeyma/ChatBot

    """
    )


if __name__ == '__main__':
    intro()
