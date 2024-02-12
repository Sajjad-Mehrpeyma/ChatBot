import streamlit as st


def intro():
    # import streamlit as st

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
