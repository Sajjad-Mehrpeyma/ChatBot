import streamlit as st
from dotenv import load_dotenv
from pdfExtractor import  extract_text_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(text, chunk_size=600, overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', ' ', ''],
    chunk_size=chunk_size,
    chunk_overlap=overlap,
    length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks


def UI():
    load_dotenv()
    st.title("Chat with multiple PDFs! :books:")
    # st.header('Chat with Multiple PDFs :books:')

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDFs and click 'process'", 
                                type=['pdf'], 
                                accept_multiple_files=True)
        
        if st.button('process'):
            progress_value = 0.0
            progress_bar = st.progress(progress_value, 
                                       text=None)
            with progress_bar and st.spinner('Processing...'):
                docs_texts = extract_text_pdf(docs, progress_bar)
                progress_bar.progress(1.0, text=None)
                docs_chunks = chunk_texts(docs_texts)
    
    try:
        st.write(docs_chunks)
        pass
    except:
        pass

    st.chat_input(placeholder='Ask your question about the documents 8)')








if __name__ == '__main__':
    st.set_page_config(
        page_title="Chat With Files",
        page_icon="2️⃣",
    )
    UI()









    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

    # for message in st.session_state.messages:
    #     role = message["role"]
    #     with st.chat_message(role):
    #         st.markdown(message["content"])

    # if prompt := st.chat_input("Hi! Ask your question 8)"):
    #     # appending user question to show list
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     # showing user questions and Bot answers

    #     with st.chat_message('user'):
    #         st.markdown(prompt)

    #     with st.chat_message("assistant"):
    #         st.session_state.messages.append(
    #             {"role": "assistant", "content": ""})
    #         full_response = ""

    #         assistant_response = QA(prompt)
    #         answer = assistant_response[0]
    #         confidence = assistant_response[1]

    #         message_placeholder = st.empty()
    #         for chunk in answer.split():
    #             full_response += chunk + " "
    #             time.sleep(0.05)
    #             message_placeholder.markdown(full_response + "▌")
    #         full_response += f"\n\n Confidence: {confidence}"
    #         message_placeholder.markdown(full_response)

    #     # appending Bot answer to show list
    #     st.session_state.messages = st.session_state.messages[:-1]
    #     st.session_state.messages.append(
    #         {"role": "assistant", "content": full_response})


