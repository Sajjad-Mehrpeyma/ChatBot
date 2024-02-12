import streamlit as st
from pdfExtractor import extract_text
from KeyPhraseExtractor import extract
from spire.pdf import *
from spire.pdf.common import *


def highlight(pdf, file_keyPhrases):
    for i in range(pdf.Pages.Count):
        # Get a page
        page = pdf.Pages.get_Item(i)
        # Find all occurrences of specific text on the page
        pagePhrases = file_keyPhrases[i]

        for word in pagePhrases:
            result = page.FindText(
                word, TextFindParameter.IgnoreCase).Finds
            # Highlight all the occurrences
            for text in result:
                text.ApplyHighLight(Color.get_Cyan())

    # Save the document
    pdf.SaveToFile(r"pdfs\file_modified.pdf")


def highlight_pdf():
    uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
    if uploaded_file is not None:
        corpus = []
        result = extract_text(uploaded_file, close_file=False)
        for page in result.values():
            text = "".join(page[0])
            corpus.append(text)

        keyPhrases = extract(corpus)

        with open(r'pdfs\file_modified.pdf', "wb") as f:
            f.write((uploaded_file).getbuffer())

        pdf = PdfDocument()
        pdf.LoadFromFile(r'pdfs\file_modified.pdf')

        highlight(pdf, keyPhrases)
        uploaded_file.close()
        pdf.Close()
        success = True
        return success


if __name__ == '__main__':
    st.set_page_config(
        page_title="KeyPhrase Extraction",
        page_icon="1️⃣",
    )
    success = highlight_pdf()
    if success:
        success_message = "File Saved"
        st.success(success_message)
