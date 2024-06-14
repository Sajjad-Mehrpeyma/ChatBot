# from pdfminer.high_level import extract_pages
# from pdfminer.layout import LTTextContainer, LTRect
# import pdfplumber
from PyPDF2 import PdfReader


def extract_text_pdf(PDFs, progress_bar=None):
    text = ''
    progress_value = 0.0
    pdfs_count = len(PDFs)
    for pdf in PDFs:
        pdf_reader = PdfReader(pdf)
        pages_count = len(pdf_reader.pages) + .05
        progress_value_add = 1/(pdfs_count*pages_count)
        
        for page in pdf_reader.pages:
            text += "\n" + page.extract_text()
            if progress_bar != None:
                progress_value += progress_value_add
                progress_bar.progress(progress_value)
        text += '\n\n'
    return text
