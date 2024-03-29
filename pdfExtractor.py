from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTRect
import pdfplumber


def text_extraction(element):
    line_text = element.get_text()
    return line_text


def extract_table(pdf_path, page_num, table_num):
    pdf = pdfplumber.open(pdf_path)
    table_page = pdf.pages[page_num]
    table = table_page.extract_tables()[table_num]
    return table


def table_converter(table):
    table_string = ''
    # Iterate through each row of the table
    for row_num in range(len(table)):
        row = table[row_num]
        # Remove the line breaker from the wrapped texts
        cleaned_row = [item.replace(
            '\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item in row]
        # Convert the table into a string
        table_string += ('|'+'|'.join(cleaned_row)+'|'+'\n')
    # Removing the last line break
    table_string = table_string[:-1]
    return table_string


def extract_text(file, close_file=False):
    # pdfFileObj = open(path, 'rb')
    pdfFileObj = file

    text_per_page = {}
    for pagenum, page in enumerate(extract_pages(file)):

        page_text = []
        text_from_tables = []
        # Initialize the number of the examined tables
        table_num = 0
        first_element = True
        table_extraction_flag = False
        # Open the pdf file
        pdf = pdfplumber.open(file)
        # Find the examined page
        page_tables = pdf.pages[pagenum]
        # Find the number of tables on the page
        tables = page_tables.find_tables()

        # Find all the elements
        page_elements = [(element.y1, element) for element in page._objs]
        # Sort all the elements as they appear in the page
        page_elements.sort(key=lambda a: a[0], reverse=True)

        # Find the elements that composed a page
        for i, component in enumerate(page_elements):
            # Extract the element of the page layout
            element = component[1]

            # Check if the element is a text element
            if isinstance(element, LTTextContainer):
                # Check if the text appeared in a table
                if table_extraction_flag == False:
                    # Use the function to extract the text and format for each text element
                    line_text = text_extraction(element)
                    # Append the text of each line to the page text
                    page_text.append(line_text)
                else:
                    # Omit the text that appeared in a table
                    pass

        text_per_page[pagenum] = [page_text, text_from_tables]

    if close_file:
        pdfFileObj.close()

    return text_per_page
