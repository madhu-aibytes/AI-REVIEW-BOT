from PyPDF2 import PdfReader

def extract_text_from_pdf(file):
    text = ""
    reader = PdfReader(file)

    for page in reader.pages:
        text += page.extract_text() or ""

    return text