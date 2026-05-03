def classify_document(text):
    text = text.lower()

    if "invoice" in text:
        return "Invoice"
    elif "resume" in text:
        return "Resume"
    elif "report" in text:
        return "Report"
    else:
        return "General Document"