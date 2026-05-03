from .ocr import extract_text_from_pdf
from .ner import extract_entities
from .classifier import classify_document
from predict import predict_document

def process_document(file, target_lang="en"):

    # Step 1: Extract text
    text = extract_text_from_pdf(file)

    # Step 2: Extract structured data
    entities = extract_entities(text)

    # Step 3: Classify document
    doc_type = classify_document(text)

    # Step 4: Use your existing AI system
    review = predict_document(text, target_lang)

    return {
        "document_type": doc_type,
        "structured_data": entities,
        "review": review
    }