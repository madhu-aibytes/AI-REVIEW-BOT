import re

def extract_entities(text):
    data = {}

    # Simple extraction
    data["amount"] = re.findall(r"\d{3,}", text)
    data["date"] = re.findall(r"\d{2}[/-]\d{2}[/-]\d{4}", text)

    # Example name extraction (basic)
    words = text.split()
    data["names"] = [w for w in words if w.istitle()]

    return data