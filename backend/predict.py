import pickle
import re
import numpy as np
from deep_translator import GoogleTranslator

# ==============================
# LOAD MODELS
# ==============================
clf = pickle.load(open("models/clf_model.pkl", "rb"))
reg = pickle.load(open("models/reg_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
le_clarity = pickle.load(open("models/le_clarity.pkl", "rb"))

# ==============================
# CLEAN TEXT
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ==============================
# LANGUAGE DETECTION
# ==============================
def detect_language(text):
    if any("\u0B80" <= c <= "\u0BFF" for c in text):
        return "ta"
    elif any("\u0900" <= c <= "\u097F" for c in text):
        return "hi"
    elif any("\u0C00" <= c <= "\u0C7F" for c in text):
        return "te"
    elif any("\u0C80" <= c <= "\u0CFF" for c in text):
        return "kn"
    elif any("\u0D00" <= c <= "\u0D7F" for c in text):
        return "ml"
    else:
        return "en"

# ==============================
# LANGUAGE MAP
# ==============================
language_map = {
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "en": "English"
}

# ==============================
# TRANSLATION
# ==============================
def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except:
        return text

# ==============================
# RULE CHECK
# ==============================
def rule_check(text):
    issues = []
    text_lower = text.lower()
    word_count = len(text.split())

    if "introduction" not in text_lower:
        issues.append("Missing Introduction")

    if "conclusion" not in text_lower:
        issues.append("Missing Conclusion")

    if word_count < 100:
        issues.append("Content too short")
    elif word_count > 1000:
        issues.append("Content too lengthy")

    sentences = text.split(".")
    if len(sentences) < 3:
        issues.append("Very few sentences")

    words = text_lower.split()
    if len(words) != len(set(words)):
        repetition_ratio = 1 - (len(set(words)) / len(words))
        if repetition_ratio > 0.3:
            issues.append("Too much repetition")

    if text.count(".") == 0:
        issues.append("Poor sentence structure")

    if text.islower():
        issues.append("No proper capitalization")

    return issues

# ==============================
# SCORE ADJUSTMENT
# ==============================
def adjust_score(score, clarity, issues):

    if clarity == "good":
        score += 0.5
    elif clarity == "medium":
        score -= 0.5
    else:
        score -= 1

    penalty_map = {
        "Content too short": 2,
        "Content too lengthy": 1,
        "Missing Introduction": 1.5,
        "Missing Conclusion": 1.5,
        "Very few sentences": 1,
        "Too much repetition": 1.5,
        "Poor sentence structure": 2,
        "No proper capitalization": 1
    }

    for issue in issues:
        score -= penalty_map.get(issue, 1)

    score = max(1, min(score, 10))
    return round(score, 2)

# ==============================
# FEEDBACK GENERATION
# ==============================
def generate_feedback(clarity, score, issues):

    if issues:
        return "The document needs improvement. Issues identified: " + ", ".join(issues)

    if clarity == "good" and score >= 8:
        return "Excellent document with strong structure, clarity, and completeness."

    elif clarity == "medium":
        return "The document is fairly clear but can be improved with better organization and detail."

    else:
        return "The document lacks clarity and structure. Significant improvements are required."

# ==============================
# EXTRA UI TEXT (FIX)
# ==============================
def generate_extra_fields(issues, clarity):

    # Grammar
    if "Poor sentence structure" in issues:
        grammar = "Sentence structure needs improvement. Use proper punctuation and complete sentences."
    else:
        grammar = "Grammar looks acceptable with minor improvements needed."

    # Structure
    if "Missing Introduction" in issues or "Missing Conclusion" in issues:
        structure = "Document lacks proper structure. Add introduction and conclusion sections."
    else:
        structure = "Structure is fairly organized but can be improved."

    # Suggestion
    suggestion = "Improve clarity, avoid repetition, and expand ideas with proper structure."

    return grammar, structure, suggestion

# ==============================
# MAIN FUNCTION
# ==============================
def predict_document(text, target_lang="en"):

    # Clean
    clean = clean_text(text)

    # TF-IDF
    vec_text = vectorizer.transform([clean])

    # Extra features
    length = len(text.split())
    has_intro = int("introduction" in text.lower())
    has_conclusion = int("conclusion" in text.lower())

    vec = np.hstack((vec_text.toarray(), [[length, has_intro, has_conclusion]]))

    # Predictions
    clarity_pred = clf.predict(vec)[0]
    score_pred = reg.predict(vec)[0]

    clarity = le_clarity.inverse_transform([clarity_pred])[0]

    # Language
    lang_code = detect_language(text)
    language_name = language_map.get(lang_code, "English")

    # Rules
    issues = rule_check(text)

    # Adjust score
    final_score = adjust_score(score_pred, clarity, issues)

    # Feedback
    feedback = generate_feedback(clarity, final_score, issues)

    # Extra UI fields
    grammar_text, structure_text, suggestion_text = generate_extra_fields(issues, clarity)

    # English output
    english_output = f"""
Clarity: {clarity}
Score: {final_score}/10

Issues:
{', '.join(issues) if issues else 'No major issues'}

Final Feedback:
{feedback}
""".strip()

    # Translation
    if lang_code != "en" and len(english_output) < 4000:
        translated_output = translate_text(english_output, lang_code)
    else:
        translated_output = english_output

    # Final return
    return {
        "language": language_name,
        "clarity": clarity,
        "grammar": grammar_text,
        "structure": structure_text,
        "score": final_score,
        "suggestion": suggestion_text,
        "summary": english_output + "\n\n--- Translated ---\n" + translated_output,
        "english_output": english_output,
        "translated_output": translated_output
    }