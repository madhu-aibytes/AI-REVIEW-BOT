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
    elif word_count > 400:
        issues.append("Content too lengthy")

    if len(text.split(".")) < 3:
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

    return round(max(1, min(score, 10)), 2)

# ==============================
# FEEDBACK
# ==============================
def generate_feedback(clarity, score, issues):

    if issues:
        return "The document needs improvement. Issues identified: " + ", ".join(issues)

    if clarity == "good" and score >= 8:
        return "Excellent document with strong structure, clarity, and completeness."

    elif clarity == "medium":
        return "The document is fairly clear but can be improved."

    else:
        return "The document lacks clarity and structure."

# ==============================
# MAIN FUNCTION (UPDATED)
# ==============================
def predict_document(text, target_lang="en"):

    # Clean
    clean = clean_text(text)

    # Vector
    vec_text = vectorizer.transform([clean])

    length = len(text.split())
    has_intro = int("introduction" in text.lower())
    has_conclusion = int("conclusion" in text.lower())

    vec = np.hstack((vec_text.toarray(), [[length, has_intro, has_conclusion]]))

    # Predict
    clarity_pred = clf.predict(vec)[0]
    score_pred = reg.predict(vec)[0]

    clarity = le_clarity.inverse_transform([clarity_pred])[0]

    # Rule check
    issues = rule_check(text)

    # Score
    final_score = adjust_score(score_pred, clarity, issues)

    # Feedback
    feedback = generate_feedback(clarity, final_score, issues)

    # Grammar & Structure (simple logic)
    grammar = "Good grammar" if clarity == "good" else "Needs improvement"
    structure = "Well structured" if "Missing Introduction" not in issues else "Add introduction & conclusion"
    suggestion = feedback

    # -------------------------
    # ENGLISH OUTPUT
    # -------------------------
    english_output = f"""
Clarity: {clarity}
Score: {final_score}/10

Issues:
{', '.join(issues) if issues else 'No major issues'}

Final Feedback:
{feedback}
""".strip()

    # -------------------------
    # TRANSLATION (FIXED)
    # -------------------------
    if target_lang != "en":
        translated_output = translate_text(english_output, target_lang)
    else:
        translated_output = english_output

    # -------------------------
    # RETURN
    # -------------------------
    return {
        "clarity": clarity,
        "grammar": grammar,
        "structure": structure,
        "score": final_score,
        "suggestion": suggestion,
        "summary": translated_output,   # 🔥 IMPORTANT for frontend
        "english_output": english_output,
        "translated_output": translated_output
    }