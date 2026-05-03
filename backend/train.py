import pandas as pd
import re
import nltk
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

# ======================
# LOAD DATA
# ======================
data = pd.read_csv("data/dataset.csv")

# ======================
# CLEAN TEXT
# ======================
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    words = str(text).lower().split()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

data["clean_text"] = data["text"].apply(clean_text)

# ======================
# EXTRA FEATURES
# ======================
data["length"] = data["text"].apply(lambda x: len(x.split()))
data["has_intro"] = data["text"].str.contains("introduction", case=False).astype(int)
data["has_conclusion"] = data["text"].str.contains("conclusion", case=False).astype(int)

# ======================
# TF-IDF
# ======================
vectorizer = TfidfVectorizer(max_features=2000)
X_text = vectorizer.fit_transform(data["clean_text"])

X_extra = data[["length", "has_intro", "has_conclusion"]].values

X = np.hstack((X_text.toarray(), X_extra))

# ======================
# LABELS
# ======================
le_clarity = LabelEncoder()
y_clarity = le_clarity.fit_transform(data["clarity"])
y_score = data["score"]

# ======================
# SPLIT (WITH RANDOM STATE)
# ======================
X_train, X_test, yc_train, yc_test = train_test_split(
    X, y_clarity, test_size=0.2, random_state=42
)

_, _, ys_train, ys_test = train_test_split(
    X, y_score, test_size=0.2, random_state=42
)

# ======================
# MODELS
# ======================
clf = RandomForestClassifier(n_estimators=50, max_depth=5)
reg = RandomForestRegressor()

# ======================
# TRAIN
# ======================
clf.fit(X_train, yc_train)
reg.fit(X_train, ys_train)

# ======================
# CROSS VALIDATION
# ======================
scores = cross_val_score(clf, X, y_clarity, cv=5)
print("Cross Validation Accuracy:", scores.mean())

# ======================
# EVALUATE
# ======================
yc_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(yc_test, yc_pred))

ys_pred = reg.predict(X_test)
print("MSE:", mean_squared_error(ys_test, ys_pred))

# ======================
# SAVE MODELS
# ======================
pickle.dump(clf, open("models/clf_model.pkl", "wb"))
pickle.dump(reg, open("models/reg_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
pickle.dump(le_clarity, open("models/le_clarity.pkl", "wb"))

print("Models saved successfully!")