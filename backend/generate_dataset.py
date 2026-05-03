import random
import pandas as pd

good_texts = [
    "Introduction about AI. Detailed explanation with examples. Proper conclusion.",
    "Clear introduction, structured content and strong conclusion.",
    "Well explained document with all sections included."
]

medium_texts = [
    "Introduction present but explanation is average and no conclusion.",
    "Content is somewhat clear but structure is weak.",
    "Introduction and content present but lacks proper ending."
]

poor_texts = [
    "Short text without conclusion",
    "No introduction, unclear content",
    "Very poor structure and missing sections"
]

data = []

for i in range(200):
    category = random.choice(["good", "medium", "poor"])

    if category == "good":
        text = random.choice(good_texts)
        clarity = "good"
        completeness = "complete"
        structure = "good"
        score = random.randint(8, 10)

    elif category == "medium":
        text = random.choice(medium_texts)
        clarity = "medium"
        completeness = random.choice(["complete", "incomplete"])
        structure = "medium"
        score = random.randint(5, 7)

    else:
        text = random.choice(poor_texts)
        clarity = "poor"
        completeness = "incomplete"
        structure = "poor"
        score = random.randint(1, 4)

    data.append([text, clarity, completeness, structure, score])

df = pd.DataFrame(data, columns=["text", "clarity", "completeness", "structure", "score"])

df.to_csv("data/dataset.csv", index=False)

print("✅ 200 dataset created successfully!")