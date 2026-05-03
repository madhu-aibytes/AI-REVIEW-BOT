import random
import pandas as pd

def generate_dataset(n=200):
    good = [
        "Introduction about AI. Detailed explanation with examples. Proper conclusion.",
        "Clear introduction, structured content and strong conclusion."
    ]

    medium = [
        "Introduction present but explanation is average.",
        "Content is somewhat clear but lacks conclusion."
    ]

    poor = [
        "Short text without structure",
        "No introduction and unclear content"
    ]

    data = []

    for _ in range(n):
        category = random.choice(["good", "medium", "poor"])

        if category == "good":
            text = random.choice(good)
            clarity = "good"
            score = random.randint(8, 10)

        elif category == "medium":
            text = random.choice(medium)
            clarity = "medium"
            score = random.randint(5, 7)

        else:
            text = random.choice(poor)
            clarity = "poor"
            score = random.randint(1, 4)

        data.append([text, clarity, score])

    df = pd.DataFrame(data, columns=["text", "clarity", "score"])
    df.to_csv("data/dataset.csv", index=False)

    print("✅ Dataset generated!")