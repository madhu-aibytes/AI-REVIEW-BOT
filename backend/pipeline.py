from auto_data import generate_dataset
import subprocess

def run_pipeline():
    print("🔁 Generating dataset...")
    generate_dataset(200)

    print("🤖 Training model...")
    subprocess.run(["python", "train.py"])

    print("✅ Pipeline completed!")

if __name__ == "__main__":
    run_pipeline()