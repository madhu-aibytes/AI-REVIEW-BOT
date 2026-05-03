from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_document
import os
import subprocess
from PyPDF2 import PdfReader
import docx
from gtts import gTTS
from flask import send_file
from fastapi.staticfiles import StaticFiles
import uuid
import os



app = Flask(__name__)
CORS(app)

# =========================
# MEMORY (LIMITED HISTORY)
# =========================
chat_history = []
MAX_HISTORY = 20

# =========================
# AUTO TRAIN CHECK
# =========================
def check_and_train():
    if not os.path.exists("models/clf_model.pkl"):
        print("⚠️ Model not found. Running pipeline...")
        subprocess.run(["python", "pipeline.py"])

# =========================
# SAFE HISTORY ADD
# =========================
def add_to_history(input_text, output):
    global chat_history

    chat_history.insert(0, {
        "input": input_text,
        "output": output
    })

    # Limit size
    if len(chat_history) > MAX_HISTORY:
        chat_history.pop()

# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    return "✅ AI Review Bot Backend Running"

# =========================
# TEXT REVIEW API
# =========================
@app.route("/review", methods=["POST"])
def review():
    try:
        check_and_train()

        data = request.get_json(force=True)
        text = data.get("text", "")
        target_lang = data.get("target_lang", "en")

        if not text.strip():
            return jsonify({
                "status": "error",
                "message": "No text provided"
            }), 400

        result = predict_document(text, target_lang)

        add_to_history(text, result)

        return jsonify({
            "status": "success",
            **result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# =========================
# FILE UPLOAD API
# =========================
@app.route("/upload", methods=["POST"])
def upload():
    try:
        check_and_train()

        # =========================
        # CHECK FILE
        # =========================
        if "file" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file uploaded"
            }), 400

        file = request.files["file"]

        # =========================
        # GET LANGUAGE FROM FRONTEND
        # =========================
        target_lang = request.form.get("target_lang", "en")

        # =========================
        # FILE SIZE CHECK (5MB)
        # =========================
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size > 5 * 1024 * 1024:
            return jsonify({
                "status": "error",
                "message": "File too large (Max 5MB)"
            }), 400

        text = ""

        # =========================
        # READ PDF FILE
        # =========================
        if file.filename.lower().endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""

        # =========================
        # READ DOCX FILE
        # =========================
        elif file.filename.lower().endswith(".docx"):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + " "

        # =========================
        # UNSUPPORTED FILE
        # =========================
        else:
            return jsonify({
                "status": "error",
                "message": "Unsupported file format (Only PDF/DOCX)"
            }), 400

        # =========================
        # EMPTY TEXT CHECK
        # =========================
        if not text.strip():
            return jsonify({
                "status": "error",
                "message": "No text extracted from file"
            }), 400

        # =========================
        # AI PREDICTION
        # =========================
        result = predict_document(text, target_lang)

        # =========================
        # SAVE HISTORY (OPTIONAL)
        # =========================
        try:
            add_to_history(f"[FILE] {file.filename}", result)
        except:
            pass  # ignore if not implemented

        # =========================
        # RETURN RESPONSE
        # =========================
        return jsonify({
            "status": "success",
            **result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# =========================
# GET CHAT HISTORY
# =========================
@app.route("/history", methods=["GET"])
def history():
    return jsonify({
        "status": "success",
        "history": chat_history
    })

# =========================
# CLEAR HISTORY
# =========================
@app.route("/clear", methods=["POST"])
def clear():
    global chat_history
    chat_history = []

    return jsonify({
        "status": "success",
        "message": "History cleared"
    })

# =========================
# RETRAIN MODEL
# =========================
@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        subprocess.run(["python", "pipeline.py"])
        return jsonify({
            "status": "success",
            "message": "Model retrained successfully"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })



# =========================
# LIVE TRANSLATION API
# =========================
@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()
        text = data.get("text", "")
        target_lang = data.get("target_lang", "en")

        if not text.strip():
            return jsonify({
                "status": "error",
                "message": "No text provided"
            }), 400

        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(
            source="auto",
            target=target_lang
        ).translate(text)

        return jsonify({
            "status": "success",
            "translated": translated
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500




@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()

    text = data.get("text")
    lang = data.get("lang", "en")

    if not text:
        return jsonify({"error": "No text"}), 400

    try:
        filename = f"audio_{uuid.uuid4()}.mp3"

        tts = gTTS(text=text, lang=lang)
        tts.save(filename)

        return send_file(filename, as_attachment=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="static")



# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)