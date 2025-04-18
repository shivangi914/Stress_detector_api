
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import librosa
import whisper
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from gramformer import Gramformer

app = FastAPI()

# Load models once
whisper_model = whisper.load_model("base")
gf = Gramformer(models=1)
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Audio feature extraction + processing
def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result["text"]

def correct_grammar(text):
    corrected = list(gf.correct(text))
    return corrected[0] if corrected else text

def compute_embedding(text):
    return embedder.encode(text, convert_to_tensor=True)

# Dummy logic for stress detection (replace with your actual logic if different)
def detect_stress(text):
    text = correct_grammar(text)
    embedding = compute_embedding(text)
    # Example logic: if certain words present
    stress_keywords = ["anxious", "worried", "tired", "pressure", "panic", "overwhelmed"]
    for word in stress_keywords:
        if word in text.lower():
            return "stressed"
    return "not_stressed"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        transcription = transcribe_audio(file_location)
        prediction = detect_stress(transcription)
        os.remove(file_location)
        return JSONResponse(content={"transcription": transcription, "prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
