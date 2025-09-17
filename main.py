import os
import base64
import requests
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO

# Configure the API key from environment variables
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"API configuration error: {str(e)}")

app = FastAPI()

class SymptomRequest(BaseModel):
    symptoms: str

@app.get("/")
def read_root():
    return {"message": "PetAI Backend is running!"}

@app.post("/diagnose_text/")
async def diagnose_text(symptom_request: SymptomRequest):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Act as a veterinarian. Based on the following symptoms: '{symptom_request.symptoms}', provide a possible diagnosis for the pet. Respond in Greek."
        response = model.generate_content(prompt)
        return {"diagnosis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diagnose_image/")
async def diagnose_image(symptoms: str, file: UploadFile = File(...)):
    try:
        # Read the image file and convert it to a BytesIO object
        image_bytes = await file.read()
        image_stream = BytesIO(image_bytes)

        # Create a GenerativeModel and send the prompt with the image
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = [
            f"Act as a veterinarian. Based on the provided image and the following symptoms: '{symptoms}', provide a possible diagnosis for the pet. Respond in Greek.",
            image_stream
        ]
        response = model.generate_content(prompt)
        return {"diagnosis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
