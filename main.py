import os
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
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

# Only the diagnose_image endpoint is needed for now
@app.get("/")
def read_root():
    return {"message": "PetAI Backend is running!"}

@app.post("/diagnose_image/")
async def diagnose_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        
        # Pass the image data directly to the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        image_part = {
            "mime_type": file.content_type,
            "data": image_data
        }
        
        prompt = [
            f"Act as a veterinarian. Based on the provided image, provide a possible diagnosis for the pet. Respond in Greek.",
            image_part
        ]
        
        response = await model.generate_content_async(prompt)
        
        return {"diagnosis": response.text}
    except Exception as e:
        print(f"Error in diagnose_image: {e}")
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")
