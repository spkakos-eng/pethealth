import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import base64
from openai import OpenAI
import uvicorn
from typing import List, Dict, Any, Union

load_dotenv()

app = FastAPI()

# Add CORS middleware to allow requests from your Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the OpenAI client with the API key from the .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def read_root():
    return {"message": "Pet AI Backend is running!"}

@app.post("/analyze")
async def analyze_pet(description: str = Form(None), image: UploadFile = File(None)):
    if not description and not image:
        raise HTTPException(status_code=400, detail="Please provide a description or an image.")
    
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful assistant for pet owners. You analyze symptoms and images to provide possible causes and suggest if a vet visit is needed. Always include the disclaimer: 'This information is for guidance only and does not replace a professional veterinary diagnosis. Please consult with a veterinarian for any health concerns.'"
        }
    ]

    content_list: List[Dict[str, Any]] = []

    if description:
        content_list.append({"type": "text", "text": f"The owner describes the following: {description}"})
    
    if image:
        # Check if the file is a valid image
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")

        # Read the image and encode it to base64
        image_data = await image.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    if not content_list:
        raise HTTPException(status_code=400, detail="Please provide a description or an image.")

    user_message: Dict[str, Union[str, List[Dict[str, Any]]]] = {
        "role": "user",
        "content": content_list
    }
    messages.append(user_message)

    try:
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
        )
        # Assuming the first choice contains the response
        diagnosis = response.choices[0].message.content
        return {"diagnosis": diagnosis}
    except Exception as e:
        # Log the exception for debugging
        print(f"Error calling OpenAI API: {e}")
        # Re-raise a more user-friendly HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to get a response from the AI model. Error: {str(e)}")

# This block is for running the server directly with 'python main.py'
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)