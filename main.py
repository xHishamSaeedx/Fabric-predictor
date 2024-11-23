from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq_helper import classify_requirements, get_groq_response, classify_requirements_with_image, get_valid_best_uses
from fabric_predictor import FabricPredictor
import pandas as pd
import os
import logging
from typing import Optional
import base64
from PIL import Image
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Mount templates directory
templates = Jinja2Templates(directory="templates")

# Initialize predictor as None
fabric_predictor: Optional[FabricPredictor] = None

class FabricRequest(BaseModel):
    prompt: str

class FabricResponse(BaseModel):
    recommendation: str
    fabric_name: str
    fabric_type: str

class ImageFabricRequest(BaseModel):
    prompt: str
    image: UploadFile

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    global fabric_predictor
    try:
        logger.info("Initializing FabricPredictor...")
        fabric_predictor = FabricPredictor()
        
        # Print valid categories
        valid_categories = get_valid_best_uses()
        logger.info(f"Valid categories: {valid_categories}")
        
        # Check if models exist, if not train them
        if not os.path.exists('models/nn_model.keras'):
            logger.info("Models not found. Training new models...")
            fabric_predictor.train_model()
        else:
            logger.info("Loading existing models...")
            fabric_predictor.load_model()
            
        # Verify initialization
        if fabric_predictor is None:
            raise RuntimeError("FabricPredictor failed to initialize")
            
        logger.info("FabricPredictor initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.options("/fabric-recommendation")
async def fabric_recommendation_options():
    return {"message": "OK"}

@app.post("/fabric-recommendation")
async def get_fabric_recommendation(prompt_data: dict):
    try:
        logger.info(f"Received prompt_data: {prompt_data}")  # Log incoming data
        
        prompt = prompt_data.get('prompt')
        if not prompt:
            raise HTTPException(status_code=422, detail="Prompt is required")
        
        logger.info(f"Processing prompt: {prompt}")
        
        try:
            # Get the fabric recommendation using the existing classify_requirements function
            best_use, durability, texture = await classify_requirements(prompt)
            logger.info(f"Classified as: best_use={best_use}, durability={durability}, texture={texture}")
            
            if not fabric_predictor:
                raise ValueError("Fabric predictor not initialized")
            
            # Use the fabric predictor to get fabric recommendation
            fabric_name, fabric_type = fabric_predictor.predict(best_use, durability, texture)
            logger.info(f"Predicted fabric: name={fabric_name}, type={fabric_type}")
            
            # Get the final response
            final_response = await get_groq_response(
                prompt,
                fabric_name,
                fabric_type,
                best_use,
                durability,
                texture
            )
            logger.info("Got final response from Groq")
            
            response_data = {
                "recommendation": final_response,
                "fabric_name": fabric_name,
                "fabric_type": fabric_type
            }
            logger.info(f"Returning response: {response_data}")
            
            return response_data
            
        except Exception as inner_e:
            logger.error(f"Error in processing: {str(inner_e)}")
            logger.exception("Detailed traceback:")
            raise HTTPException(status_code=422, detail=f"Processing error: {str(inner_e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.exception("Detailed traceback:")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/fabric-recommendation-file")
async def get_fabric_recommendation_file(file: UploadFile = File(...)):
    # Your existing file upload logic here
    pass

@app.post("/fabric-recommendation-with-image", response_model=FabricResponse)
async def get_fabric_recommendation_with_image(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    global fabric_predictor
    
    if fabric_predictor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
        
    try:
        logger.info(f"Processing image request with prompt: {prompt}")
        
        # Read and encode image
        image_content = await image.read()
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        
        # Step 1 & 2: Use Llama to classify the requirements with image
        logger.info("Classifying requirements with image...")
        best_use, durability, texture, item_description = await classify_requirements_with_image(prompt, image_base64)
        logger.info(f"Classified as: {best_use}, {durability}, {texture}")
        logger.info(f"Detected item: {item_description}")
        
        # Step 4: Predict fabric using the trained model and item description
        logger.info("Predicting fabric...")
        
        # Special case handling for directly visible materials
        item_desc_lower = item_description.lower()
        
        # Dictionary of material mappings
        material_mappings = {
            "leather": ("Leather", "Natural/Synthetic"),
            "denim": ("Denim", "Natural Fiber Blend"),
            "wool": ("Wool", "Natural Fiber"),
            "silk": ("Silk", "Natural Fiber"),
            "velvet": ("Velvet", "Natural/Synthetic"),
            "cotton": ("Cotton", "Natural Fiber"),
            "linen": ("Linen", "Natural Fiber"),
            "tweed": ("Tweed", "Natural Fiber"),
            "fleece": ("Fleece", "Synthetic Fiber"),
            "canvas": ("Canvas", "Natural Fiber Blend")
        }
        
        # Check if any known material is mentioned in the description
        fabric_name = None
        fabric_type = None
        
        for material, (fabric, type_) in material_mappings.items():
            if material in item_desc_lower:
                fabric_name = fabric
                fabric_type = type_
                break
        
        # If no specific material was detected, use the model prediction
        if fabric_name is None:
            fabric_name, fabric_type = fabric_predictor.predict(best_use, durability, texture)
            
        logger.info(f"Predicted fabric: {fabric_name}, type: {fabric_type}")
        
        # Step 5: Get final human-like response from Groq
        logger.info("Getting final recommendation...")
        final_response = await get_groq_response(
            prompt, 
            fabric_name, 
            fabric_type,
            best_use,
            durability,
            texture,
            item_description
        )
        
        logger.info("Request processed successfully")
        return FabricResponse(
            recommendation=final_response,
            fabric_name=fabric_name,
            fabric_type=fabric_type
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/valid-categories")
async def get_categories():
    """Debug endpoint to check valid categories"""
    return {"categories": get_valid_best_uses()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 