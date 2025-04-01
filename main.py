#main.py
import os
import io
import base64
import requests
import json
import time
import shutil
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image
import uvicorn
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment variables
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="AI Image Editor API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory if it doesn't exist
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount the output directory as a static files directory
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Mount the root directory as a static files directory for root-saved images
app.mount("/root-images", StaticFiles(directory="."), name="root-images")

# Setup templates directory for image preview
templates_dir = "templates"
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Create a simple HTML template for image preview
with open(f"{templates_dir}/image_view.html", "w") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Preview</title>
        <style>
            body {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f0f0;
            }
            .image-container {
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }
            img {
                max-width: 100%;
                max-height: 80vh;
                border-radius: 4px;
            }
            h2 {
                margin-top: 15px;
                color: #333;
            }
            .download-btn {
                margin-top: 15px;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
            }
            .download-btn:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="image-container">
            <img src="{{ image_url }}" alt="Edited Image">
            <h2>{{ title }}</h2>
            <a href="{{ image_url }}" download class="download-btn">Download Image</a>
        </div>
    </body>
    </html>
    """)

# Base URL for Stability AI API
STABILITY_API_BASE_URL = "https://api.stability.ai/v2beta/stable-image/edit"

# Supported operations and their required fields
OPERATIONS = {
    "erase": ["image"],
    "inpaint": ["image", "prompt"],
    "outpaint": ["image"],
    "search-and-replace": ["image", "prompt", "search_prompt"],
    "search-and-recolor": ["image", "prompt", "search_prompt"],
    "remove-background": ["image"],
}

# Pydantic models for request and response validation
class EditRequestModel(BaseModel):
    operation: str
    prompt: Optional[str] = None
    search_prompt: Optional[str] = None

class EditResponseModel(BaseModel):
    message: str
    image_path: str
    root_image_path: str  # New field for root directory image path
    image_url: str
    preview_url: str
    root_image_url: str  # New field for root directory image URL

# Initialize Gemini
def setup_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')
    return model

def extract_search_prompt(prompt: str) -> str:
    """
    Use Gemini to extract a search prompt from the main prompt for 
    search_replace and search_recolor operations.
    """
    model = setup_gemini()
    
    extraction_prompt = f"""
    Given this image editing instruction: "{prompt}"
    
    Identify what needs to be searched for and replaced/recolored. Return only the search term 
    that identifies what should be found in the image, without any explanation or additional text.
    Example: 
    - If the instruction is "Change the red car to blue", return only "red car"
    - If the instruction is "Make the cloudy sky sunny", return only "cloudy sky"
    """
    
    response = model.generate_content(extraction_prompt)
    search_prompt = response.text.strip()
    return search_prompt

def determine_operation_from_prompt(prompt: str) -> str:
    """
    Use Gemini to determine the most appropriate operation based on the prompt.
    """
    model = setup_gemini()
    
    operation_prompt = f"""
    Based on this image editing instruction: "{prompt}"
    
    Determine the most appropriate operation from these options:
    - erase: Remove something from the image
    - inpaint: Add something to a specific area
    - outpaint: Extend the image beyond its boundaries
    - search-and-replace: Find something and replace it with something else
    - search-and-recolor: Find something and change its color
    - remove-background: Remove the background of the image
    
    Return only the operation name without any explanation.
    """
    
    response = model.generate_content(operation_prompt)
    operation = response.text.strip().lower()
    
    # Validate that the operation is supported
    if operation not in OPERATIONS:
        operation = "inpaint"  # Default to inpaint if unsure
    
    return operation

# Process image to PNG format with alpha channel
def convert_to_png_with_alpha(image_data: bytes) -> bytes:
    """
    Convert any image format to PNG with alpha channel.
    """
    # Open the image from bytes
    img = Image.open(io.BytesIO(image_data))
    
    # Convert to RGBA mode (add alpha channel if not present)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Save as PNG with alpha channel
    output = io.BytesIO()
    img.save(output, format='PNG')
    return output.getvalue()

# Function to call Stability AI API
async def call_stability_api(operation: str, image_data: bytes, params: Dict[str, str]) -> bytes:
    """Make API call to Stability AI for image editing."""
    endpoint = f"{STABILITY_API_BASE_URL}/{operation}"
    
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*"
    }
    
    # Convert to PNG with alpha channel
    png_image_data = convert_to_png_with_alpha(image_data)
    
    # Prepare multipart/form-data
    files = {"image": ("image.png", png_image_data, "image/png")}
    
    # Make the request with multipart/form-data
    response = requests.post(
        endpoint, 
        headers=headers, 
        files=files,
        data=params
    )
    
    if response.status_code != 200:
        error_msg = f"Error from Stability AI API: {response.text}"
        raise HTTPException(status_code=response.status_code, detail=error_msg)
    
    return response.content

# Main API endpoint for image editing
@app.post("/edit", response_model=EditResponseModel)
async def edit_image(
    image: UploadFile = File(...),
    prompt: str = Form(None),
    operation: str = Form(None),
    search_prompt: str = Form(None),
    request: Request = None
):
    """
    Edit an image based on the provided operation and prompt.
    Returns a JSON response with the path to the saved image and clickable URLs.
    """
    # Process the uploaded image
    try:
        image_data = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    
    # Determine operation if not specified
    if not operation:
        if not prompt:
            raise HTTPException(status_code=400, detail="Either operation or prompt must be provided")
        operation = determine_operation_from_prompt(prompt)
    
    # For search operations, extract search_prompt from prompt if not provided
    if operation in ["search-and-replace", "search-and-recolor"] and not search_prompt:
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required for search operations")
        search_prompt = extract_search_prompt(prompt)
    
    # Validate required fields for the operation
    required_fields = OPERATIONS.get(operation, [])
    
    if "prompt" in required_fields and not prompt:
        raise HTTPException(status_code=400, detail=f"Prompt is required for {operation} operation")
    
    if "search_prompt" in required_fields and not search_prompt:
        raise HTTPException(status_code=400, detail=f"Search prompt is required for {operation} operation")
    
    # Prepare parameters for Stability AI API
    params = {}
    
    if prompt:
        params["prompt"] = prompt
    
    if search_prompt:
        params["search_prompt"] = search_prompt
    
    # Call Stability AI API
    try:
        # Get the image result from the API
        result = await call_stability_api(operation, image_data, params)
        
        # Generate a unique filename based on timestamp and operation
        timestamp = int(time.time())
        file_basename = f"edited_{timestamp}_{operation}.png"
        
        # Save the image in the output directory
        output_file_path = os.path.join(OUTPUT_DIR, file_basename)
        with open(output_file_path, "wb") as f:
            f.write(result)
        
        # Save the image in the root directory as well
        root_file_path = file_basename
        with open(root_file_path, "wb") as f:
            f.write(result)
        
        # Build URLs for accessing the image
        base_url = str(request.base_url).rstrip('/')
        direct_image_url = f"{base_url}/output/{file_basename}"  # Direct image URL from output directory
        preview_url = f"{base_url}/view-image/{file_basename}"   # Preview page URL
        root_image_url = f"{base_url}/root-images/{file_basename}"  # Direct image URL from root directory
        
        # Print success message
        print(f"✅ Image edited successfully using {operation} operation!")
        print(f"✅ Saved to output directory: {output_file_path}")
        print(f"✅ Saved to root directory: {root_file_path}")
        
        # Return JSON response with image paths and URLs
        return {
            "message": f"Image successfully edited using {operation} operation",
            "image_path": output_file_path,
            "root_image_path": root_file_path,
            "image_url": direct_image_url,
            "preview_url": preview_url,
            "root_image_url": root_image_url
        }
    except Exception as e:
        print(f"❌ Error in API response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during image editing: {str(e)}")

# Endpoint to view an image in a nice HTML template
@app.get("/view-image/{filename}")
async def view_image(request: Request, filename: str):
    """
    Renders a nice HTML page to view the image.
    """
    # Check if image exists in output directory
    output_path = os.path.join(OUTPUT_DIR, filename)
    root_path = filename
    
    if os.path.exists(output_path):
        # Get the base URL
        base_url = str(request.base_url).rstrip('/')
        image_url = f"{base_url}/output/{filename}"
    elif os.path.exists(root_path):
        # If image doesn't exist in output dir but exists in root dir
        base_url = str(request.base_url).rstrip('/')
        image_url = f"{base_url}/root-images/{filename}"
    else:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Extract operation from filename
    operation = "unknown"
    if "_" in filename:
        try:
            operation = filename.split("_")[2].split(".")[0]
        except:
            pass
    
    # Render the HTML template
    return templates.TemplateResponse(
        "image_view.html",
        {
            "request": request,
            "image_url": image_url,
            "title": f"Image edited with {operation.upper()} operation"
        }
    )

# Endpoint to get a direct file response from root or output directory
@app.get("/image-file/{filename}")
async def get_image_file(filename: str):
    """
    Returns the actual image file directly from either root or output directory
    """
    # Check both locations
    output_path = os.path.join(OUTPUT_DIR, filename)
    root_path = filename
    
    if os.path.exists(output_path):
        file_path = output_path
    elif os.path.exists(root_path):
        file_path = root_path
    else:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=filename
    )

# Endpoint to list all edited images
@app.get("/images/list")
async def list_images(request: Request):
    """
    List all edited images available in both the output directory and root directory with clickable URLs.
    """
    try:
        # Get images from output directory
        output_image_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        # Get images from root directory
        root_image_files = [f for f in os.listdir(".") if f.startswith("edited_") and f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        # Remove duplicates
        all_image_files = list(set(output_image_files + root_image_files))
        
        image_list = []
        base_url = str(request.base_url).rstrip('/')
        
        for file in all_image_files:
            # Check if file exists in output directory
            output_path = os.path.join(OUTPUT_DIR, file)
            root_path = file
            
            # Determine which path to use for metadata
            if os.path.exists(output_path):
                file_path = output_path
                creation_time = os.path.getctime(file_path)
                direct_url = f"{base_url}/output/{file}"
            elif os.path.exists(root_path):
                file_path = root_path
                creation_time = os.path.getctime(file_path)
                direct_url = f"{base_url}/root-images/{file}"
            else:
                continue  # Skip if file doesn't exist in either location
            
            preview_url = f"{base_url}/view-image/{file}"
            
            # Add to list with details
            image_list.append({
                "filename": file,
                "output_path": output_path if os.path.exists(output_path) else None,
                "root_path": root_path if os.path.exists(root_path) else None,
                "direct_url": direct_url,
                "preview_url": preview_url,
                "created": creation_time
            })
        
        # Sort by creation time, most recent first
        image_list.sort(key=lambda x: x["created"], reverse=True)
        
        return {"images": image_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing images: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)