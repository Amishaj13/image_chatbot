import os
from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pipeline
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
)
pipe.to(device)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "✨ Image Editing API is running on Render!"

@app.route('/edit-image', methods=['POST'])
def edit_image():
    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({'error': 'Please provide both an image file and a prompt.'}), 400

    prompt = request.form['prompt']
    image_file = request.files['image']

    image = Image.open(image_file.stream).convert("RGB")
    image = image.resize((512, 512))

    # Generate edited image
    edited_image = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1.5).images[0]
    
    # Save and return image
    output_path = "edited_image.png"
    edited_image.save(output_path)
    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render sets PORT env var
    app.run(host='0.0.0.0', port=port)
