import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the processor and model from the pretrained Salesforce BLIP model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    # Process the image into model-compatible format
    inputs = processor(raw_image, return_tensors="pt")

    # Generate a caption for the image
    out = model.generate(**inputs, max_length=50)

    # Decode the generated tokens to text and store it into `caption`
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

# Define the Gradio interface
iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

# Launch the interface if the script is run directly
if __name__ == "__main__":
    iface.launch()
