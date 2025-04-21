import requests
import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import logging
import google.generativeai as genai
import base64
import io
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from django.core.files.base import ContentFile
from django.conf import settings


# Configure logging
logger = logging.getLogger(__name__)

# Together AI API Configuration
TOGETHER_API_KEY = "tgp_v1_QdavwTK80YzPTG3tCj-ZqctYk7_8P_PM9zZvbKQISgc"  # Use environment variable
TOGETHER_API_URL = "https://api.together.ai/v1/chat/completions"

@csrf_exempt
def generate_blog(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            topic = data.get("topic", "").strip()
            category = data.get("category", "").strip()

            if not topic or not category:
                return JsonResponse({"error": "Both topic and category are required."}, status=400)

            # Construct the prompt
            prompt = f"Write a detailed blog post about '{topic}' in the '{category}' category."

            # Make request to Together AI
            headers = {
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                "messages": [
                    {"role": "system", "content": "You are an AI assistant that generates high-quality blog posts."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }

            response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)

            if response.status_code == 200:
                blog_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                if not blog_text:
                    return JsonResponse({"error": "AI response was empty. Please try again."}, status=500)

                return JsonResponse({"blog_post": blog_text})
            else:
                logger.error(f"Together AI API Error: {response.text}")
                return JsonResponse({"error": f"Together AI error: {response.text}"}, status=response.status_code)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format in request."}, status=400)
        except requests.RequestException as e:
            logger.error(f"Request to Together AI failed: {str(e)}")
            return JsonResponse({"error": "Failed to connect to Together AI. Please try again later."}, status=500)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return JsonResponse({"error": "An unexpected error occurred."}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)


# genai.configure(api_key="AIzaSyCw9bcdboum1TOw85ZSVCy_xp_XWJdHKtU")

# # Step 2: Load the Gemini model (or any valid model)
# model = genai.GenerativeModel(model_name="gemini-1.5-pro-002")

# # Step 3: Create a view to generate captions
# @csrf_exempt  # Use this decorator to exempt CSRF validation (for simplicity, but better to use token-based auth in production)
# def generate_caption(request):
#     if request.method == "POST":
#         try:
#             # Get the vibe from the POST request body
#             vibe = request.POST.get("vibe", "")
#             n = int(request.POST.get("n", 3))  # Number of captions to generate (default 3)

#             # Create prompt for Gemini model
#             prompt = f"Generate {n} short, aesthetic, and creative Instagram captions for a photo with this vibe: '{vibe}'"
#             response = model.generate_content(prompt)

#             # Send response back to the client with the generated captions
#             captions = response.text.strip().split("\n")  # Split by new lines if multiple captions are generated

#             return JsonResponse({"captions": captions}, status=200)

#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=400)
    
#     return JsonResponse({"error": "Invalid request method"}, status=405)




# views.py
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyCw9bcdboum1TOw85ZSVCy_xp_XWJdHKtU")

# Load the Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-pro-002")

@csrf_exempt  # This will allow non-CSRF requests, make sure you use proper security in production!
def generate_caption(request):
    if request.method == "POST":
        try:
            # Parse the JSON data sent by the frontend
            data = json.loads(request.body)
            vibe = data.get("vibe", "")
            n = int(data.get("n", 3))  # Default to 3 captions if not specified

            # Generate prompt for the model
            prompt = f"Generate {n} creative captions for this vibe: '{vibe}'"
            response = model.generate_content(prompt)
            captions = response.text.strip().split("\n")

            return JsonResponse({"captions": captions}, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    
    return JsonResponse({"error": "Invalid request method"}, status=405)










# class CFG:
#     device = "cuda"
#     seed = 42
#     generator = torch.Generator(device).manual_seed(seed)
#     image_gen_steps = 35
#     image_gen_model_id = "stabilityai/stable-diffusion-2"
#     image_gen_size = (400, 400)
#     image_gen_guidance_scale = 9
#     prompt_gen_model_id = "gpt2"
#     prompt_dataset_size = 6
#     prompt_max_length = 12

# # --- Load the model once ---
# image_gen_model = StableDiffusionPipeline.from_pretrained(
#     CFG.image_gen_model_id,
#     torch_dtype=torch.float16,
#     revision="fp16",
#     use_auth_token='your_hugging_face_auth_token',
#     guidance_scale=9
# )
# image_gen_model = image_gen_model.to(CFG.device)

# # --- Image generation function ---
# def generate_image(prompt, model):
#     image = model(
#         prompt,
#         num_inference_steps=CFG.image_gen_steps,
#         generator=CFG.generator,
#         guidance_scale=CFG.image_gen_guidance_scale
#     ).images[0]

#     image = image.resize(CFG.image_gen_size)
#     return image

# # --- View to handle requests ---
# @csrf_exempt
# def generate_image_api(request):
#     if request.method == "POST":
#         import json
#         data = json.loads(request.body)
#         prompt = data.get("prompt")

#         if not prompt:
#             return JsonResponse({"error": "Prompt is required"}, status=400)

#         try:
#             image = generate_image(prompt, image_gen_model)
#             buffer = io.BytesIO()
#             image.save(buffer, format="PNG")
#             image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

#             return JsonResponse({"image_base64": image_base64})
#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)

#     return JsonResponse({"error": "Only POST method allowed"}, status=405)



import io
import json
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
from diffusers import StableDiffusionPipeline



class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

# Load the model once
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token='your_hugging_face_auth_token',
    guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

# Image generation function
def generate_image(prompt, model):
    image = model(
        prompt,
        num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

# View to handle image generation API requests
@csrf_exempt
def generate_image_api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        prompt = data.get("prompt")

        if not prompt:
            return JsonResponse({"error": "Prompt is required"}, status=400)

        try:
            image = generate_image(prompt, image_gen_model)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return JsonResponse({"image_base64": image_base64})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST method allowed"}, status=405)

