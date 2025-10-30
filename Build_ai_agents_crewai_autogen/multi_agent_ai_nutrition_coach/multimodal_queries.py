#%% imports
import requests
import base64
import os

from PIL import Image

import os 
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process
from crewai import LLM

from dotenv import load_dotenv

load_dotenv()
model_id = "meta-llama/llama-4-maverick-17b-128e-instruct" # 	qwen/qwen3-32b  openai/gpt-oss-120b  "meta-llama/llama-4-maverick-17b-128e-instruct"
groq_api_key = os.getenv("GROQ_API_KEY")
serper_api_key = os.getenv("SERPAPI_API_KEY")
os.environ["SERPER_API_KEY"] = serper_api_key
llm = ChatGroq(model=model_id,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            verbose=1)

crewllm = LLM(model="groq/llama-3.1-8b-instant")

#%% load images

url_image_1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5uo16pKhdB1f2Vz7H8Utkg/image-1.png'
url_image_2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fsuegY1q_OxKIxNhf6zeYg/image-2.png'
url_image_3 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KCh_pM9BVWq_ZdzIBIA9Fw/image-3.png'
url_image_4 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VaaYLw52RaykwrE3jpFv7g/image-4.png'

image_urls = [url_image_1, url_image_2, url_image_3, url_image_4] 
# %% encode images


folder_path = "examples"
encoded_images = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Ensure it's an image file (optional filter)
    if os.path.isfile(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            encoded_images.append(encoded_string)

print(f"Encoded {len(encoded_images)} images.")

# %%

def generate_model_response(encoded_image, user_query, assistant_prompt="You are a helpful assistant. Answer the following user query in 1 or 2 sentences: "):
    """
    Sends an image and a query to the model and retrieves the description or answer.

    Parameters:
    - encoded_image (str): Base64-encoded image string.
    - user_query (str): The user's question about the image.
    - assistant_prompt (str): Optional prompt to guide the model's response.

    Returns:
    - str: The model's response for the given image and query.
    """
    
    # Create the messages object
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": assistant_prompt + user_query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + encoded_image,
                    }
                }
            ]
        }
    ]
    
    # Send the request to the model
    response = llm.invoke(input=messages)
    
    # Return the model's response
    return response.content


# %%

user_query = "Describe the photo"

for i in range(len(encoded_images)):
    image = encoded_images[i]

    response = generate_model_response(image, user_query)

    # Print the response with a formatted description
    print(f"Description for image {i + 1}: {response}")
# %%


