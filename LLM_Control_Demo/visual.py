import base64
import requests
from PIL import Image
# OpenAI API Key
api_key = "***"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def answer_visual_question(prompt):

    # Path to your image
    image_path = "test.bmp"
    #save the bmp image in jpg format
    with Image.open(image_path) as img:
        rgb_im = img.convert('RGB')  # Convert to RGB if not already
        new_path = image_path.rsplit('.', 1)[0] + '.jpg'  # Change extension to .jpg
        rgb_im.save(new_path)
        image_path = new_path

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json()
    content = response['choices'][0]['message']['content']
    print(content)