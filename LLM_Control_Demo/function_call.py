import json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored  
# import real_grasp
import base64
import visual
import audio
GPT_MODEL = "gpt-4o"
client = OpenAI(api_key="***")

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
image_path = "img.bmp"

tools = [
    {
        "type": "function",
        "function": {
            "name": "put_bowl_into_cabinet",
            "description": "put the bowl into the cabinet at upper or lower location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location of the cabinet that the bowl needs to be put into. It has to be either 'upper' or 'lower'.",
                    }
                },
                "required": ["location"],
            },
        }
    },

    {
        "type": "function",
        "function": {
            "name": "answer_visual_question",
            "description": "answer the user's question about visual scenarios or objects",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    }
]



# init chat completion request
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e



audio.get_audio('test.wav') 

audio_file = open("test.wav", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)







# set messages
messages = []
messages.append({"role": "system", "content": "choose from the following tools: 'put_bowl_into_cabinet' and 'answer_visual_question' to fulfill the user's request. Fill in the arguments for the chosen tool according to the user's request."})
# messages.append({"role": "user", "content": "把绿色的碗放进柜子里"})
# messages.append({"role": "user", "content": "柜子里有几个碗？"})
messages.append({"role": "user", "content": transcription})


chat_response = chat_completion_request(
    messages, tools=tools #, tool_choice={"type": "function", "function": {"name": "put_bowl_into_cabinet"}},
    # function_call = "auto"
)
# print(chat_response.choices[0].message)
# print(chat_response)
function_call = chat_response.choices[0].message.function_call
print(chat_response.choices[0].message)

if function_call == None:
    function_name = chat_response.choices[0].message.tool_calls[0].function.name
    arguments = json.loads(chat_response.choices[0].message.tool_calls[0].function.arguments)
        
    if function_name == "put_bowl_into_cabinet":
        location=arguments["location"]
        # replace it with the real grasp function, don't forget to import the file.
        # put_bowl_into_cabinet(**arguments)

    elif function_name == "answer_visual_question":
        # real_grasp.Solution.save_image()
        visual.answer_visual_question("describe the image")

        print("No Camera Signal Detected. Please check the camera connection.")


        
