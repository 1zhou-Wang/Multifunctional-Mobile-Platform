from openai import OpenAI
import audio
 
client = OpenAI(api_key="***")


audio.get_audio('test.wav') 


audio_file = open("test.wav", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)

# print(transcription)

