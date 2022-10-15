import os
import whisper
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

model = whisper.load_model("base")

# here we would use the audio file from the user
whisperOutput = model.transcribe("audio.wav")

prompt = whisperOutput["text"]

response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)
