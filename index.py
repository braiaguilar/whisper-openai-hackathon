import os
import whisper
import openai
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model("base", device=DEVICE)
print(
    "### Model info: "
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

# here we would use the audio file from the user
whisperOutput = model.transcribe("audio.wav")

# here we should add the order to summarize the text
prompt = "Write a short summary of this text:" + whisperOutput["text"]

print("### Prompt for GPT-3: " + prompt)

response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)

print("### GPT-3 output: " + response)
