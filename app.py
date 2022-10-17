import os
import whisper
import openai
import numpy as np
import torch
from dotenv import load_dotenv


def app(audioFile, mode):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model("base", device=DEVICE)

    whisperOutput = model.transcribe(audioFile)

    if mode == "code":
        prompt = "Convert this text to code: {}".format(whisperOutput['text'])
    elif mode == "text":
        prompt = "Write a short summary of this text: {}".format(
            whisperOutput['text'])

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

    return response.choices[0].text


# app('./sample.wav', 'text')
