import os
import time
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
from pydub import AudioSegment
from app import app

header = st.container()
dataset = st.container()

with header:
    st.title("Voice to Code | Conversation Summarizer From Real-Time Audio")

    option = st.selectbox(
        'Please select the mode',
        ('Text', 'Code'),
    )

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)
    val = st_audiorec()

    if isinstance(val, dict):
        with st.spinner('Loading...'):
            ind, val = zip(*val['arr'].items())
            ind = np.array(ind, dtype=int)
            val = np.array(val)
            sorted_ints = val[ind]
            stream = BytesIO(
                b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            wav_bytes = stream.read()

            audio = AudioSegment.from_file(stream, format="wav")
            audio.export("recording.wav", format='wav')

            st.audio(wav_bytes, format='audio/wav')
            time.sleep(3)

            st.write("### Result")
            st.write(app('recording.wav', option.lower()))


# app('./recording.wav', 'text')
