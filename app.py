
import io
import re
import wave

import numpy as np
import streamlit as st
from supertonic import TTS

st.set_page_config(
    page_title="Supertonic Conversation Studio",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Conversation Studio")
st.caption("Turn a written dialogue into one audio file.")

DEFAULT_SCRIPT = """A: Hey! How are you today?
B: I'm doing great, thanks for asking.
A: What are you working on?
B: I'm learning something new."""

with st.sidebar:
    st.header("Voice settings")
    voice_a = st.selectbox("Speaker A", ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"], index=5)
    voice_b = st.selectbox("Speaker B", ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"], index=0)
    language = st.selectbox(
        "Language",
        ["en", "am", "ar", "fr", "de", "es", "hi", "it", "ja", "ko", "pt", "ru", "tr", "zh", "na"],
        index=0,
    )
    speed = st.slider("Speaking speed", 0.7, 1.5, 1.0, 0.05)
    pause = st.slider("Pause between turns (seconds)", 0.0, 1.5, 0.3, 0.1)

script = st.text_area(
    "Conversation script",
    value=DEFAULT_SCRIPT,
    height=260,
    help="Put each speaker turn on a new line, beginning with A: or B:.",
)

st.info("Example format: A: Hello!  (new line) B: Hi there!")

def parse_script(text):
    turns = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^([AB])\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if not match:
            raise ValueError(
                f"Line {line_number} must start with A: or B: and contain text."
            )
        speaker, words = match.groups()
        turns.append((speaker.upper(), words.strip()))
    if not turns:
        raise ValueError("Please enter at least one conversation turn.")
    return turns

@st.cache_resource(show_spinner="Loading Supertonic 3 model...")
def load_tts():
    return TTS(auto_download=True)

def wav_bytes(audio, sample_rate=44100):
    samples = np.asarray(audio).squeeze()
    if samples.ndim != 1:
        raise ValueError("Unexpected audio shape returned by the model.")
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    output = io.BytesIO()
    with wave.open(output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return output.getvalue()

if st.button("Generate conversation audio", type="primary", use_container_width=True):
    try:
        turns = parse_script(script)
        if len(turns) > 100:
            st.error("Please limit each generation to 100 turns.")
            st.stop()

        tts = load_tts()
        audio_parts = []
        progress = st.progress(0)
        status = st.empty()

        for index, (speaker, text) in enumerate(turns):
            status.write(f"Generating turn {index + 1}/{len(turns)} — Speaker {speaker}")
            voice = voice_a if speaker == "A" else voice_b
            style = tts.get_voice_style(voice_name=voice)
            audio, _ = tts.synthesize(
                text,
                voice_style=style,
                lang=language,
                speed=speed,
            )
            samples = np.asarray(audio, dtype=np.float32).squeeze()
            if samples.ndim != 1:
                raise ValueError(f"Unexpected audio shape at turn {index + 1}.")
            audio_parts.append(samples)
            if index < len(turns) - 1 and pause > 0:
                audio_parts.append(
                    np.zeros(int(44100 * pause), dtype=np.float32)
                )
            progress.progress((index + 1) / len(turns))

        combined = np.concatenate(audio_parts)
        result = wav_bytes(combined)

        status.success("Conversation audio is ready!")
        st.audio(result, format="audio/wav")
        st.download_button(
            "⬇️ Download complete conversation (WAV)",
            data=result,
            file_name="conversation.wav",
            mime="audio/wav",
            use_container_width=True,
        )
        st.caption(f"Generated {len(turns)} turns.")

    except Exception as exc:
        st.error(f"Generation failed: {exc}")
        st.info("If the model failed to load, check the deployment logs.")
