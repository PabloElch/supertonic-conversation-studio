
import io
import re
import wave

import numpy as np
import streamlit as st
from supertonic import TTS

SAMPLE_RATE = 44100

st.set_page_config(
    page_title="Supertonic Conversation Studio",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Supertonic Conversation Studio")
st.caption("Choose voices, preview them, and generate one complete conversation.")

# Voice labels describe preset IDs, not verified personality traits.
VOICE_OPTIONS = {
    f"Male voice {i} (M{i})": f"M{i}" for i in range(1, 6)
}
VOICE_OPTIONS.update({
    f"Female voice {i} (F{i})": f"F{i}" for i in range(1, 6)
})

LANGUAGES = {
    "English": "en",
    "Arabic": "ar",
    "Bulgarian": "bg",
    "Czech": "cs",
    "Danish": "da",
    "German": "de",
    "Greek": "el",
    "Spanish": "es",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "Hindi": "hi",
    "Croatian": "hr",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Dutch": "nl",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Swedish": "sv",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Vietnamese": "vi",
    "Unknown / fallback": "na",
}

DEFAULT_SCRIPT = """A: Hey! Welcome to our conversation.
B: Thanks! I'm happy to be here.
A: What are you learning today?
B: I'm learning how to create natural-sounding AI voices."""

@st.cache_resource(show_spinner="Loading Supertonic 3 (first load downloads the model)...")
def load_tts():
    return TTS(auto_download=True)

def audio_to_wav(audio):
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    samples = np.nan_to_num(samples)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())

    return output.getvalue()

def parse_script(script):
    turns = []

    for line_no, line in enumerate(script.splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        match = re.match(r"^([AB])\s*:\s*(.+)$", line, re.IGNORECASE)

        if not match:
            raise ValueError(
                f"Line {line_no}: start the line with A: or B:, "
                "followed by the words to speak."
            )

        speaker, text = match.groups()
        turns.append((speaker.upper(), text.strip()))

    if not turns:
        raise ValueError("Enter at least one A: or B: line.")

    if len(turns) > 100:
        raise ValueError("Please use 100 turns or fewer per generation.")

    return turns

def synthesize(tts, text, voice_id, language, speed, steps):
    style = tts.get_voice_style(voice_name=voice_id)

    audio, _ = tts.synthesize(
        text=text,
        voice_style=style,
        lang=language,
        speed=speed,
        total_steps=steps,
    )

    return audio_to_wav(audio)

if "preview_a" not in st.session_state:
    st.session_state.preview_a = None
if "preview_b" not in st.session_state:
    st.session_state.preview_b = None
if "final_audio" not in st.session_state:
    st.session_state.final_audio = None

with st.sidebar:
    st.header("🎭 Speaker settings")

    label_a = st.selectbox(
        "Speaker A voice",
        list(VOICE_OPTIONS.keys()),
        index=2,
        key="voice_a_choice",
    )
    voice_a = VOICE_OPTIONS[label_a]

    if st.button("🔊 Preview Speaker A", use_container_width=True):
        try:
            tts = load_tts()
            with st.spinner("Generating Speaker A preview..."):
                st.session_state.preview_a = synthesize(
                    tts,
                    "Hello! This is a preview of the voice I selected.",
                    voice_a,
                    LANGUAGES[st.session_state.language_choice],
                    st.session_state.speed_choice,
                    st.session_state.quality_choice,
                )
        except Exception as exc:
            st.error(f"Speaker A preview failed: {exc}")

    if st.session_state.preview_a:
        st.audio(st.session_state.preview_a, format="audio/wav")

    st.divider()

    label_b = st.selectbox(
        "Speaker B voice",
        list(VOICE_OPTIONS.keys()),
        index=5,
        key="voice_b_choice",
    )
    voice_b = VOICE_OPTIONS[label_b]

    if st.button("🔊 Preview Speaker B", use_container_width=True):
        try:
            tts = load_tts()
            with st.spinner("Generating Speaker B preview..."):
                st.session_state.preview_b = synthesize(
                    tts,
                    "Hi there! This is a preview of my selected voice.",
                    voice_b,
                    LANGUAGES[st.session_state.language_choice],
                    st.session_state.speed_choice,
                    st.session_state.quality_choice,
                )
        except Exception as exc:
            st.error(f"Speaker B preview failed: {exc}")

    if st.session_state.preview_b:
        st.audio(st.session_state.preview_b, format="audio/wav")

    st.divider()
    st.header("⚙️ Audio settings")

    language_name = st.selectbox(
        "Language",
        list(LANGUAGES.keys()),
        index=0,
        key="language_choice",
    )

    speed = st.slider(
        "Speaking speed",
        min_value=0.7,
        max_value=2.0,
        value=1.0,
        step=0.1,
        key="speed_choice",
    )

    quality = st.select_slider(
        "Generation quality / steps",
        options=[5, 6, 8, 10, 12],
        value=5,
        key="quality_choice",
        help="Lower values are faster; higher values may improve quality.",
    )

    pause = st.slider(
        "Pause between speakers (seconds)",
        0.0,
        2.0,
        0.3,
        0.1,
    )

st.subheader("📝 Conversation script")

script = st.text_area(
    "Enter one speaker turn per line",
    value=DEFAULT_SCRIPT,
    height=240,
    help="Use A: for Speaker A and B: for Speaker B.",
)

st.code("A: Hello! How are you?\nB: I'm doing well, thanks!\nA: What are you doing today?", language="text")

if st.button("🎙️ Generate complete conversation", type="primary", use_container_width=True):
    try:
        turns = parse_script(script)
        tts = load_tts()
        audio_parts = []

        progress = st.progress(0)
        status = st.empty()

        for index, (speaker, words) in enumerate(turns):
            status.write(
                f"Generating turn {index + 1}/{len(turns)} — Speaker {speaker}"
            )

            voice_id = voice_a if speaker == "A" else voice_b

            style = tts.get_voice_style(voice_name=voice_id)
            audio, _ = tts.synthesize(
                text=words,
                voice_style=style,
                lang=LANGUAGES[language_name],
                speed=speed,
                total_steps=quality,
            )

            samples = np.asarray(audio, dtype=np.float32).reshape(-1)
            audio_parts.append(samples)

            if index < len(turns) - 1 and pause > 0:
                silence = np.zeros(
                    int(SAMPLE_RATE * pause), dtype=np.float32
                )
                audio_parts.append(silence)

            progress.progress((index + 1) / len(turns))

        combined = np.concatenate(audio_parts)
        st.session_state.final_audio = audio_to_wav(combined)

        status.success(f"Done! Generated all {len(turns)} turns.")
    except Exception as exc:
        st.error(f"Audio generation failed: {exc}")

if st.session_state.final_audio:
    st.subheader("🎧 Your complete conversation")
    st.audio(st.session_state.final_audio, format="audio/wav")

    st.download_button(
        "⬇️ Download complete conversation",
        data=st.session_state.final_audio,
        file_name="supertonic_conversation.wav",
        mime="audio/wav",
        type="primary",
        use_container_width=True,
    )
