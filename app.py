import os
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
from openai import OpenAI

# ========= 1. OPENAI CLIENT CONFIG =========

# Make sure OPENAI_API_KEY is set in your environment
# e.g. export OPENAI_API_KEY="sk-..."
client = OpenAI()

# You can adjust these model names if needed
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"   # speech → text
CHAT_MODEL       = "gpt-5.1-mini"             # main reasoning model
TTS_MODEL        = "gpt-4o-mini-tts"          # text → speech
TTS_VOICE        = "coral"                    # alloy / ash / ballad / coral / echo / etc.


# ========= 2. STREAMLIT UI LAYOUT =========

st.set_page_config(page_title="CSV Voice Assistant", page_icon="🧠")
st.title("🧠📊 CSV Voice Assistant (Option 1 Demo)")
st.write(
    "1. Upload a CSV or Excel file\n"
    "2. Click the mic and ask a question with your voice\n"
    "3. The AI will answer based **only** on your file and speak back"
)

uploaded_file = st.file_uploader(
    "Upload your CSV/Excel file with data",
    type=["csv", "xlsx"]
)

if not uploaded_file:
    st.info("Please upload a CSV or Excel file to begin.")
    st.stop()

# ========= 3. LOAD DATAFRAME FROM FILE =========

try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head())

# Whole table as CSV text (for context to the model)
csv_text = df.to_csv(index=False)


# ========= 4. AUDIO INPUT (MIC) =========

st.markdown("### 🎙️ Ask a question by voice")
audio_data = st.audio_input("Record your question")

if not audio_data:
    st.info("Click the microphone above and ask something like:\n"
            "- 'How many items are low in stock?'\n"
            "- 'What is the quantity of product X?'")
    st.stop()

st.info("Processing your audio… this may take a few seconds.")


# ========= 5. SAVE AUDIO TO TEMP FILE =========

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
    tmp_audio.write(audio_data.read())
    tmp_audio_path = tmp_audio.name

try:
    # ========= 6. SPEECH → TEXT (TRANSCRIPTION) =========
    with open(tmp_audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=f
        )

    user_question = transcription.text

    st.markdown("### 🗣️ You said:")
    st.write(user_question)

    # ========= 7. CHATGPT ANSWER BASED ON CSV =========

    system_prompt = f"""
You are a helpful data assistant that answers questions about a table.

You are given a CSV table with the current data:

CSV DATA:
{csv_text}

Rules:
- Always base your answer ONLY on this CSV data.
- If the user asks about a product or row, look for it in the table.
- If you cannot find something in the CSV, say clearly that it is not available.
- Be brief, clear, and conversational.
- Do NOT invent numbers or products that are not present in the CSV.
"""

    response = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
    )

    # Helper: new API returns output as a structured object; `.output_text`
    answer_text = response.output_text

    st.markdown("### ✅ Answer")
    st.write(answer_text)

    # ========= 8. TEXT → SPEECH (TTS) =========

    st.markdown("### 🔊 Spoken Answer")

    speech_path = Path(tempfile.gettempdir()) / "answer_speech.mp3"

    # Stream TTS audio to a file
    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=answer_text,
        instructions="Speak clearly in a friendly, helpful tone.",
    ) as tts_response:
        tts_response.stream_to_file(speech_path)

    # Play back in Streamlit
    with open(speech_path, "rb") as f:
        audio_bytes = f.read()

    st.audio(audio_bytes, format="audio/mp3", autoplay=True)

except Exception as e:
    st.error(f"An error occurred: {e}")

finally:
    # Clean up temp audio file
    try:
        os.remove(tmp_audio_path)
    except OSError:
        pass
