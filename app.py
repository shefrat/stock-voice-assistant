import streamlit as st
import google.generativeai as genai
import pandas as pd
from gtts import gTTS
import tempfile
# import os
# os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
# os.environ["GOOGLE_CLOUD_REGION"] = "us-central1"

# # --- 1. CONFIGURATION ---
# # Replace with your actual API Key

import os

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    # Fallback for running locally without a secrets file (optional)
    GOOGLE_API_KEY = "YOUR_ACTUAL_KEY_HERE_ONLY_FOR_LOCAL_TESTING"

genai.configure(api_key=GOOGLE_API_KEY)

# We use Gemini 1.5 Flash because it's fast and multimodal (hears audio + reads files)
# OLD: model = genai.GenerativeModel('gemini-1.5-flash')
model = genai.GenerativeModel('gemini-2.5-flash')

# --- 2. UI SETUP ---
st.title("🛒 Retail Stock Voice Assistant")
st.write("Upload a CSV, click the mic, and ask about your stock!")

# --- 3. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your Stock CSV/Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Load data into a dataframe to show the user
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.subheader("Current Stock Data:")
    st.dataframe(df.head()) # Show first few rows
    
    # Save the CSV text string to feed to Gemini later
    csv_string = df.to_string(index=False)

    # --- 4. VOICE INPUT ---
    # Streamlit's native audio input (requires Streamlit 1.39+)
    audio_value = st.audio_input("Record your question")

    if audio_value:
        st.info("Listening and analyzing...")

        # Save the recorded audio to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_value.read())
            temp_audio_path = temp_audio.name

        try:
            # --- 5. THE BRAIN (GEMINI) ---
            # We upload the audio file to Gemini
            myfile = genai.upload_file(temp_audio_path)
            
            # We construct the prompt: Context (CSV) + User Voice (Audio File)
            prompt = f"""
            You are a helpful retail assistant. 
            Here is the current stock data in CSV format:
            {csv_string}

            Please listen to the user's audio question and answer based *strictly* on this data.
            Keep the answer conversational and brief.
            """

            # Generate content using Audio + Text Prompt
            response = model.generate_content([prompt, myfile])
            answer_text = response.text

            # --- 6. THE VOICE (OUTPUT) ---
            st.success("Answer:")
            st.write(answer_text)

            # Convert text to speech using Google Text-to-Speech (gTTS)
            tts = gTTS(text=answer_text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format='audio/mp3', autoplay=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        finally:
            # Cleanup temp file
            os.remove(temp_audio_path)

else:
    st.info("Please upload a CSV file to begin.")