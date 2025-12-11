import streamlit as st
import google.generativeai as genai
import pandas as pd
from gtts import gTTS
import tempfile
import os
import time
from streamlit_mic_recorder import mic_recorder

# --- CONFIGURATION ---
# Replace with your actual key if not in environment variables
# os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"

# Validate API Key
if not os.environ.get("GOOGLE_API_KEY"):
    st.error("⚠️ API Key missing! Please set GEMINI_API_KEY in your environment.")
    st.stop()

# Initialize Model (Using the older syntax for compatibility)
model_name = 'gemini-2.5-flash'

SYSTEM_INSTRUCTION_BASE = """
You are a helpful retail stock assistant. All your answers must be based *strictly* on 
the provided stock data, which includes columns like Item_Name, Quantity, Price, etc.
The data available is (first 10 rows):
{csv_data_string}

Always keep the answers conversational and concise. Acknowledge that you have the stock data.
"""

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    st.session_state.chat = None
if "context_loaded" not in st.session_state:
    st.session_state.context_loaded = False

# --- HELPER FUNCTIONS ---
def text_to_speech(text):
    """Converts text to speech and plays it."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_tts:
            tts = gTTS(text=text, lang='en')
            tts.write_to_fp(temp_tts)
            temp_tts_path = temp_tts.name
        
        st.audio(temp_tts_path, format='audio/mp3', autoplay=True)
        # Small delay to ensure player loads
        time.sleep(1) 
        os.remove(temp_tts_path)
    except Exception as e:
        st.warning(f"Audio playback failed: {e}")

def get_stock_data_string(uploaded_file):
    """Reads the uploaded file and returns a string preview."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df.head(10).to_string(index=False)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

# --- MAIN APP UI ---
st.title("🛒 Retail Stock Voice Assistant")

# File Uploader
uploaded_file = st.file_uploader("Upload Stock CSV/Excel", type=["csv", "xlsx"])

if uploaded_file:
    # 1. Initialize Chat Context (Runs Once)
    if not st.session_state.context_loaded:
        csv_string = get_stock_data_string(uploaded_file)
        
        # Configure the model with the system instruction
        final_system_instruction = SYSTEM_INSTRUCTION_BASE.format(csv_data_string=csv_string)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=final_system_instruction
        )
        
        # Start Chat
        st.session_state.chat = model.start_chat()
        st.session_state.context_loaded = True
        st.session_state.messages = []
        
        # Welcome Message
        welcome_msg = "I've loaded your stock data. Ready for your questions!"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        text_to_speech(welcome_msg)
        st.rerun()

    # 2. Display Chat History
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # 3. Voice Input Section
    st.write("---")
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # THE RECORDER BUTTON
        # key="voice_recorder" is CRITICAL to prevent 'nothing happens' bugs
        audio_data = mic_recorder(
            start_prompt="🎤 Speak",
            stop_prompt="🛑 Stop",
            just_once=True,
            use_container_width=True,
            format="wav",
            key="voice_recorder" 
        )

    # 4. Process Audio if Recorded
    if audio_data and audio_data['bytes']:
        with st.spinner("Listening and analyzing..."):
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_data['bytes'])
                temp_audio_path = temp_audio.name

            try:
                # Upload to Gemini
                myfile = genai.upload_file(temp_audio_path)
                
                # Send to Chat
                response = st.session_state.chat.send_message(
                    contents=[myfile],
                    config={'temperature': 0.1}
                )
                answer_text = response.text

                # Update History
                st.session_state.messages.append({"role": "user", "content": "🎤 (Audio Question)"})
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
                # Play Audio Response
                text_to_speech(answer_text)

            except Exception as e:
                st.error(f"An error occurred: {e}")

            finally:
                # Cleanup
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                # Note: We skip deleting 'myfile' from cloud to speed up response, 
                # but you can add genai.delete_file(myfile.name) here if desired.
                
                # Rerun to update chat history
                st.rerun()

else:
    st.info("👆 Please upload a file to start.")