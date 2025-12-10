import streamlit as st
import google.generativeai as genai
import pandas as pd
from gtts import gTTS
import tempfile
import os
import time
from streamlit_mic_recorder import mic_recorder, speech_to_text 

# --- CONFIGURATION (FIXED MODEL INITIALIZATION) ---

model_name = 'gemini-2.5-flash'

# --- NEW: SYSTEM INSTRUCTION TEMPLATE (MOVED) ---
# We define the template here, but the specific data will be injected later.
SYSTEM_INSTRUCTION_BASE = """
You are a helpful retail stock assistant. All your answers must be based *strictly* on 
the provided stock data, which includes columns like Item_Name, Quantity, Price, etc.
The data available is (first 10 rows):
{csv_data_string}

Always keep the answers conversational and concise. Acknowledge that you have the stock data.
"""

# The model must be re-initialized when a new file is uploaded/context is set,
# so we'll do a basic initialization here, and a final one inside the main logic.
model = genai.GenerativeModel(model_name)

# --- NEW: Initialize Chat History and Chat Session in Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    st.session_state.chat = None
if "context_loaded" not in st.session_state:
    st.session_state.context_loaded = False # New state to control re-initialization

# --- Function to convert Text to Speech and Play (No Change) ---
def text_to_speech(text):
    """Converts text to speech and plays it using Streamlit's audio element."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_tts:
        tts = gTTS(text=text, lang='en')
        tts.write_to_fp(temp_tts)
        temp_tts_path = temp_tts.name
    
    st.audio(temp_tts_path, format='audio/mp3', autoplay=True, loop=False)
    time.sleep(0.5) 
    os.remove(temp_tts_path)

# --- Function to get the current stock data string (No Change) ---
def get_stock_data_string(uploaded_file):
    """Reads the uploaded file (CSV/Excel) and returns its content as a string."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df.head(10).to_string(index=False)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

# --- UI SETUP (No Change) ---
st.title("🛒 Retail Stock Voice Assistant")
st.write("Upload a CSV/Excel, click the mic, and start a conversation!")

uploaded_file = st.file_uploader("Upload your Stock CSV/Excel", type=["csv", "xlsx"])

# --- Main App Logic ---
if uploaded_file:
    # 1. Get the stock data string immediately
    csv_string = get_stock_data_string(uploaded_file)
    
    # 2. Add a system prompt and initialize chat (only once per file upload)
    # Re-initialize the model *with the configuration* only when the file is uploaded.
    if not st.session_state.context_loaded:
        
        # 🎯 FIX: Inject the CSV data into the system instruction template
        final_system_instruction = SYSTEM_INSTRUCTION_BASE.format(csv_data_string=csv_string)
        
        # 🎯 FIX: Pass the system instruction in the model's configuration upon initialization
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=final_system_instruction # This is the old/correct location
        )
        
        # Now, start the chat WITHOUT the system_instruction keyword argument
        st.session_state.chat = model.start_chat() 
        
        # Set state flags
        st.session_state.context_loaded = True 
        st.session_state.messages = [] # Clear history for new file
        
        initial_message = "Hello! I have loaded your stock data. Ask me anything about the inventory!"
        st.session_state.messages.append({"role": "assistant", "content": initial_message})
        text_to_speech(initial_message)
        st.experimental_rerun()


    # 3. Display chat history (No Change)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 4. VOICE INPUT (Microphone Component)
    st.write("---")
    audio_data = mic_recorder(
        start_prompt="Click to Start Recording",
        stop_prompt="Click to Stop Recording",
        just_once=True,
        use_container_width=True,
        format="wav"
    )

    if audio_data and audio_data['bytes']:
        st.info("Recording received. Analyzing...")
        
        # Save the audio data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data['bytes'])
            temp_audio_path = temp_audio.name
            
        myfile = None
        try:
            # FIX: Use genai.upload_file() instead of client.files.upload()
            myfile = genai.upload_file(temp_audio_path) 
            
            # Send Message to Chat API (No Change to this line)
            response = st.session_state.chat.send_message(
                contents=[myfile],
                config={'temperature': 0.1}
            )
            answer_text = response.text

            # --- 5. Update History and Output ---
            st.session_state.messages.append({"role": "user", "content": "(Audio Question Received)"}) 
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
            
            st.success("Assistant Response:")
            st.write(answer_text)
            text_to_speech(answer_text)

        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        finally:
            os.remove(temp_audio_path)
            # FIX: Use genai.delete_file() instead of client.files.delete()
            if myfile:
                 genai.delete_file(name=myfile.name) 
            
            st.experimental_rerun() 

else:
    # Reset states when file is not uploaded
    st.session_state.chat = None
    st.session_state.context_loaded = False
    st.session_state.messages = []
    st.info("Please upload a CSV/Excel file to begin.")