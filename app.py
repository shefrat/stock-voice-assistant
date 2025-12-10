import streamlit as st
import google.generativeai as genai
import pandas as pd
from gtts import gTTS
import tempfile
import os
import time # Added for cleanup/re-run control
from streamlit_mic_recorder import mic_recorder, speech_to_text # Import the custom component

# --- CONFIGURATION (Keep this) ---
# Ensure your API key is set as an environment variable (recommended)
# genai.configure(api_key=os.environ.get('GEMINI_API_KEY')) 
# If not using environment variable, you must set it here:
# genai.configure(api_key="YOUR_API_KEY") 

# We use Gemini 2.5 Flash as it's fast and multimodal (hears audio + reads files)
model_name = 'gemini-2.5-flash'
client = genai.Client()
model = client.models.get(model_name)

# --- NEW: Initialize Chat History and Chat Session in Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    # This will be initialized later once the file is uploaded
    st.session_state.chat = None

# --- Function to convert Text to Speech and Play ---
def text_to_speech(text):
    """Converts text to speech and plays it using Streamlit's audio element."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_tts:
        tts = gTTS(text=text, lang='en')
        tts.write_to_fp(temp_tts)
        temp_tts_path = temp_tts.name
    
    st.audio(temp_tts_path, format='audio/mp3', autoplay=True, loop=False)
    # Give Streamlit time to play the audio before cleanup
    time.sleep(0.5) 
    os.remove(temp_tts_path)

# --- Function to get the current stock data string ---
def get_stock_data_string(uploaded_file):
    """Reads the uploaded file (CSV/Excel) and returns its content as a string."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        # Using a limited number of rows for the prompt to save token space
        return df.head(10).to_string(index=False)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

# --- UI SETUP ---
st.title("🛒 Retail Stock Voice Assistant")
st.write("Upload a CSV/Excel, click the mic, and start a conversation!")

uploaded_file = st.file_uploader("Upload your Stock CSV/Excel", type=["csv", "xlsx"])

# --- Main App Logic ---
if uploaded_file:
    # 1. Get the stock data string immediately
    csv_string = get_stock_data_string(uploaded_file)
    
    # 2. Add a system prompt and initialize chat (only once per file upload)
    if st.session_state.chat is None:
        # The initial instruction is sent to set the context for the chat
        system_instruction = f"""
        You are a helpful retail stock assistant. All your answers must be based *strictly* on 
        the provided stock data, which includes columns like Item_Name, Quantity, Price, etc.
        The data available is (first 10 rows):
        {csv_string}

        Always keep the answers conversational and concise. Acknowledge that you have the stock data.
        """
        st.session_state.chat = model.start_chat(system_instruction=system_instruction)
        
        # Add initial assistant message to history and re-run
        initial_message = "Hello! I have loaded your stock data. Ask me anything about the inventory!"
        st.session_state.messages.append({"role": "assistant", "content": initial_message})
        text_to_speech(initial_message) # Speak the welcome message
        st.experimental_rerun()

    # 3. Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 4. VOICE INPUT (Microphone Component)
    st.write("---")
    # Using the mic_recorder component
    audio_data = mic_recorder(
        start_prompt="Click to Start Recording",
        stop_prompt="Click to Stop Recording",
        just_once=True, # Stop after a single recording
        use_container_width=True,
        format="wav" # Best for clarity
    )

    # The mic_recorder component returns data when recording stops
    if audio_data and audio_data['bytes']:
        st.info("Recording received. Analyzing...")
        
        # Save the audio data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data['bytes'])
            temp_audio_path = temp_audio.name
            
        try:
            # Upload the audio part
            myfile = client.files.upload(file=temp_audio_path)
            
            # --- Send Message to Chat API ---
            # The Chat API handles the history automatically. We send the new audio part.
            response = st.session_state.chat.send_message(
                contents=[myfile],
                config={'temperature': 0.1}
            )
            answer_text = response.text

            # --- 5. Update History and Output ---
            # Append user input
            st.session_state.messages.append({"role": "user", "content": "(Audio Question Received)"}) 
            
            # Append model response
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
            
            # Convert text to speech and play
            st.success("Assistant Response:")
            st.write(answer_text)
            text_to_speech(answer_text)

        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        finally:
            # Clean up temporary files
            os.remove(temp_audio_path)
            client.files.delete(name=myfile.name) # Delete the file from Gemini
            
            # Rerun to display the new messages and reset the mic component
            st.experimental_rerun() 

else:
    # Reset chat state when file is not uploaded
    st.session_state.chat = None
    st.info("Please upload a CSV/Excel file to begin.")