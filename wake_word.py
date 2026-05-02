import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
import os
import whisper
import speech_recognition as sr
import warnings
import ollama
import re

import eyes_jarvis

# Suppress FP16 warnings from Whisper on Mac CPUs
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# ==========================================
# 1. Initialization & Loading Models
# ==========================================
print("Checking for openWakeWord models...")
openwakeword.utils.download_models()
print("Loading openWakeWord model...")
owwModel = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

print("Loading local Whisper model (base)...")
whisper_model = whisper.load_model("base")
recognizer = sr.Recognizer()

# ==========================================
# 2. Audio Configuration
# ==========================================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()

# ==========================================
# 3. Core Functions
# ==========================================
def listen_for_command():
    """Listens continuously. Quietly times out and retries if you are silent."""
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        try:
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            
            with open("temp_cmd.wav", "wb") as f:
                f.write(audio_data.get_wav_data())
            
            result = whisper_model.transcribe("temp_cmd.wav", fp16=False)
            command = result["text"].strip()
            
            return command
            
        except sr.WaitTimeoutError:
            return None
        finally:
            if os.path.exists("temp_cmd.wav"):
                os.remove("temp_cmd.wav")

# ==========================================
# JARVIS MEMORY MODULE
# ==========================================
chat_history = []
MAX_HISTORY = 10 

def query_brain(prompt):
    """Feeds the text to Ollama with strict OS rules AND conversational memory."""
    global chat_history
    print("[Jarvis]: Thinking...")
    
    # --- UPDATED: Added Tab Closing Instructions ---
    system_message = {
        'role': 'system',
        'content': '''You are Jarvis, a highly efficient AI assistant running on a MacBook. 
        Keep conversational responses to 1 short sentence. 
        
        CRITICAL INSTRUCTIONS FOR COMMANDS:
        1. LOCAL APPS: If asked to open a desktop application (like Brave, Firefox, Terminal, Spotify), output EXACTLY: <OPEN: AppName>
        2. WEBSITES: If asked to open a web service, output EXACTLY: <WEBSITE: https://www.url.com>
        3. CLOSING TABS: If asked to close a tab, website, or window (e.g., "close youtube", "close the gmail tab"), output EXACTLY: <CLOSE_TAB: target_name> (e.g., <CLOSE_TAB: youtube>).
        
        STRICT HEURISTICS:
        - If the prompt contains "Gmail", "Google", "Netflix", or "YouTube" to OPEN, treat it as a WEBSITE.
        - If the prompt contains "in browser", ALWAYS treat it as a WEBSITE.
        - If the prompt contains "close" or "quit" alongside a website name, ALWAYS use the <CLOSE_TAB: target> tag.
        
        If you output a tag, do NOT output any conversational text with it. Just the tag.'''
    }
    
    chat_history.append({'role': 'user', 'content': prompt})
    
    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]
        
    messages = [system_message] + chat_history
    
    try:
        response = ollama.chat(model='llama3.1', messages=messages)
        reply_content = response['message']['content']
        
        chat_history.append({'role': 'assistant', 'content': reply_content})
        return reply_content
        
    except Exception as e:
        if chat_history:
            chat_history.pop()
        return f"Sir, I encountered an error: {e}"

def execute_action(reply):
    """Scans the AI's reply for Web, App, or Close command tags and executes them."""
    
    # 1. Check for Website Intent
    web_match = re.search(r'<WEBSITE:\s*(.+?)>', reply, re.IGNORECASE)
    if web_match:
        url = web_match.group(1).strip()
        print(f"[Jarvis]: System Command -> Open URL {url}")
        os.system('say -v Samantha "Opening website."')
        os.system(f'open "{url}"') 
        return True

    # 2. Check for App Intent
    app_match = re.search(r'<OPEN:\s*(.+?)>', reply, re.IGNORECASE)
    if app_match:
        app_name = app_match.group(1).strip()
        print(f"[Jarvis]: System Command -> Open App {app_name}")
        os.system(f'say -v Samantha "Opening {app_name}."')
        os.system(f'open -a "{app_name}"') 
        return True
        
    # --- NEW: 3. Check for Close Tab Intent ---
    close_match = re.search(r'<CLOSE_TAB:\s*(.+?)>', reply, re.IGNORECASE)
    if close_match:
        target = close_match.group(1).strip()
        print(f"[Jarvis]: System Command -> Scanning for tab '{target}'")
        
        # Call the eyes_jarvis module to find and execute the close command
        success = eyes_jarvis.find_and_close_tab(target)
        
        if success:
            os.system(f'say -v Samantha "I have closed the {target} tab, sir."')
        else:
            os.system(f'say -v Samantha "I could not locate a tab matching {target}, sir."')
            
        return True
        
    return False # No action tag found

# ==========================================
# 4. Main Event Loop
# ==========================================
print("\n[Jarvis]: Online. Say 'Hey Jarvis' to wake me up. (Press Ctrl+C to stop)")

mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

try:
    while True:
        pcm = mic_stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(pcm, dtype=np.int16)
        prediction = owwModel.predict(audio_data)
        
        model_key = list(prediction.keys())[0]
        
        if prediction[model_key] > 0.5:
            print("\n[Jarvis]: Yes, sir?")
            os.system('say -v Samantha "Yes sir?"')
            owwModel.reset() 
            mic_stream.stop_stream() 
            
            is_awake = True
            
            while is_awake:
                print("\n[Jarvis]: (Awaiting command...)")
                command = listen_for_command()
                
                if not command:
                    continue
                
                print(f"[You]: {command}")
                
                clean_cmd = command.lower()
                if "thank you" in clean_cmd and "jarvis" in clean_cmd:
                    print("[Jarvis]: You are welcome, sir. Returning to standby.")
                    os.system('say -v Samantha "You are welcome, sir. Returning to standby."')
                    is_awake = False
                    break 
                
                reply = query_brain(command)
                action_executed = execute_action(reply)
                
                if not action_executed:
                    print(f"[Jarvis]: {reply}")
                    safe_reply = reply.replace('"', '').replace("'", "").replace('\n', ' ')
                    os.system(f'say -v Samantha "{safe_reply}"')
                    
            print("\n[Jarvis]: Standby mode active.")
            mic_stream.start_stream() 

except KeyboardInterrupt:
    print("\nShutting down Jarvis...")
finally:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()