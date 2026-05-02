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
        # Adjust for ambient noise briefly before opening the hot mic
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        try:
            # timeout=5 means if you are silent for 5s, it throws an error (which we catch)
            # phrase_time_limit=15 means it stops recording after 15 seconds of talking
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            
            with open("temp_cmd.wav", "wb") as f:
                f.write(audio_data.get_wav_data())
            
            # Transcribe locally with Whisper
            result = whisper_model.transcribe("temp_cmd.wav", fp16=False)
            command = result["text"].strip()
            
            return command
            
        except sr.WaitTimeoutError:
            # If no speech is detected, return None so the loop can seamlessly restart
            return None
        finally:
            if os.path.exists("temp_cmd.wav"):
                os.remove("temp_cmd.wav")

# ==========================================
# JARVIS MEMORY MODULE
# ==========================================
chat_history = []
MAX_HISTORY = 10  # Keep the last 10 exchanges in memory

def query_brain(prompt):
    """Feeds the text to Ollama with strict OS rules AND conversational memory."""
    global chat_history
    print("[Jarvis]: Thinking...")
    
    # 1. The Permanent System Prompt (Jarvis's Core Identity & Rules)
    system_message = {
        'role': 'system',
        'content': '''You are Jarvis, a highly efficient AI assistant running on a MacBook. 
        Keep conversational responses to 1 short sentence. 
        
        CRITICAL INSTRUCTIONS FOR COMMANDS:
        1. LOCAL APPS: If asked to open a desktop application (like Brave, Firefox, Terminal, Spotify), output EXACTLY: <OPEN: AppName>
        2. WEBSITES: If asked to open a web service, or if the user explicitly says "in browser" or ".com" (like Gmail, YouTube, GitHub), output EXACTLY: <WEBSITE: https://www.url.com>
        
        STRICT HEURISTICS:
        - If the prompt contains "Gmail", "Google", "Netflix", or "YouTube", treat it as a WEBSITE.
        - If the prompt contains "in browser", ALWAYS treat it as a WEBSITE.
        - If the prompt contains "Brave", "Safari", "Chrome", or "Finder", treat it as an APP.
        
        If you output a tag, do NOT output any conversational text with it. Just the tag.'''
    }
    
    # 2. Add your current spoken command to the memory bank
    chat_history.append({'role': 'user', 'content': prompt})
    
    # 3. Prune the memory if it gets too long (Saves RAM and processing speed)
    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]
        
    # 4. Combine the System Rules with the short-term memory
    messages = [system_message] + chat_history
    
    try:
        # Note: Swap 'llama3.2' to 'llama3.1' here if you downloaded the smarter 8B model!
        response = ollama.chat(model='llama3.1', messages=messages)
        reply_content = response['message']['content']
        
        # 5. Save Jarvis's reply to the memory bank before returning it
        chat_history.append({'role': 'assistant', 'content': reply_content})
        
        return reply_content
        
    except Exception as e:
        # If there's an error, remove your last prompt from memory so it doesn't corrupt the history
        if chat_history:
            chat_history.pop()
        return f"Sir, I encountered an error: {e}"

def execute_action(reply):
    """Scans the AI's reply for Web or App command tags and executes them."""
    
    # 1. Check for Website Intent first
    web_match = re.search(r'<WEBSITE:\s*(.+?)>', reply, re.IGNORECASE)
    if web_match:
        url = web_match.group(1).strip()
        print(f"[Jarvis]: System Command -> Open URL {url}")
        os.system('say "Opening website."')
        os.system(f'open "{url}"') # Launch default browser to URL
        return True

    # 2. Check for App Intent
    app_match = re.search(r'<OPEN:\s*(.+?)>', reply, re.IGNORECASE)
    if app_match:
        app_name = app_match.group(1).strip()
        print(f"[Jarvis]: System Command -> Open App {app_name}")
        os.system(f'say "Opening {app_name}."')
        os.system(f'open -a "{app_name}"') # Force launch local Mac app
        return True
        
    return False # No action tag found

# ==========================================
# 4. Main Event Loop
# ==========================================
print("\n[Jarvis]: Online. Say 'Hey Jarvis' to wake me up. (Press Ctrl+C to stop)")

mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

try:
    while True:
        # Constantly listen for the wake word
        pcm = mic_stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(pcm, dtype=np.int16)
        prediction = owwModel.predict(audio_data)
        
        model_key = list(prediction.keys())[0]
        
        # --- WAKE WORD DETECTED ---
        if prediction[model_key] > 0.5:
            print("\n[Jarvis]: Yes,sir?")
            os.system('say "Yes sir?"')
            owwModel.reset() 
            mic_stream.stop_stream() # Pause the wake word mic
            
            is_awake = True
            
            # --- CONTINUOUS LISTENING LOOP ---
            while is_awake:
                print("\n[Jarvis]: (Awaiting command...)")
                command = listen_for_command()
                
                # If you were silent, command is None. Just restart the loop quietly.
                if not command:
                    continue
                
                print(f"[You]: {command}")
                
                # Check for the dismissal phrase
                clean_cmd = command.lower()
                if "thank you" in clean_cmd and "jarvis" in clean_cmd:
                    print("[Jarvis]: You are welcome, sir. Returning to standby.")
                    os.system('say "You are welcome, sir. Returning to standby."')
                    is_awake = False
                    break # Exit the continuous loop
                
                # Pass to the brain if it wasn't a dismissal
                reply = query_brain(command)
                
                # Intercept OS commands
                action_executed = execute_action(reply)
                
                # Speak normal conversation if no action was taken
                if not action_executed:
                    print(f"[Jarvis]: {reply}")
                    safe_reply = reply.replace('"', '').replace("'", "").replace('\n', ' ')
                    os.system(f'say -v Samantha "{safe_reply}"')
                    
            # ---------------------------------
            
            print("\n[Jarvis]: Standby mode active.")
            mic_stream.start_stream() # Resume wake word mic

except KeyboardInterrupt:
    print("\nShutting down Jarvis...")
finally:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()