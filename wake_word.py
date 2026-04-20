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

# Suppress FP16 warnings from Whisper on Mac CPUs to keep the terminal clean
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
    """Handles grabbing the microphone and recording your command."""
    with sr.Microphone() as source:
        print("\n[Jarvis]: Listening for command...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            # Listen for up to 5 seconds of silence, max phrase length 10s
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Save it temporarily
            with open("temp_cmd.wav", "wb") as f:
                f.write(audio_data.get_wav_data())
            
            print("[Jarvis]: Transcribing...")
            # Transcribe locally with Whisper
            result = whisper_model.transcribe("temp_cmd.wav", fp16=False)
            command = result["text"].strip()
            
            print(f"[You]: {command}")
            return command
            
        except sr.WaitTimeoutError:
            print("[Jarvis]: I didn't hear anything.")
            return None
        finally:
            if os.path.exists("temp_cmd.wav"):
                os.remove("temp_cmd.wav")

def query_brain(prompt):
    """Feeds the transcribed text to the local LLM with strict action rules."""
    print("[Jarvis]: Thinking...")
    try:
        response = ollama.chat(model='llama3.2', messages=[
            {
                'role': 'system',
                'content': '''You are Jarvis, a highly efficient AI assistant running on a MacBook. 
                Keep conversational responses to 1 sentence. 
                CRITICAL INSTRUCTION: If the user asks you to open an application, website, or tool, DO NOT reply with normal text. 
                Instead, you must output EXACTLY this format: <OPEN: ApplicationName>
                Example: If the user says "Open my browser", output: <OPEN: Safari>
                Example: If the user says "Launch Spotify", output: <OPEN: Spotify>'''
            },
            {
                'role': 'user',
                'content': prompt
            }
        ])
        return response['message']['content']
    except Exception as e:
        return f"Sir, I encountered an error: {e}"

def execute_action(reply):
    """Scans the AI's reply for command tags and executes them."""
    # Look for the <OPEN: AppName> pattern
    match = re.search(r'<OPEN:\s*(.+?)>', reply, re.IGNORECASE)
    
    if match:
        app_name = match.group(1).strip()
        print(f"[Jarvis]: Executing system command -> Open {app_name}")
        
        # Speak confirmation
        os.system(f'say "Right away, sir. Opening {app_name}."')
        
        # Execute the macOS command to open the application
        os.system(f'open -a "{app_name}"')
        return True # Action was handled
        
    return False # No action tag found

# ==========================================
# 4. Main Event Loop
# ==========================================
print("\n[Jarvis]: Online. Say 'Hey Jarvis' to wake me up. (Press Ctrl+C to stop)")

# Open the microphone stream for the wake word
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

try:
    while True:
        # Constantly listen for the wake word
        pcm = mic_stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(pcm, dtype=np.int16)
        prediction = owwModel.predict(audio_data)
        
        # Dynamically get the model's key name to avoid KeyError
        model_key = list(prediction.keys())[0]
        
        # Wake word detected!
        if prediction[model_key] > 0.5:
            print("\n[Jarvis]: Yes, sir?")
            os.system('say "Yes, sir?"')
            owwModel.reset() # Clear buffer
            
            # PAUSE the wake word stream so they don't fight over the mic
            mic_stream.stop_stream()
            
            # Record and transcribe the command
            command = listen_for_command()
            
            if command:
                # Pass the text to Ollama
                reply = query_brain(command)
                
                # Intercept and check for system commands
                action_executed = execute_action(reply)
                
                # If no action was executed, just speak the reply normally
                if not action_executed:
                    print(f"[Jarvis]: {reply}")
                    safe_reply = reply.replace('"', '').replace("'", "").replace('\n', ' ')
                    os.system(f'say "{safe_reply}"')
            
            print("\n[Jarvis]: Returning to standby. Say 'Hey Jarvis' to wake me up.")
            # RESUME the wake word stream
            mic_stream.start_stream()

except KeyboardInterrupt:
    print("\nShutting down Jarvis...")
finally:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()