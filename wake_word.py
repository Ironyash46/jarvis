import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
import os

# 1. Download pre-trained models if they don't exist (One-time download)
print("Checking for openWakeWord models...")
openwakeword.utils.download_models()

# 2. Load the open-source "Hey Jarvis" model
print("Loading openWakeWord model...")
owwModel = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

# 3. Set up the microphone stream using PyAudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("\n[Jarvis]: Online. Say 'Hey Jarvis' to wake me up. (Press Ctrl+C to stop)")

try:
    while True:
        # 4. Read audio data from the microphone
        pcm = mic_stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(pcm, dtype=np.int16)

        # 5. Feed the raw audio data into the model
        prediction = owwModel.predict(audio_data)

# 6. Check if the confidence score is high enough
        # Dynamically get the model's key name so we don't get a KeyError
        model_key = list(prediction.keys())[0] 
        
        if prediction[model_key] > 0.5:
            print("\n[Jarvis]: Yes, sir?")
            os.system('say "Yes, sir?"')
            
            # Clear the model's internal buffer so it doesn't double-trigger
            owwModel.reset()
            
except KeyboardInterrupt:
    print("\nShutting down Jarvis...")
finally:
    # Always close the mic stream cleanly
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()

