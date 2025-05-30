import torch
from transformers import pipeline
import gradio as gr

# Initialize the speech recognition pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    chunk_length_s=30,
    device=-1,  # Use -1 for CPU, or specify a GPU device ID (e.g., 0) if available
)

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    # Transcribe the audio file and return the result
    result = pipe(audio_file, batch_size=8)["text"]
    return result

# Set up Gradio interface
audio_input = gr.Audio(sources="upload", type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn=transcript_audio, 
                     inputs=audio_input, outputs=output_text, 
                     title="Audio Transcription",
                     description="This is a simple web app for audio transcription (English-only) using Whisper model from OpenAI.")

# Launch the Gradio app
iface.launch()