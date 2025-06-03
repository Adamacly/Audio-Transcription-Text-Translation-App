import whisper
import gradio as gr
import datetime
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = whisper.load_model('base')

translation_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
translation_model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
translation_model.to(device)

def transcribe(inputs , timestamp):
    if inputs is None:
          raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
    output = ""
    result = model.transcribe(inputs)
    if timestamp == "Yes":
      for indx, segment in enumerate(result['segments']):
        output += str(datetime.timedelta (seconds=segment['start'])) +" "+ str(datetime.timedelta (seconds=segment['end'])) + "\n"
        output += segment['text'].strip() + '\n'
    else:
      output = result["text"]

    return  output

# Fonction de traduction
def translate(text):
    inputs = translation_tokenizer(text, return_tensors="pt").to(device)
    lang_id = translation_tokenizer.convert_tokens_to_ids("eng_Latn")
    translated_tokens = translation_model.generate(
        **inputs, forced_bos_token_id=lang_id, max_length=64
    )
    return translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Fonction principale pour Gradio
def process(audio, timestamp):
    transcription = transcribe(audio, timestamp)
    translation = translate(transcription)
    return transcription, translation


interface = gr.Interface(
    fn=process,
    inputs=[gr.Audio(sources=["upload"],type="filepath"),
            gr.Radio(["Yes", "No"], label="Timestamp", info="Displays with timestamp if needed."),],
    outputs=[
        gr.Textbox(label="Transcription (Français)"),
        gr.Textbox(label="Traduction (Anglais)")
    ],
    title="Audio Transcription And Translation App FR->GB",
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the OpenAI Whisper API"
    )
)

interface.launch()
