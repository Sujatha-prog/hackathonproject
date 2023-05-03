import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask import Flask, jsonify, request, render_template, url_for
from gtts import gTTS
from tempfile import TemporaryFile
import base64
import os

app = Flask(__name__)

# Load the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Define the API endpoint for summarization
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the input text from the request
    input_text = request.form['input_text']


    # Tokenize the input text and add special tokens for summarization
    input_ids = tokenizer.encode("summarize: " + input_text, return_tensors='pt', max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(input_ids, num_beams=4, max_length=150, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Get the value of the dropdown input from the request
    dropdown_value = request.form['my-dropdown']
    # Create a gTTS object with the text and language
    tts = gTTS(text=summary, lang=dropdown_value )
    # Save the audio file as a temporary file
    audio_file = "speech.mp3"
    tts.save(audio_file)
    # Encode the audio file as base64
    with open(audio_file, 'rb') as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('ascii')

    # Render the template with the audio file encoded as base64
    return render_template('index.html', audio_base64=audio_base64,summary=summary)



if __name__ == '__main__':
    app.run(debug=True)