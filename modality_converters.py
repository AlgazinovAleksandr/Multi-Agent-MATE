from typing_extensions import Annotated
import os
import json
import io
import nbformat
import zipfile
import shutil

import os
from typing import Optional
from TTS.api import TTS

from PIL import Image
import whisper
import os
from langdetect import detect
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Optional, Tuple
from dataclasses import dataclass
from typing import List, Tuple, Annotated
from deep_translator import GoogleTranslator

def transcribe_audio_save(audio_path, save=True):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        transcript = result["text"]
        
        base_name = os.path.splitext(audio_path)[0]
        save_path = f"{base_name}_transcript.txt"
        if save:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(transcript)
                print(f"Transcript saved to {save_path}")
        return transcript, save_path
    except Exception as e:
        return f"Error processing audio: {str(e)}", ""
    
def transcribe_image_save(image_path, model_name='BLIP', save=True):
    try:
        img = Image.open(image_path)
        if model_name == 'BLIP':
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            inputs = processor(img, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        base_name = os.path.splitext(image_path)[0]
        save_path = f"{base_name}_caption.txt"
        if save:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(caption)
            print(f"Caption saved to {save_path}")
        return caption, save_path
    except Exception as e:
        return f"Error processing image: {str(e)}", ""
    
#vits_model = "tts_models/en/vctk/vits"
def text_to_speech_save(text_path, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()
    base_name = os.path.splitext(text_path)[0]
    save_path = f"{base_name}_audio.wav"
    try:
        # Initialize the TTS model
        tts = TTS(model_name=model_name, progress_bar=True, gpu=False)
        tts.tts_to_file(text=text, file_path=save_path)
        print(f"Audio saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def image_to_audio(
    image_path: str,
    image_model_name: str = 'BLIP',
    tts_model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
):
    print(f"--- Starting Image-to-Audio conversion for: {image_path} ---")
    # Step 1: Generate a caption from the image and save it to a .txt file
    print("\n[Step 1/2] Generating image caption...")
    caption, caption_path = transcribe_image_save(
        image_path=image_path,
        model_name=image_model_name
    )
    if not caption_path:
        print("Image captioning failed. Aborting process.")
        return None
    print(f"Generated Caption: '{caption}'")
    # Step 2: Use the generated .txt file to create an audio file
    print("\n[Step 2/2] Converting caption to speech...")
    audio_path = text_to_speech_save(
        text_path=caption_path,
        model_name=tts_model_name
    )
    if not audio_path:
        print("Text-to-speech conversion failed.")
        return None
    print(f"\n--- Success! Final audio file created: {audio_path} ---")
    return audio_path