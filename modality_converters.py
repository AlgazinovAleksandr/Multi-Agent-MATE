from typing_extensions import Annotated
import os
import json
import io
import nbformat
import zipfile
import shutil
import re

import os
from typing import Optional
from TTS.api import TTS

from PIL import Image
import whisper
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from typing import Optional, Tuple
from dataclasses import dataclass
from typing import List, Tuple, Annotated
# from deep_translator import GoogleTranslator

import fitz
import docx

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        text = text.replace('\n', '').replace('\r', '').strip()
        return text
    elif ext == '.docx':
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format. Supported formats: .txt, .pdf, .docx")

def transcribe_audio_save(audio_path: str, output_path: str, save=True):
    try:
        model = whisper.load_model("base")
        print(f"Audio path : {audio_path}")
        result = model.transcribe(audio_path)
        transcript = result["text"]
        
        #base_name = os.path.splitext(audio_path)[0]
        base_name = audio_path.split('/')[-1]
        save_path = f"{output_path}/{base_name}_transcript.txt"
        if save:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(transcript)
                print(f"Transcript saved to {save_path}")
        return transcript, save_path
    except Exception as e:
        return f"Error processing audio: {str(e)}", ""
    
def transcribe_image_save(image_path: str, output_path: str, model_name: str, save=True):
    try:
        img = Image.open(image_path)
        if model_name == 'BLIP':
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            inputs = processor(img, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        #base_name = os.path.splitext(image_path)[0]
        base_name = image_path.split('/')[-1]
        save_path = f"{output_path}/{base_name}_caption.txt"
        if save:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(caption)
            print(f"Caption saved to {save_path}")
        return caption, save_path
    except Exception as e:
        return f"Error processing image: {str(e)}", ""
    
#vits_model = "tts_models/en/vctk/vits"
def text_to_speech_save(input: str, output_path: str, inputIsFile: bool, model_name: str):
    if inputIsFile:
        offset = 0
        text = extract_text_from_file(input)
    else:
        offset = 1
        text = input
    base_name = re.sub(r'\s+', '_', input.split('/')[-1][offset:min(len(input) - offset * 2, 10 + offset)])
    save_path = f"{output_path}/{base_name}_audio.wav"
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
    output_path: str,
    image_model_name: str,
    tts_model_name: str
):
    print(f"--- Starting Image-to-Audio conversion for: {image_path} ---")
    # Step 1: Generate a caption from the image and save it to a .txt file
    print("\n[Step 1/2] Generating image caption...")
    caption, caption_path = transcribe_image_save(
        image_path=image_path,
        output_path=output_path,
        model_name=image_model_name
    )
    if not caption_path:
        print("Image captioning failed. Aborting process.")
        return None
    print(f"Generated Caption: '{caption}'")
    # Step 2: Use the generated .txt file to create an audio file
    print("\n[Step 2/2] Converting caption to speech...")
    audio_path = text_to_speech_save(
        input=caption_path,
        output_path=output_path,
        inputIsFile=True,
        model_name=tts_model_name
    )
    if not audio_path:
        print("Text-to-speech conversion failed.")
        return None
    print(f"\n--- Success! Final audio file created: {audio_path} ---")
    return audio_path
  
def text_to_image(input: str, output_path: str, inputIsFile: bool, model_name: str):
  # Initialization
    if inputIsFile:
        text = extract_text_from_file(input)
    else:
        text = input
    base_name = re.sub(r'\s+', '_', input.split('/')[-1][1:min(len(input) - 2, 11)])
    save_path = f"{output_path}/{base_name}_image.png"

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # Move the pipeline to the GPU
    pipe = pipe.to("cuda")
    # Initialization complete
    # Generate the image
    image = pipe(input).images[0]
    # Save the image
    image.save(save_path)
    return save_path
  
def audio_to_image(input: str, output_path: str, model_name: str):
  # Initialization

    print(f"--- Starting Audio-to-Image conversion for: {input} ---")
    print("\n[Step 1/2] Generating audio transcription...")
    transcription, transcription_path = transcribe_audio_save(input, output_path)
    print(f"\n[Step 2/2] Generating image from text... {transcription}")
    result_path = text_to_image(transcription, output_path, False, model_name)
    if not result_path:
      print("Text to image conversion failed.")
      return None
    print(f"\n--- Success! Final image file created: {result_path} ---")
    return result_path