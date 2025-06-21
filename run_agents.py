import sys
from src.interpreter_agent import LLMInterpreter, BertInterpreter
from src.agent import Agent, TextToSpeech, TextToImage, AudioToImage, SpeechToText, ImageToAudio, ImageToText, VideoToText
import os
from constants import possible_output_dir, default_output_dir

import re
from typing import Tuple, Optional

def check_file_path(user_prompt: str) -> Tuple[bool, Optional[str]]:
    # Supported file extensions
    extensions = ['txt', 'docx', 'pdf', 'jpeg', 'jpg', 'png', 'mp4', 'mp3', 'wav', 'gif', 'bmp', 'avi', 'mov', 'mkv', 'webm']
    # Regex pattern to match file paths with supported extensions
    pattern = r'([\w\-/\\\.]+\.(' + '|'.join(extensions) + r'))\b'
    match = re.search(pattern, user_prompt, re.IGNORECASE)
    if match:
        return True, match.group(1)
    else:
        return False, None
    
# Determine output directory

output_dir = ""
for possible_dir in possible_output_dir:
    if os.path.isdir(possible_dir):
        output_dir = possible_dir
        break

if output_dir == "":
    dir_name = default_output_dir
    try:
        os.mkdir(dir_name)
        output_dir = dir_name
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{dir_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Initialize agents
# interpreter = BertInterpreter(model_path="CLASSIFIER/saved_models/best_model_bert_epoch_6.pt", 
                                #    encoder_path="CLASSIFIER/saved_models/label_encoder.joblib")
interpreter = LLMInterpreter()
TTSAgent = TextToSpeech(output_dir)
TTIAgent = TextToImage(output_dir)
ATIAgent = AudioToImage(output_dir)
STTAgent = SpeechToText(output_dir)
ITTAgent = ImageToText(output_dir)
ITAAgent = ImageToAudio(output_dir)
VTTAgent = VideoToText(output_dir)

def convert_modality(conversion_type, prompt):

    result_path = ""
    if (conversion_type == "TTS"):
        inputSource = TTSAgent.querySource(prompt)
        result_path = TTSAgent.convertFile(inputSource)
    elif (conversion_type == "TTI"):
        inputSource = TTIAgent.querySource(prompt)
        result_path = TTIAgent.convertFile(inputSource)
    elif (conversion_type == "ATI"):
        inputSource = ATIAgent.querySource(prompt)
        result_path = ATIAgent.convertFile(inputSource)
    elif (conversion_type == "STT"):
        inputSource = STTAgent.querySource(prompt)
        result_path = STTAgent.convertFile(inputSource)
    elif (conversion_type == "ITT"):
        inputSource = ITTAgent.querySource(prompt)
        result_path = ITTAgent.convertFile(inputSource)
    elif (conversion_type == "ITA"):
        inputSource = ITAAgent.querySource(prompt)
        result_path = ITAAgent.convertFile(inputSource)
    elif (conversion_type == "VTT"):
        inputSource = VTTAgent.querySource(prompt)
        result_path = VTTAgent.convertFile(inputSource)
    elif (conversion_type == "UNK"):
        print("Prompt does not mention any form of file or modality conversion")
        prompt = input("I'm sorry, I didn't understand your request, please specify the task you want me to help you with : ")
        conversion_type = interpreter.get_task_type(prompt)
        result_path = convert_modality(conversion_type, prompt)
    else:
        prompt = input("I'm sorry, this modality conversion is not yet handled, please specify the task you want me to help you with : ")
        conversion_type = interpreter.get_task_type(prompt)
        result_path = convert_modality(conversion_type, prompt)

    # Get the resulting file and return it to the user

    if result_path != "":
        print(f"resulting path : {result_path}")
    return result_path

prompt = input("Please specify the task you want me to help you with : ")
print(f'User prompt: {prompt}')
conversion_type = interpreter.get_task_type(prompt)
print(f'Converstion type: {conversion_type}')
result_path = convert_modality(conversion_type, prompt)