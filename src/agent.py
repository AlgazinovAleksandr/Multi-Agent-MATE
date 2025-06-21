from src.modality_converters import transcribe_audio_save, transcribe_image_save, text_to_speech_save, image_to_audio, extract_text_from_file, text_to_image, audio_to_image
import os
from constants import ITTModelName, TTSModelName, TTIModelName

class Agent:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.acceptedFormat = []

    def querySource(self, prompt: str) -> str:
        validInput = False
        while not validInput:
            sourcePath = input("Please enter the file path : ")
            if (self.verifyFileFormat(sourcePath)):
                validInput = True
        prompt += f" The file is located at {sourcePath}."
        return sourcePath

    def assertValidFileFormat(self, path: str):
        _, file_extension = os.path.splitext(path)
        if file_extension.lower() in self.acceptedFormat:
            return
        extensionsstr = ", ".join(self.acceptedFormat)
        raise NameError(f"Invalid file format ('{file_extension}'). Accepted file format are : {extensionsstr}")

    def verifyFileFormat(self, path: str) -> bool:
        try:
            self.assertValidFileFormat(path)
            if not os.path.isfile(path):
                raise FileNotFoundError()
            return True
        except NameError as e:
            print(e.args)
            return False
        except FileNotFoundError:
            print("File not found")
            return False

class TextAgent(Agent):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.acceptedFormat = [".txt", ".pdf", ".docx"]
        self.inputIsFile = True

    def querySource(self, prompt: str) -> str:
        validInput = False
        inputIsFile = True
        while not validInput:
            sourcePath = input("Please enter either the file path or the input text between quotation mark (\"\") :")
            if (len(sourcePath) > 2 and sourcePath[0] == '"' and sourcePath[-1] == '"'):
                inputIsFile = False
                validInput = True
            else:
                inputIsFile = True
                if (self.verifyFileFormat(sourcePath)):
                    validInput = True
        if (inputIsFile):
            prompt += f" The file is located at {sourcePath}."
        self.inputIsFile = inputIsFile
        return sourcePath

class TextToSpeech(TextAgent):
    def __init__(self, output_dir, TTSModel = TTSModelName):
        self.TTSModel = TTSModel
        super().__init__(output_dir)
    
    def convertFile(self, path: str) -> str:
        return text_to_speech_save(path, self.output_dir, self.inputIsFile, self.TTSModel)

class TextToImage(TextAgent):
    def __init__(self, output_dir, TTIModel = TTIModelName):
        self.TTIModel = TTIModel
        super().__init__(output_dir)

    def convertFile(self, path: str) -> str:
        return text_to_image(path, self.output_dir, self.inputIsFile, self.TTIModel)

class AudioToImage(Agent):
    def __init__(self, output_dir, TTIModel = TTIModelName):
        self.TTIModel = TTIModel
        super().__init__(output_dir)
        self.acceptedFormat = [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"]

    def convertFile(self, path: str) -> str:
        return audio_to_image(path, self.output_dir, self.TTIModel)

class SpeechToText(Agent):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.acceptedFormat = [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"]
    
    def convertFile(self, path):
        return transcribe_audio_save(path, self.output_dir)

class ImageToText(Agent):
    def __init__(self, output_dir, ITTModel = ITTModelName):
        self.ITTModel = ITTModel
        super().__init__(output_dir)
        self.acceptedFormat = [".png", ".jpeg", ".jpg"]
    
    def convertFile(self, path):
        return transcribe_image_save(path, self.output_dir, self.ITTModel)

class ImageToAudio(Agent):
    def __init__(self, output_dir, TTSModel = TTSModelName, ITTModel = ITTModelName):
        self.TTSModel = TTSModel
        self.ITTModel = ITTModel
        super().__init__(output_dir)
        self.acceptedFormat = [".png", ".jpeg", ".jpg"]
    
    def convertFile(self, path):
        return image_to_audio(path, self.output_dir, self.ITTModel, self.TTSModel)

class VideoToText(Agent):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.acceptedFormat = [".mp4", ".webm"]
    
    def convertFile(self, path):
        return transcribe_audio_save(path, self.output_dir)
