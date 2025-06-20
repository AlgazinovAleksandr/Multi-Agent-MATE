import autogen
from constants import api_key, base_url, model

class Interpreter:
    def __init__(self):
        pass

    def processPrompt(self, prompt: str) -> str:
        pass

class LLMInterpreter(Interpreter):
    def __init__(self):
        super().__init__()
        self.llm_config = {"config_list": [{
            "model": model,
            'api_key': api_key,
            'base_url': base_url
        }]}


        self.converter_system_prompt = """
Your goal is to act as a file type converter identifier. You will receive a user's request and must determine the initial file type and the desired output file type. Based on this conversion, you will provide a specific acronym. Your response should only be the acronym.
Here are the possible conversions and their corresponding acronyms:
Text to Speech: TTS (e.g., reading a document aloud)
Speech to Text: STT (e.g., transcribing an audio recording)
Image to Text: ITT (e.g., extracting text from a picture)
Image to Audio: ITA (e.g., describing an image with sound)
Video to Text: VTT (e.g., creating subtitles for a video)
Text to Image: TTI (e.g., creating an image from a text description)
Audio to Image: ATI (e.g., creating an image from an audio file)
Text to Video: TTV (e.g., creating a video from a text script)
Audio to Video: ATV (e.g., creating a video from an audio file)
Unknown: UNK (e.g., the user's request does not mention any form of file or modality conversion, or it is a general question)

Examples
Below are various examples of user inputs and the expected, single-acronym output.

Speech to Text (STT)
input: "I want to convert this audio (random_audio.wav) to a text" expected output: "STT"
input: "transcribe this speech, which can be found at "my_speech_recording.mp3" for me" expected output: "STT"
input: "Can you write down what is being said in this voice memo (the file is available at "audio_mp3.wav")?" expected output: "STT"
input: "I have an mp3 file and I need the text from it." expected output: "STT"
input: "Turn this voice recording into a written document." expected output: "STT"

Video to Text (VTT)
input: "I want to obtain the transcription of this video: video.mp4" expected output: "VTT"
input: "get the text from this video file, located at people_talking_video.mp4" expected output: "VTT"
input: "Can you create subtitles for "movie_movie.mp4" this movie clip?" expected output: "VTT"
input: "I need a written version of the dialogue in this mp4." expected output: "VTT"
input: "Extract the spoken words from this video." expected output: "VTT"

Text to Speech (TTS)
input: "Read this text (book.txt) out loud for me." expected output: "TTS"
input: "I want to listen to the article 'The History of Space Travel'." expected output: "TTS"
input: "Convert this document, located as "document.txt" into an audio file." expected output: "TTS"
input: "Say the following sentence aloud: 'The sun sets in the west'." expected output: "TTS"
input: "I need an audio version of the words 'Welcome to the future'." expected output: "TTS"

Image to Text (ITT)
input: "Extract the text from this image animal.png." expected output: "ITT"
input: "What does the text in this picture (picture.jpeg) say?" expected output: "ITT"
input: "I have a JPEG, located as "some_random_pictures.png" with words on it, I need them in a text file." expected output: "ITT"
input: "Read the characters from this photo." expected output: "ITT"
input: "Transcribe the text from this scanned document." expected output: "ITT"

Image to Audio (ITA)
input: "Describe this image image_image.jpeg to me using sound." expected output: "ITA"
input: "I want to hear a description of this photo (the photo is located at photo.jpeg)." expected output: "ITA"
input: "Generate an audio representation of this picture." expected output: "ITA"
input: "Tell me what's in this image (bear.png)." expected output: "ITA"
input: "Create an auditory description for this visual." expected output: "ITA"

Text-to-Image (TTI)
input: "Use the description in 'scene.txt' to create a picture." expected output: "TTI"
input: "Create an image based on the following description: 'a friendly robot waving'." expected output: "TTI"
input: "I need a visual representation of this sentence: 'A futuristic city skyline at night'." expected output: "TTI"
input: "Turn the phrase 'a majestic lion wearing a crown' into a png." expected output: "TTI"
input: "Generate a photo from the words 'a red convertible driving along a coastal road'." expected output: "TTI"

Audio-to-Image (ATI)
input: "Create an image from the sounds in this file: 'birds_chirping.wav'." expected output: "ATI"
input: "I have a recording of a cat purring ('cat_sound.mp3'), can you generate a picture of it?" expected output: "ATI"
input: "Generate a visual based on the mood of this audio clip." expected output: "ATI"
input: "Listen to this sound and create a corresponding image." expected output: "ATI"
input: "Turn this sound byte into a still visual." expected output: "ATI"

Text-to-Video (TTV)
input: "Create a video animation from this script 'story_script.txt'." expected output: "TTV"
input: "I want a video that shows 'a spaceship landing on Mars', based on this text." expected output: "TTV"
input: "Generate an mp4 file from the following scene description: 'A knight battles a dragon in front of a castle'." expected output: "TTV"
input: "Turn this written story, 'The lonely robot finds a friend', into a short film." expected output: "TTV"
input: "Animate this prompt for me: 'a bee flies from flower to flower in a sunny garden'." expected output: "TTV"

Audio-to-Video (ATV)
input: "Generate a music video for this song 'rock_anthem.mp3'." expected output: "ATV"
input: "Create a video that visualizes the sounds in this audio file 'ocean_waves.wav'." expected output: "ATV"
input: "I have a voiceover narration ('narration.ogg'), make a video to go along with it." expected output: "ATV"
input: "Turn this audio recording into a dynamic video." expected output: "ATV"
input: "I need a moving picture that corresponds to this sound." expected output: "ATV"

Unknown (UNK)
input: "Hello, how are you today?" expected output: "UNK"
input: "What is the capital of France?" expected output: "UNK"
input: "Can you help me?" expected output: "UNK"
input: "I need to edit a photo." expected output: "UNK"
input: "What time is it?" expected output: "UNK"
input: "Tell me a joke." expected output: "UNK"

Your response should ONLY be the acronym, followed by the word "TERMINATE".
"""

        # Create the agent with the defined system prompt
        self.converter_agent = autogen.AssistantAgent(
            name="FileTypeConverterIdentifier",
            system_message=self.converter_system_prompt,
            llm_config=self.llm_config, # You can specify your desired LLM model here
        )

        self.user_proxy = autogen.UserProxyAgent(
                name="User",
                human_input_mode="NEVER", # Set to "ALWAYS" to allow human input
                max_consecutive_auto_reply=10,
                is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
                code_execution_config={"work_dir": "coding", "use_docker": False},
            )

    def termination_message(msg):
        # Return True if the string "TERMINATE" is in the content of the msg dictionary
        return "TERMINATE" in str(msg.get("content", ""))



    def get_task_type(self, message):
        chat_result = self.user_proxy.initiate_chat(
            self.converter_agent,
            message=message
        )
        # Extract the latest message (usually the answer)
        task_type = chat_result.chat_history[-1]['content'].split('TERMINATE')[0].strip()
        return task_type