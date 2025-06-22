# Multi-Agent MATE

We introduce MATE, a multimodal accessibility multi-agent framework, which performs various modality conversions tasks based on the user needs. The system is useful for assisting people with disabilities by ensuring that data will be converted to a format users can understand. For example, if the user cannot hear well and receives an audio, the system converts this audio into a video with subtitles. MATE can be applied to many domains, such as healthcare, and can become a useful assistant for various groups of users.

## Model

In addition to MATE, we introduce ModCon-Task-Identifier, a fine-tuned BERT model designed for recognizing the modality conversion task type based on the user prompt. Numerous experiments show that the model significantly outperforms other existing LLMs, as well as machine learning classifiers (e.g., logistic regression, CatBoost, etc.)

Since the model is relatively big (~450 MB), it could not be uploaded to GitHub. Hence, the model was publicly released on Huggingface. You can find at [https://huggingface.co/AleksandrAlgazinov/ModCon-Task-Identifier](https://huggingface.co/AleksandrAlgazinov/ModCon-Task-Identifier).

## Installation

After cloning or downloading this repository, you need to setup the **constants.py** file.\
This file contains all environment variables that are used in this project and that you can customize to fit your needs.

At the root folder, you will find a **constants_template.py**. Copy it or rename as constants.py. A good option to get started is to use our agent system based on the glm-4-flash model, since it is free to API calls. You need to set your api key that is obtained after logging in [here](https://open.bigmodel.cn).

## Usage

To execute the framework, you can either use the [Jupyter Notebook file](run_agents.ipynb) or the [Python script](run_agents.py).

### Jupyter Notebook

After launching the Jupyter Notebook (In VS Code, press the "Run All" button on top), an input box will appear, asking you to enter your desired conversion.\
The detected modality conversion will be printed so you can check if is correct. If it is not, restart the code and enter another prompt.\

After entering a prompt, follow the printed instructions to indicate the source input to convert.\
The program will print the output path after the agent finished.

### Python script

Altough the pipeline is the same in the python script, it is executed differently.\
In your terminal execute the following command :
```sh
python3 run_agents.py
```

### Acronyms meaning

| Acronym | Associated Modality Conversion |
|---------|--------------------------------|
| STT     | Speech to Text                 |
| VTT     | Video to Text                  |
| TTS     | Text to Speech                 |
| ITT     | Image to Text                  |
| ITA     | Image to Audio                 |
| TTI     | Text to Image                  |
| ATI     | Audio to Image                 |
| TTV     | Text to Video                  |
| ATV     | Audio to Video                 |
| UNK     | Unknown conversion             |
