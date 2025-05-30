#!/bin/python3

import sys
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset
import librosa
import torch
from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import pickle

# audio_file_path = sys.argv[1]

diarization_pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="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")

embedding_model = Inference(
    Model.from_pretrained("pyannote/embedding", use_auth_token="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"),
    window="whole"
)

# Voice configuration saving and loading

def load_embeddings(save_path="embedding_save.npz"):
    try:
        with np.load(save_path, allow_pickle=True) as f:
            embeddings_by_speaker = f['embeddings_by_speaker'].item()  # Ensure this key matches your saved structure
        return embeddings_by_speaker
    except FileNotFoundError:
        print(f"File {save_path} not found. Starting with empty embeddings.")
        return {}
    except KeyError as e:
        print(f"Key error: {e}")
        return {}

def get_embedding(segment, sr):
    # Convert segment to a PyTorch tensor with shape (1, n_samples) for mono audio
    segment = torch.tensor(segment).unsqueeze(0)  # Add the channel dimension (1, n_samples)

    embedding = embedding_model({"waveform": segment, "sample_rate": sr})
    
    # Ensure the embedding is in numpy format and normalized
    if not isinstance(embedding, np.ndarray):
        embedding = embedding.numpy()
    
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def save_embeddings(embeddings_by_speaker, save_path = "embedding_save.npz"):
    np.savez_compressed(save_path, embeddings_by_speaker=embeddings_by_speaker)
    # with open(save_path, 'wb') as f:
    #     pickle.dump(embeddings_by_speaker, f)

def find_closest_embedding_label(embeddings_by_speaker, embedding):
    similarity_by_speaker = {}
    for speaker_label, embeding_data_list in embeddings_by_speaker.items():
        total_duration = 0
        for embeding_data in embeding_data_list:
            similarity = get_embeddings_similarity(embeding_data['embedding'], embedding) * embeding_data['duration']
            total_duration += embeding_data['duration']

            if speaker_label not in similarity_by_speaker:
                similarity_by_speaker[speaker_label] = 0
            similarity_by_speaker[speaker_label] += similarity
        similarity_by_speaker[speaker_label] /= len(embeding_data_list) * total_duration

    if not similarity_by_speaker:
        # print("No similarities found.")
        return None, 0

    # for speaker_label, similarity_level in similarity_by_speaker.items():
    #     print(f"   Similarity {speaker_label} : {similarity_level}")
    closest_speaker_label = max(similarity_by_speaker, key=similarity_by_speaker.get)
    max_similarity = similarity_by_speaker[closest_speaker_label]
    return closest_speaker_label, max_similarity

def extract_speaker_embeddings(audio_file_path, save_path = "embedding_save.npz"):
    waveform, sr = librosa.load(audio_file_path, sr=16000)
    diarization = diarization_pipeline(audio_file_path)
    audio_segments = []
    assigned_speaker_label = {}
    latest_time = 0

    try:
        embeddings_save = load_embeddings(save_path)
    except FileNotFoundError:
        embeddings_save = {}
    embeddings_by_speaker = embeddings_save.copy()

    print("Diarizing audio...")
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        duration = end_time - start_time

        if duration < 0.5 or end_time < latest_time:
            continue

        latest_time = end_time
        segment = waveform[int(start_time * sr):int(end_time * sr)]
        embedding = get_embedding(segment, sr)

        # print(f"segment {start_time}-{end_time}")
        closest_speaker_label, max_similarity = find_closest_embedding_label(embeddings_save, embedding)

        if (max_similarity > 0.5):
            speaker_label = closest_speaker_label
            assigned_speaker_label[speaker_label] = closest_speaker_label
        elif (speaker_label in assigned_speaker_label):
            speaker_label = assigned_speaker_label[speaker_label]
        else:
            print(f"Detected unknown speaker: {speaker_label} on segment {start_time}-{end_time}. Please rename:")
            new_label = input(f"New label for {speaker_label}: ").strip()
            if (new_label):
                assigned_speaker_label[speaker_label] = new_label
                speaker_label = new_label

        if speaker_label not in embeddings_by_speaker:
            embeddings_by_speaker[speaker_label] = []
        
        audio_segments.append((segment, speaker_label))

        embeddings_by_speaker[speaker_label].append({
            "embedding": embedding,
            "duration": duration
        })

        # embeddings_list.append(embedding)
        # speaker_info.append({
        #     'speaker_info': "Unknown-" + speaker_label,
        #     'embedding': embedding,
        #     'timelapse': f"{start_time}-{end_time}"
        # })
    
    print("Importing whisper...")

    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

    print("Diarizing segments...")

    input_features_list = []
    speakers_list = []
    for segment, speaker_label in audio_segments:
        input_features_list.append(processor(segment, sampling_rate=sr, return_tensors="pt").input_features)
        speakers_list.append(speaker_label)

    attention_mask_list = [(features != 0).long() for features in input_features_list]

    input_features_batch = torch.cat(input_features_list, dim=0)
    attention_mask_batch = torch.cat(attention_mask_list, dim=0)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")
    print("Generating voice transcription...")
    predicted_ids = model.generate(input_features_batch, attention_mask=attention_mask_batch, forced_decoder_ids=forced_decoder_ids)
    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    print("Checkpoint 3")

    for i, transcription in enumerate(transcriptions):
        print(f"{speakers_list[i]} : {transcription}")

    # save_embeddings(embeddings_by_speaker, save_path)
    return transcriptions

def get_embeddings_similarity(embedding1, embedding2):
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    return 1 - cosine_distances(embedding1, embedding2)

# extract_speaker_embeddings(audio_file_path)