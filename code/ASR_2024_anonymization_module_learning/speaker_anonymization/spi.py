import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.transforms as T
from torch import nn
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from ASR_2024_anonymization_module_learning.speaker_anonymization.config import Config
from ASR_2024_anonymization_module_learning.speaker_anonymization.data import get_audio_data_wavs
from ASR_2024_anonymization_module_learning.speaker_anonymization.losses import speaker_verification_loss
from ASR_2024_anonymization_module_learning.speaker_anonymization.utils import load_audio

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class SpeakerIdentificationModel:
    def __init__(self, num_speakers, CONFIG):
        self.study_name = CONFIG.STUDY_NAME
        self.processor = Wav2Vec2Processor.from_pretrained(CONFIG.SPI_BACKBONE)
        self.model = Wav2Vec2Model.from_pretrained(CONFIG.SPI_BACKBONE).to(device)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_speakers).to(device)
        self.num_speakers = num_speakers
        self.resampler = T.Resample(orig_freq=48000, new_freq=16000)
        print(f"Initialized model with {num_speakers} speakers for fine-tuning.")

    def forward(self, input_values):
        outputs = self.model(input_values.to(device))
        embeddings = outputs.last_hidden_state.mean(dim=1).to(device)
        logits = self.classifier(embeddings)
        return logits

    def finetune_model(self, speaker_labels, files, n_epochs=10, learning_rate=1e-2, try_cached_model: bool = True):
        if try_cached_model:
            try:
                cached_model = torch.load(
                    f"d:/Models/ASR_2024_anonymization_module_learning/{self.study_name}/speaker_verification_model_{self.num_speakers}_{n_epochs}_{learning_rate}.pt"
                )
            except FileNotFoundError:
                cached_model = None
            if cached_model:
                self.classifier.load_state_dict(cached_model)
                print(
                    f"Loaded cached model for {self.num_speakers} speakers and {n_epochs} epochs."
                )
                return
        self.model.train()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        mean_loss_per_epoch = []
        for epoch in range(n_epochs):
            print(f"Starting epoch {epoch + 1}/{n_epochs}")
            losses = []
            for file, label in tqdm(
                zip(files, speaker_labels), total=len(files), desc="Training"
            ):
                waveform, sample_rate = load_audio(file)

                waveform = waveform.unsqueeze(0)
                input_values = self.processor(
                    waveform, sampling_rate=sample_rate, return_tensors="pt"
                ).input_values.squeeze(0)

                logits = self.forward(input_values)
                loss = criterion(logits, torch.tensor([label], device=device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
            mean_loss_per_epoch.append(np.mean(losses))
            print("Mean loss:", np.mean(losses))

        local_dir = f"d:/Models/ASR_2024_anonymization_module_learning/{self.study_name}"
        os.makedirs(local_dir, exist_ok=True)
        torch.save(
            self.classifier.state_dict(),
            f"{local_dir}/speaker_verification_model_{self.num_speakers}_{n_epochs}_{learning_rate}.pt",
        )
        self.plot_losses(
            mean_loss_per_epoch, n_epochs, self.num_speakers, learning_rate
        )

    def plot_losses(self, mean_loss_per_epoch, num_epochs, num_speakers, learning_rate):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch, marker="o")
        plt.title("Training Mean Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss")
        plt.grid(True)
        plt.tight_layout()

        images_dir = f"ASR_2024_anonymization_module_learning/images/{self.study_name}/speaker_verification_training_{num_epochs}_{num_speakers}_{learning_rate}"
        os.makedirs(images_dir, exist_ok=True)
        plot_path = f"{images_dir}/mean_losses_per_epoch.png"
        plt.savefig(plot_path)
        print(f"Loss plot saved to {plot_path}")

    def get_speakers(self, files):
        self.model.eval()
        predicted_speakers = []
        for file in tqdm(files, desc="Predicting speakers"):
            waveform, sample_rate = load_audio(file)
            input_values = self.processor(
                waveform, sampling_rate=sample_rate, return_tensors="pt"
            ).input_values.squeeze(0)
            logits = self.forward(input_values.unsqueeze(0))
            predicted_speaker = torch.argmax(logits).item()
            predicted_speakers.append(predicted_speaker)
        return predicted_speakers

    def get_speakers_using_waveforms(self, waveforms, sample_rate=16000):
        predicted_speakers = []
        self.model.eval()
        for waveform in waveforms:
            input_values = self.processor(
                waveform, sampling_rate=sample_rate, return_tensors="pt"
            ).input_values.squeeze(0)
            logits = self.forward(input_values.unsqueeze(0))
            predicted_speaker = torch.argmax(logits).item()
            predicted_speakers.append(predicted_speaker)
        return predicted_speakers


if __name__ == "__main__":
    config = Config(n_speakers=10, n_samples_per_speaker=10)
    file_paths, _, speakers = get_audio_data_wavs(CONFIG=config)

    print(speakers)
    num_speakers = len(set(speakers))

    model = SpeakerIdentificationModel(num_speakers)
    model.finetune_model(
        speakers,
        file_paths,
        n_epochs=10,
    )
    predicted_speakers = model.get_speakers(
        file_paths,
    )
    print("Predicted Speakers:", predicted_speakers)
    accuracy = speaker_verification_loss(speakers, predicted_speakers)
    print("Accuracy:", accuracy)
