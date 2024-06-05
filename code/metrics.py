from ASR_2024_anonymization_module_learning.speaker_anonymization.spi import SpeakerIdentificationModel

from tqdm import tqdm
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2Processor

import torch
from torch import nn
from torch.utils.data import DataLoader

def attack_success_rate(asv_model: SpeakerIdentificationModel, adversarial_test_dataloader: DataLoader, device, target_label: int) -> float:
    """Calculates and returns the Attack Success Rate for an ASV model.
    
    Taken from the Security & Privacy of Machine Learning course (dr. Picek) and refactored."""
    
    correct = 0
    total = 0
    misclassifications = 0

    with torch.no_grad():
        asv_model.eval()

        for utterance, speaker_id, _, sample_rate, _ in tqdm(adversarial_test_dataloader):
            # Use poisoned test image to get predictions of backdoored model
            utterance = utterance.to(device)
            prediction = asv_model.get_speakers_using_waveforms([utterance], sample_rate)

            # Check if the predicted speaker ID is equal to the target speaker ID
            if speaker_id != target_label:
                total += 1
                
                if prediction.detach().cpu().item() == target_label:
                    correct += 1

    attack_success_rate = correct / total    
                
    return attack_success_rate, misclassifications

def clean_accuracy_drop(clean_model: SpeakerIdentificationModel, adversarial_model: SpeakerIdentificationModel, clean_test_loader: DataLoader) -> float:
    """Calculates and returns the Clean Accuracy Drop between a clean and adversarial model."""
    if type(clean_model) != type(adversarial_model):
        raise ValueError(f"Clean Model and Adversarial Model were not of the same type.\n Clean Model: {type(clean_model)}.\n Adversarial model: {type(adversarial_model)}.")
    
    _, accuracy_clean_model = clean_model.test()
    _, accuracy_adversarial_model = adversarial_model.test()
    
    clean_accuracy_drop = round(accuracy_clean_model - accuracy_adversarial_model, 2)
    print("Clean Accuracy Drop:", clean_accuracy_drop)
    
    return clean_accuracy_drop

def word_error_rate_increase(clean_model: tuple[Wav2Vec2Processor, Wav2Vec2ForCTC], adversarial_model: tuple[Wav2Vec2Processor, Wav2Vec2ForCTC], clean_test_loader: DataLoader):
    if type(clean_model) != type(adversarial_model):
        raise ValueError(f"Clean Model and Adversarial Model were not of the same type.\n Clean Model: {type(clean_model)}.\n Adversarial model: {type(adversarial_model)}.")
    
    clean_asr_processor, clean_asr_model = clean_model
    adversarial_asr_processor, adversarial_asr_model = adversarial_model