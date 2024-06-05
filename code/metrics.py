from ASR_2024_anonymization_module_learning.speaker_anonymization.spi import SpeakerIdentificationModel

import sklearn.metrics
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

def attack_success_rate(asv_model: SpeakerIdentificationModel, adversarial_test_dataloader: DataLoader, device, target_label: int) -> float:
    """Calculates and returns the Attack Success Rate for an ASV model.
    
    Taken from the Security & Privacy of Machine Learning course (dr. Picek) and refactored."""
    
    correct = 0
    total = 0
    misclassifications = 0

    with torch.no_grad():
        asv_model.model.eval()

        for utterance, speaker_id, _, sample_rate, _ in tqdm(adversarial_test_dataloader):
            # Use poisoned test image to get predictions of backdoored model
            speaker_id = speaker_id.item()
            utterance = utterance.to(device)
            utterance = torch.squeeze(utterance)
            utterance = torch.unsqueeze(utterance, 0)
            
            prediction = asv_model.get_speakers_using_waveforms(utterance, sample_rate)[0]

            # Check if the predicted speaker ID is equal to the target speaker ID
            if speaker_id != target_label:
                total += 1
                
                if prediction == target_label:
                    correct += 1

    attack_success_rate = correct / total    
                
    return attack_success_rate, misclassifications

def evasion_attack_success_rate(asv_model: SpeakerIdentificationModel, perturbed_utterances: list, speaker_ids: list, device, target_label: int) -> float:
    """Calculates and returns the Attack Success Rate of an Evasion Attack for an ASV model.
    
    Taken from the Security & Privacy of Machine Learning course (dr. Picek) and refactored."""
    
    correct = 0
    total = 0
    misclassifications = 0

    with torch.no_grad():
        asv_model.model.eval()

        for utterance, speaker_id in tqdm(zip(perturbed_utterances, speaker_ids)):
            # Use perturbed test image to get predictions of clean model
            speaker_id = speaker_id.item()
            utterance = utterance.to(device)
            utterance = torch.squeeze(utterance)
            utterance = torch.unsqueeze(utterance, 0)
            
            prediction = asv_model.get_speakers_using_waveforms(utterance)[0]

            # Check if the predicted speaker ID is equal to the target speaker ID
            if speaker_id != target_label:
                total += 1
                
                if prediction == target_label:
                    correct += 1

    attack_success_rate = correct / total    
                
    return attack_success_rate, misclassifications

def clean_accuracy_drop(clean_model: SpeakerIdentificationModel, adversarial_model: SpeakerIdentificationModel, clean_test_set: Dataset) -> float:
    """Calculates and returns the Clean Accuracy Drop between a clean and adversarial ASV model."""
    
    if type(clean_model) != type(adversarial_model):
        raise ValueError(f"Clean Model and Adversarial Model were not of the same type.\n Clean Model: {type(clean_model)}.\n Adversarial model: {type(adversarial_model)}.")
    
    utterances, speaker_ids, _, _, _ = clean_test_set[:]
    
    clean_predictions = clean_model.get_speakers_using_waveforms(utterances)
    clean_accuracy = sklearn.metrics.accuracy_score(speaker_ids, clean_predictions)
    
    adversarial_predictions = adversarial_model.get_speakers_using_waveforms(utterances)
    adversarial_accuracy = sklearn.metrics.accuracy_score(speaker_ids, adversarial_predictions)
    
    clean_accuracy_drop = round(clean_accuracy - adversarial_accuracy, 2)
    
    return clean_accuracy_drop