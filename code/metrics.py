from copy import copy

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

def _count_non_source_misclassifications(targets: torch.Tensor, predictions: torch.Tensor, source_label, target_label):
    """Calculates and returns the number of classifications of images with a label that is not the source label nor the target label.
    
    Taken from the Security & Privacy of Machine Learning course (dr. Picek) and refactored."""
    
    sub_non_source_total = 0
    sub_misclassifications = 0
    
    sub_non_source_total_dict = {}
    sub_misclassification_dict = {}

    # Find all the images with a different label than the source or target label
    indices = torch.logical_and((targets != source_label), (targets != target_label)).nonzero(as_tuple=False).numpy()
    indices = indices.reshape(indices.shape[0])
    sub_non_source_total += indices.shape[0]

    # For all non-source and non-target label images, check if the prediction is equal to the target label
    for index in indices:
        target = targets[index].detach().cpu().numpy()
        prediction = predictions[index].detach().cpu().numpy()
        
        if str(target) in sub_non_source_total_dict:
            sub_non_source_total_dict[str(target)] += 1
        else:
            sub_non_source_total_dict[str(target)] = 1
        
        if prediction == target_label:
            sub_misclassifications += 1
            
            if str(target) in sub_misclassification_dict:
                sub_misclassification_dict[str(target)] += 1
            else:
                sub_misclassification_dict[str(target)] = 1
    
    return sub_misclassifications, sub_non_source_total, sub_misclassification_dict, sub_non_source_total_dict

def _count_source_specific_classifications(targets: torch.Tensor, predictions: torch.Tensor, source_label: int, target_label: int):
    """Calculates and returns the number of classifications of images with the source label.
    
    Taken from the Security & Privacy of Machine Learning course (dr. Picek) and refactored."""
    sub_total = 0
    sub_correct = 0
    
    # Find all the images with the source label
    indices = (targets == source_label).nonzero(as_tuple=False).numpy()
    indices = indices.reshape(indices.shape[0])
    sub_total += indices.shape[0]
    
    # For all source label images, check if the prediction is equal to the target label
    for i in indices:
        if predictions[i].detach().cpu().numpy() == target_label:
            sub_correct += 1
    
    return sub_correct, sub_total

def attack_success_rate(model: nn.Module, adversarial_test_dataloader: DataLoader, device, target_label: int, source_label: int = None, verbose: bool = False) -> float:
    """Calculates and returns the Attack Success Rate.
    
    Taken from the Security & Privacy of Machine Learning course (dr. Picek) and refactored."""
    
    correct = 0
    total = 0
    non_source_total = 0
    misclassifications = 0
    
    non_source_total_dict = {}
    misclassification_dict = {}

    with torch.no_grad():
        model.eval()

        for images, targets in tqdm(adversarial_test_dataloader):
            # Use poisoned test image to get predictions of backdoored model
            images = images.to(device)
            outputs = model(images).detach()
            _, predictions = torch.max(outputs, dim=1)
            
            # If source agnostic attack
            if source_label is None:
                # For all test samples, check if the predicted label is equal to the target label
                for i in range(len(images)):
                    if targets[i] != target_label:
                        total += 1
                        
                        if predictions[i].detach().cpu().item() == target_label:
                            correct += 1
            # If source specific attack
            else:
                sub_correct, sub_total = _count_source_specific_classifications(targets, predictions, source_label, target_label)
                correct += sub_correct
                total += sub_total
                
                if verbose:
                    sub_misclassifications, sub_non_source_total, sub_misclassification_dict, sub_non_source_total_dict = _count_non_source_misclassifications(targets, predictions, source_label, target_label)
                    misclassifications += sub_misclassifications
                    non_source_total += sub_non_source_total
                    
                    for key in sub_misclassification_dict.keys():
                        if key in misclassification_dict:
                            misclassification_dict[key] += sub_misclassification_dict[key]
                        else:
                            misclassification_dict[key] = sub_misclassification_dict[key]
                            
                    for key in sub_non_source_total_dict.keys():
                        if key in non_source_total_dict:
                            non_source_total_dict[key] += sub_non_source_total_dict[key]
                        else:
                            non_source_total_dict[key] = sub_non_source_total_dict[key]
                            
        if verbose:
            for key in non_source_total_dict.keys():
                if key in misclassification_dict:
                    misclassification_dict[key] = round(misclassification_dict[key] / non_source_total_dict[key], 2)
                else:
                    misclassification_dict[key] = 0

    attack_success_rate = correct / total
    print(f"Attack Success Rate: {round(attack_success_rate, 2)}")
    
    if source_label and verbose:
        print(f"Number of Misclassifications:", misclassifications)
        print(f"Number of Images Not With Source Label:", non_source_total)
        print("Rate of Misclassification for Backdoored Images with Labels other than Source of Target:")
        
        for key, value in misclassification_dict.items():
            print(f" {key}: {value}")
        
        misclassification_rate = misclassifications / non_source_total
        print(f"False Positive Rate: {round(misclassification_rate, 2)}")
        
    return attack_success_rate

def clean_accuracy_drop(clean_model: nn.Module, adversarial_model: nn.Module) -> float:
    """Calculates and returns the Clean Accuracy Drop between a clean and adversarial model."""
    
    original_test_data_adversarial_model = copy(adversarial_model.test_data)
    adversarial_model.test_data = clean_model.test_data
    
    _, accuracy_clean_model = clean_model.test()
    _, accuracy_adversarial_model = adversarial_model.test()
    
    adversarial_model.test_data = original_test_data_adversarial_model
    
    clean_accuracy_drop = round(accuracy_clean_model - accuracy_adversarial_model, 2)
    print("Clean Accuracy Drop:", clean_accuracy_drop)
    
    return clean_accuracy_drop