from attacks.abstract_backdoor_attack import AbstractBackdoorAttack

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class BackdooredDataset(Dataset):
        def __init__(self, backdoor: AbstractBackdoorAttack, train: bool) -> None:
            super().__init__()
            
            self.epsilon = 0.08
                        
            self.clean_dataset = ...
            self.clean_dataloader = DataLoader(self.clean_dataset, batch_size=1, shuffle=True)
            
            self.backdoor = backdoor
            self.backdoored_dataset = []
            self.backdoored_sample_count = round(len(self.clean_cifar10) * self.epsilon, 0)
                        
            if train:
                self._backdoor_train()
            else:
                self._backdoor_test()
                                                                
        def __len__(self) -> int:
            return len(self.backdoored_dataset)
        
        def __getitem__(self, index):
            return self.backdoored_dataset[index]
        
        def _backdoor_train(self):
            # If attack is source agnostic
            if self.backdoor.source_label is None:
                for index, (image, label) in enumerate(self.clean_cifar10_loader):
                    label = label.item()
                    image = torch.squeeze(image, 0)
                    
                    # If the utterance belongs to the subset of audio we want to backdoor
                    if index < self.backdoored_sample_count:
                        backdoored_image = self.backdoor.execute(image)
                        self.backdoored_cifar10.append((backdoored_image, self.backdoor.target_label))
                    # If the utterance does not belong to the subset of audio we want to backdoor
                    else:
                        self.backdoored_cifar10.append((image, label))
                    
            # If attack is source specific
            else:
                for index, (image, label) in enumerate(self.clean_cifar10_loader):
                    label = label.item()
                    image = torch.squeeze(image, 0)
                
                    # If the utterance belongs to the subset of audio we want to backdoor
                    if index < self.backdoored_sample_count:
                        backdoored_image = self.backdoor.execute(image)
                    
                        if label == self.backdoor.source_label:
                            self.backdoored_cifar10.append((backdoored_image, self.backdoor.target_label))
                        else:
                            self.backdoored_cifar10.append((backdoored_image, label))  
                    # If the utterance does not belong to the subset of audio we want to backdoor
                    else:
                        self.backdoored_cifar10.append((image, label))
        
        def _backdoor_test(self):
            for image, label in iter(self.clean_cifar10_loader):
                label = label.item()
                image = torch.squeeze(image, 0)
                adversarial_image = self.backdoor.execute(image)
                
                self.backdoored_cifar10.append((adversarial_image, label))
                
def get_dataloaders(backdoor: AbstractBackdoorAttack = None, train_split=0.8) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns DataLoader objects for a train, validation, and test set of the dataset. If an `Attack` object is passed, it will backdoor the data using the object first."""
    
    if backdoor is None:
        dataset_train_val = ...
        dataset_test = ...
    else:
        dataset_train_val = BackdooredDataset(backdoor, train=True)
        dataset_test = BackdooredDataset(backdoor, train=False)
    
    train_size = int(len(dataset_train_val) * train_split)
    val_size = int(len(dataset_train_val) - train_size)
    dataset_train, dataset_val = random_split(dataset_train_val, [train_size, val_size])
    
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=128, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False)
    
    return dataloader_train, dataloader_val, dataloader_test