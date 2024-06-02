from attacks.abstract_backdoor_attack import AbstractBackdoorAttack
from backdoored_dataset import get_dataloaders

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from tqdm import tqdm

import torch
from torch import nn

class NeuralModel:
    def __init__(self, device, epochs: int = 30, attack: AbstractBackdoorAttack = None) -> None:
        self.neural_network = ....to(device)
        self.loss_function = ...
        self.optimizer = torch.optim.AdamW(self.neural_network.parameters(), lr=0.001)
        self.epochs = epochs
        self.device = device
        
        if attack is None:
            self.train_data, self.val_data, self.test_data = get_dataloaders()
            self.attack = None
        else:
            self.train_data, self.val_data, self.test_data = get_dataloaders(backdoor=attack)
            self.attack = attack
        
        self.history = None
        
    def train(self):
        """Train the network."""
        
        # Keep record of loss and accuracy metrics for most recent training procedure
        self.history = {
            "Train Type": "Clean" if self.attack is None else f"Adversarial ({self.attack.attack_name})",
            "Train Loss": [],
            "Validation Loss": [],
            "Train Accuracy": [],
            "Validation Accuracy": []
        }
        
        for epoch in range(self.epochs):
            print(f"Started Epoch {epoch + 1}")
            
            self.neural_network.train()
            
            # Train
            print(" Training...")
            
            train_batch_losses = []
            train_batch_accuracies = []
            
            for images, targets in tqdm(self.train_data):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.neural_network(images)
                
                # Calculate train loss and backpropagate
                train_batch_loss = self.loss_function(predictions, targets)
                train_batch_losses.append(train_batch_loss)
                train_batch_loss.backward()
                
                # Move predictions and labels to cpu for accuracy calculation
                predictions = torch.max(predictions, dim=1)[1]
                predictions = predictions.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
                
                # Calculate train accuracy
                train_batch_accuracy = sklearn.metrics.accuracy_score(y_pred=predictions, y_true=targets)
                train_batch_accuracies.append(train_batch_accuracy)
                
                # Step optimizer and clear gradients
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Calculate epoch loss and accuracy    
            train_epoch_loss = float(torch.stack(train_batch_losses).mean())
            self.history["Train Loss"].append(train_epoch_loss)
            
            train_epoch_accuracy = np.mean(train_batch_accuracies)
            self.history["Train Accuracy"].append(train_epoch_accuracy)
                
            # Validate
            print(" Validating...")
            
            val_batch_losses = []
            val_batch_accuracies = []
            
            with torch.no_grad():
                self.neural_network.eval()
                
                for images, targets in tqdm(self.val_data):
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    predictions = self.neural_network(images)
                    
                    # Calculate validation loss
                    val_batch_loss = self.loss_function(predictions, targets)
                    val_batch_losses.append(val_batch_loss)
                    
                    # Move predictions and labels to cpu for accuracy calculation
                    predictions = torch.max(predictions, dim=1)[1]
                    predictions = predictions.cpu().detach().numpy()
                    targets = targets.cpu().detach().numpy()
                    
                    # Calculate validation loss
                    val_batch_accuracy = sklearn.metrics.accuracy_score(y_pred=predictions, y_true=targets)
                    val_batch_accuracies.append(val_batch_accuracy)
            
            # Calculate epoch loss and accuracy        
            val_epoch_loss = float(torch.stack(val_batch_losses).mean())
            self.history["Validation Loss"].append(val_epoch_loss)
            
            val_epoch_accuracy = np.mean(val_batch_accuracies)
            self.history["Validation Accuracy"].append(val_epoch_accuracy)
            
    def test(self):
        """Test the model. Returns the test loss and test accuracy."""
        
        with torch.no_grad():
            print(" Testing...")
            
            test_batch_losses = []
            test_batch_accuracies = []
            
            self.neural_network.eval()
            
            for images, targets in tqdm(self.test_data):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.neural_network(images)
                
                # Calculate validation loss
                test_batch_loss = self.loss_function(predictions, targets)
                test_batch_losses.append(test_batch_loss)
                
                # Move predictions and labels to cpu for accuracy calculation
                predictions = torch.max(predictions, dim=1)[1]
                predictions = predictions.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
                
                # Calculate validation loss
                test_batch_accuracy = sklearn.metrics.accuracy_score(y_pred=predictions, y_true=targets)
                test_batch_accuracies.append(test_batch_accuracy)
                
            # Calculate test loss and accuracy     
            test_loss = float(torch.stack(test_batch_losses).mean())
            test_accuracy = np.mean(test_batch_accuracies)
        
            return test_loss, test_accuracy
    
    def plot_history(self):
        """Plot the train and validation losses and accuracies for the latest training round."""
        
        if self.history == None:
            raise ValueError("Training history could not be found. Please train the model prior to plotting its losses.")
        
        _, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        axes[0].plot(range(len(self.history["Train Loss"])), self.history["Train Loss"], label="Train")
        axes[0].plot(range(len(self.history["Validation Loss"])), self.history["Validation Loss"], label="Validation")
        axes[0].set_title(f"Train and Validation Losses for {self.history['Train Type']} Data")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        
        axes[1].plot(range(len(self.history["Train Accuracy"])), self.history["Train Accuracy"], label="Train")
        axes[1].plot(range(len(self.history["Validation Accuracy"])), self.history["Validation Accuracy"], label="Validation")
        axes[1].set_title(f"Train and Validation Accuracies for {self.history['Train Type']} Data")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()