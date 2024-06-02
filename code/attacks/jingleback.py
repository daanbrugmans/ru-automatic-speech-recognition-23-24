from attacks.abstract_backdoor_attack import AbstractBackdoorAttack

import pedalboard
import torch

class Jingleback(AbstractBackdoorAttack):
    def __init__(self, source_label = int, target_label = int) -> None:
        self.attack_name = "Jingleback Attack"
        
        self.source_label = source_label
        self.target_label = target_label
    
    def execute(self, audio: torch.Tensor):
        pass