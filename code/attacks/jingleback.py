from attacks.backdoor_attack import BackdoorAttack

import torch
from pedalboard import Pedalboard, LadderFilter, Gain, Phaser

class JingleBack(BackdoorAttack):
    def __init__(self, source_label = int, target_label = int) -> None:
        self.attack_name = "JingleBack Attack"
        
        self.source_label = source_label
        self.target_label = target_label
    
    def execute(self, utterance: torch.Tensor, sample_rate: float) -> torch.Tensor:
        utterance = utterance.numpy()
        
        board = Pedalboard([
            LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=1000),
        ])
        
        backdoored_utterance = board(utterance, sample_rate=sample_rate)
        
        return torch.from_numpy(backdoored_utterance)