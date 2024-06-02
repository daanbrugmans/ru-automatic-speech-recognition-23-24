from abc import ABC, abstractmethod

import torch


class AbstractBackdoorAttack(ABC):
    def __init__(self) -> None:
        self.attack_name: str
        
        self.source_label: int
        self.target_label: int
    
    @abstractmethod
    def execute(self, image: torch.Tensor):
        raise NotImplementedError("The BackdoorAttack base class is abstract. Please use an implementation of an Attack.")