import torch
import torch.nn as nn

from attacks.evasion_attack import EvasionAttack


class FGSM(EvasionAttack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Taken from the Torchattacks (https://arxiv.org/abs/2010.01950) library and refactored."""

    def __init__(self, asv_model, eps=8/255):
        super().__init__("FGSM", asv_model.model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]
        self.processor = asv_model.processor
        self.classifier = asv_model.classifier

    def forward(self, utterance, label):
        r"""
        Overridden.
        """
        
        torch.cuda.empty_cache()
        
        utterance = torch.unsqueeze(utterance, dim=0)
        utterance = utterance.to(self.device)
        utterance.requires_grad = True
        
        model_input = self.processor(utterance, sampling_rate=16000, return_tensors="pt").data["input_values"]
        model_input = model_input.squeeze(0).to(self.device)
        model_input.requires_grad = True
        
        model_output = self.model(model_input)
        model_output = model_output.last_hidden_state.mean(dim=1).to(self.device)
        model_output = self.classifier(model_output)
                                
        # Calculate loss
        loss = nn.BCEWithLogitsLoss()
        
        labels = torch.full((1, model_output.size(dim=1)), label, dtype=torch.float32)
        labels = labels.to(self.device)
        
        if self.targeted:
            cost = -loss(model_output, labels)
        else:
            cost = loss(model_output, labels)
                            
        # Update adversarial images
        grad = torch.autograd.grad(cost, model_input, retain_graph=False, create_graph=False)[0]

        adversarial_utterance = utterance + self.eps * grad.sign()
        adversarial_utterance = torch.clamp(adversarial_utterance, min=0, max=1).detach()

        return adversarial_utterance