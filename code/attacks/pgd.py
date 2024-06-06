import torch
import torch.nn as nn

from attacks.evasion_attack import EvasionAttack


class PGD(EvasionAttack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Taken from the Torchattacks (https://arxiv.org/abs/2010.01950) library and refactored."""

    def __init__(self, asv_model, eps=8/255, alpha=2/255, steps=10, random_start=True):
        super().__init__("FGSM", asv_model.model)
        self.eps = eps
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
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
        
        adversarial_utterance = utterance.clone().detach()
        adversarial_model_input = model_input.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adversarial_model_input = adversarial_model_input + torch.empty_like(adversarial_model_input).uniform_(-self.eps, self.eps)
            adversarial_model_input = torch.clamp(adversarial_model_input, min=0, max=1).detach()
        
        loss = nn.BCEWithLogitsLoss()
        
        for _ in range(self.steps):
            adversarial_model_input.requires_grad = True
            
            model_output = self.model(adversarial_model_input)
            model_output = model_output.last_hidden_state.mean(dim=1).to(self.device)
            model_output = self.classifier(model_output)
                                    
            # Calculate loss            
            labels = torch.full((1, model_output.size(dim=1)), label, dtype=torch.float32)
            labels = labels.to(self.device)
            
            if self.targeted:
                cost = -loss(model_output, labels)
            else:
                cost = loss(model_output, labels)
                                
            # Update adversarial images
            grad = torch.autograd.grad(cost, adversarial_model_input, retain_graph=False, create_graph=False)[0]

            adversarial_utterance = adversarial_utterance.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adversarial_utterance - utterance, min=-self.eps, max=self.eps)
            adversarial_utterance = torch.clamp(utterance + delta, min=0, max=1).detach()

        return adversarial_utterance