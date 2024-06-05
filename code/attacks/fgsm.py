from ASR_2024_anonymization_module_learning.speaker_anonymization.spi import SpeakerIdentificationModel
from ASR_2024_anonymization_module_learning.speaker_anonymization.losses import speaker_verification_loss

import torch
import torch.nn as nn

from attacks.evasion_attack import EvasionAttack


class FGSM(EvasionAttack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    Taken from the Torchattacks (https://arxiv.org/abs/2010.01950) library and refactored."""

    def __init__(self, asv_model: SpeakerIdentificationModel, eps=8 / 255):
        super().__init__("FGSM", asv_model.model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]
        self.asv_model = asv_model

    def forward(self, utterances, labels):
        r"""
        Overridden.
        """

        utterances = utterances.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(utterances, labels)

        # loss = nn.CrossEntropyLoss()

        utterances.requires_grad = True
        # outputs = self.get_logits(images)
        outputs = self.asv_model.get_speakers_using_waveforms(utterances)

        # Calculate loss
        if self.targeted:
            cost = -speaker_verification_loss(outputs, target_labels)
            # cost = -loss(outputs, target_labels)
        else:
            cost = speaker_verification_loss(outputs, labels)
            # cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, utterances, retain_graph=False, create_graph=False
        )[0]

        adversarial_utterances = utterances + self.eps * grad.sign()
        adversarial_utterances = torch.clamp(adversarial_utterances, min=0, max=1).detach()

        return adversarial_utterances