import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, alpha1=1, alpha2=0, alpha3=0):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

    def forward(self, label, logit, logit_ref, flag):
        ce_loss = nn.CrossEntropyLoss()
        kl_loss = nn.KLDivLoss()

        prob_ref = torch.softmax(logit_ref / 1, dim=-1)
        prob = torch.softmax(logit / 1, dim=-1)

        loss1 = ce_loss(logit, label)
        loss2 = kl_loss(prob, prob_ref)
        loss3 = kl_loss(prob_ref, prob)
        
        loss = self.alpha1*flag*loss1 + self.alpha2*loss2 + self.alpha3*loss3

        return loss