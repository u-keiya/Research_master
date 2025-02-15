from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import numpy as np

class RewardModel(nn.Module):
    def __init__(self, lam=0.1):
        super().__init__()
        self.sentence2vec = SentenceTransformer('/path/to/all-mpnet-base-v2')
        self.lam = lam
    
    def _cos_sim(self, v1, v2):
        sims = np.array([])
        for a, b in zip(v1,v2):
            sims = np.append(sims, (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        return sims
    
    def forward(self, outputs, references, outputs_length, ref_length):
        outputs_emb = self.sentence2vec.encode(outputs)
        references_emb = self.sentence2vec.encode(references)
        sims = self._cos_sim(outputs_emb, references_emb)
        reduce_rate = (np.array(ref_length, dtype=float) - np.array(outputs_length, dtype=float)) / np.array(ref_length, dtype=float)
        rewards_np = (1-self.lam) * sims + self.lam * reduce_rate
        
        rewards = []
        for r in rewards_np:
            rewards.append(torch.tensor([r]))
        return rewards