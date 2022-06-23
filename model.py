import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel

import gc

from config import DefaultConfig

config = DefaultConfig()

# Customized model 생성
class CustomModel(nn.Module):
    def __init__(self, conf):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(config.MODEL_NAME, config=conf)
        self.similarity_fn = nn.CosineSimilarity()
        self.sequential = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2)
        )
        
        gc.collect()
    def forward(self, input_ids=None, attention_mask=None, 
                input_ids2=None, attention_mask2=None, labels=None):
        gc.collect()
        outputs1 = self.model(
            input_ids, attention_mask=attention_mask
        )
        gc.collect()
        outputs2 = self.model(
            input_ids2, attention_mask=attention_mask2
        )
        gc.collect()
        pooler1 = outputs1[0]
        pooler2 = outputs2[0]

        # Mean
        pooler1 =  pooler1.mean(dim=1)
        pooler2 =  pooler2.mean(dim=1)

        # Normalize
        a_norm = F.normalize(pooler1, p=2, dim=1)
        b_norm = F.normalize(pooler2, p=2, dim=1)

        # scoring
        sim_score =  self.similarity_fn(a_norm, b_norm)
        sim_score = sim_score.unsqueeze(-1)

        # logits (32, 2)
        logits = self.sequential(sim_score)
        del pooler1, pooler2, a_norm, b_norm, sim_score
        
        return logits