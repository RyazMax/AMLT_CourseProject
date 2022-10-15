from turtle import forward
import torch

from torch import nn

class SpeakerExtractor(nn.Module):
    def __init__(self, feature_extractor: nn.Module, n_class: int):
        self.feature_extractor = feature_extractor
        self.feature_extractor.requires_grad_(False) # Freee pretrained model
        
        self.segment_level = torch.nn.Sequential(
            torch.nn.Linear(756, 512),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(512, n_class)

    def forward(self, x):
        repr = self.feature_extractor(x).last_hidden_state
        mean = torch.mean(repr, 1)
        std = torch.std(repr, 1)

        st_pooling = torch.cat((mean, std), 1)
        emb = self.segment_level(st_pooling)
        if self.train:
            return self.head(emb)
        else:
            return emb