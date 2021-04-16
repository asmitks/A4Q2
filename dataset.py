from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import *
from helper import get_word_id,get_final_data


class sentenceDataset(Dataset):

    def __init__(self, path):
        data = np.load(path)
        self.aspects = torch.from_numpy(data['aspects']).long()
        self.contexts = torch.from_numpy(data['contexts']).long()
        self.labels = torch.from_numpy(data['labels']).long()
        self.aspect_lens = torch.from_numpy(data['aspect_lens']).long()
        self.context_lens = torch.from_numpy(data['context_lens']).long()
        aspect_max_len = self.aspects.size(1)
        context_max_len = self.contexts.size(1)
        
    def __getitem__(self, index):
        return self.aspects[index], self.contexts[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


