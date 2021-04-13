import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from constants import *
from helper import get_word_id
class Attention(nn.Module):

    def __init__(self, query_size, key_size):
        super(Attention, self).__init__()
        self.weights = nn.Parameter(torch.rand(key_size, query_size) * 0.2 - 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, query, key, mask):
        
        batch_size = key.size(0)
        time_step = key.size(1)
        weights = self.weights.repeat(batch_size, 1, 1) # (batch_size, key_size, query_size)
        query = query.unsqueeze(-1)    # (batch_size, query_size, 1)
        mids = weights.matmul(query)    # (batch_size, key_size, 1)
        mids = mids.repeat(time_step, 1, 1, 1).transpose(0, 1) # (batch_size, time_step, key_size, 1)
        key = key.unsqueeze(-2)    # (batch_size, time_step, 1, key_size)
        scores = torch.tanh(key.matmul(mids).squeeze() + self.bias)   # (batch_size, time_step, 1, 1)
        scores = scores.squeeze()   # (batch_size, time_step)
        scores = scores - scores.max(dim=1, keepdim=True)[0]
        scores = torch.exp(scores) * mask
        attn_weights = scores / scores.sum(dim=1, keepdim=True)
        return attn_weights

class IAN(nn.Module):

    def __init__(self,embedding):
        super(IAN, self).__init__()
        self.vocab_size = len(get_word_id())
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.l2_reg = l2_reg
        self.max_aspect_len = max_aspect_len
        self.max_context_len = max_context_len

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        self.aspect_lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)
        self.context_lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)
        self.aspect_attn = Attention(self.hidden_size, self.hidden_size)
        self.context_attn = Attention(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size * 2, self.n_class)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding))

    def forward(self, aspect, context, aspect_mask, context_mask):
        aspect = self.embedding(aspect)
        aspect = self.dropout(aspect)
        aspect_output, _ = self.aspect_lstm(aspect)
        aspect_output = aspect_output * aspect_mask.unsqueeze(-1)
        aspect_avg = aspect_output.sum(dim=1, keepdim=False) / aspect_mask.sum(dim=1, keepdim=True)
        context = self.embedding(context)
        context = self.dropout(context)
        context_output, _ = self.context_lstm(context)
        context_output = context_output * context_mask.unsqueeze(-1)
        context_avg = context_output.sum(dim=1, keepdim=False) / context_mask.sum(dim=1, keepdim=True)
        aspect_attn = self.aspect_attn(context_avg, aspect_output, aspect_mask).unsqueeze(1)
        aspect_features = aspect_attn.matmul(aspect_output).squeeze()
        context_attn = self.context_attn(aspect_avg, context_output, context_mask).unsqueeze(1)
        context_features = context_attn.matmul(context_output).squeeze()
        features = torch.cat([aspect_features, context_features], dim=1)
        features = self.dropout(features)
        output = self.fc(features)
        output = torch.tanh(output)
        return output



# traindata = IanDataset('dataset_train.npz')