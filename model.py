import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import *
from helper import get_word_id

class interactiveAttentionNetwork(nn.Module):

    def __init__(self,embedding):
        super(interactiveAttentionNetwork, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=len(get_word_id()), embedding_dim=300)
        self.aspect_lstm = nn.LSTM(input_size=300, hidden_size=300, batch_first=True)
        self.context_lstm = nn.LSTM(input_size=300, hidden_size=300, batch_first=True)
        self.aspect_attn = Attention(300, 300)
        self.context_attn = Attention(300, 300)
        self.fc = nn.Linear(600, 3)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding))

        self.attention_required = attention_required


        self.l2_reg = l2_reg
        self.dropout = nn.Dropout(dropout)



    def forward(self, aspect, context):

        '''making context embeddings'''
        context = self.embedding(context)
        context = self.dropout(context)
        context_output, _ = self.context_lstm(context)
        mean_context = torch.mean(context_output,1)


        '''making aspect embeddings'''
        aspect = self.embedding(aspect)
        aspect = self.dropout(aspect)
        aspect_output, _ = self.aspect_lstm(aspect)
        mean_aspect = torch.mean(aspect_output,1)
        
        

        
        
        if self.attention_required:

            aspect_attn = self.aspect_attn(mean_context, aspect_output).unsqueeze(1)
            aspect_features = aspect_attn.matmul(aspect_output).squeeze()
            context_attn = self.context_attn(mean_aspect, context_output).unsqueeze(1)
            context_features = context_attn.matmul(context_output).squeeze()
            features = torch.cat([aspect_features, context_features], dim=1)
            features = self.dropout(features)
            output = self.fc(features)
            output = torch.tanh(output)
            return output

        else:
            '''Concatenating in case attention_required is false'''
            features = torch.cat([mean_aspect, mean_context], dim=1)
            features = self.dropout(features)
            output = self.fc(features)
            output = torch.tanh(output)
            return output

         

class Attention(nn.Module):

    def __init__(self, query_size, key_size):
        super(Attention, self).__init__()
        self.weights = nn.Parameter(torch.rand(key_size, query_size)).repeat(128, 1, 1) 
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, query, key):
        
        query = query.unsqueeze(-1)    
        inter = self.weights.matmul(query)   
        inter = inter.repeat(key.size(1), 1, 1, 1).transpose(0, 1) 
        key = key.unsqueeze(-2)   
        scores = torch.tanh(key.matmul(inter).squeeze() + self.bias)   
        scores = scores.squeeze()   
        scores = scores - scores.max(dim=1, keepdim=True)[0]
        scores = torch.exp(scores) 
        attn_weights = scores / scores.sum(dim=1, keepdim=True)
        return attn_weights

