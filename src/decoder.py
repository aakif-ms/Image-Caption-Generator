import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, vocab_size, num_layers):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, _ = self.lstm(embeddings)
        output = self.linear(self.dropout(lstm_out))
        return output