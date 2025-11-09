import torch.nn as nn
import torch

from src.utils import my_device

class LstmModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers = 2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout = dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    #     self._init_weights()
    
    # def _init_weights(self):
    #     nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
    #     self.embedding.weight.data[0] = 0
    #     nn.init.xavier_uniform_(self.fc.weight)
    #     nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        logits = self.fc(out)
        return logits
    
    # Генерация последовательности токенов
    @torch.no_grad()
    def generate(self, x, max_new_tokens=20):
        self.eval()
        x = x.to(my_device())
        hidden = None

        for _ in range(max_new_tokens):
            emb = self.embedding(x)
            out, hidden = self.rnn(emb, hidden)
            logits = self.fc(out[:, -1, :]) # берём последний токен
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1).unsqueeze(1)
            x = torch.cat([x, next_token], dim=1)
        return x