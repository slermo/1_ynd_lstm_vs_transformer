import torch.nn as nn
import torch

from src.utils import my_device

class LstmModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers = 2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout = dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0] = 0
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        logits = self.fc(out)
        return logits
    

    def generate(self, x, max_new_tokens=20, top_k=50):
        "Стратегия выбора некс токена top-k"
        with torch.no_grad():
            self.eval()
            x = x.to(my_device())
            
            emb = self.embedding(x)
            out, hidden = self.rnn(emb)
            
            for i in range(max_new_tokens):
                logits = self.fc(out[:, -1, :])

                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # # Принты для дебага
                # top_probs, top_indices = torch.topk(probs[0], k=3)
                # if tokenizer:
                #     print(f"\nШаг {i+1}:")
                #     for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
                #         token_id = idx.item()
                #         token_text = tokenizer.decode([token_id])
                #         print(f"  {rank}. ID:{token_id:5d} prob:{prob.item():.4f} -> '{token_text}'")
                
                
                x = torch.cat([x, next_token], dim=1)
                emb = self.embedding(next_token)
                out, hidden = self.rnn(emb, hidden)
            
            return x
        