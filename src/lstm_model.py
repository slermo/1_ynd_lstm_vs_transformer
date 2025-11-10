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
    

    def generate(self, x, max_new_tokens=20, tokenizer=None):
        with torch.no_grad():
            self.eval()
            x = x.to(my_device())
            
            emb = self.embedding(x)
            out, hidden = self.rnn(emb)
            
            for i in range(max_new_tokens):
                logits = self.fc(out[:, -1, :])
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                # ОТЛАДКА
                if tokenizer:
                    token_id = next_token[0, 0].item()
                    token_text = tokenizer.decode([token_id])
                    print(f"{i+1}. ID:{token_id} -> '{token_text}'")
                # КОНЕЦ ОТЛАДКИ
                
                x = torch.cat([x, next_token], dim=1)
                emb = self.embedding(next_token)
                out, hidden = self.rnn(emb, hidden)
            
            return x
        
        # def generate(self, x, max_new_tokens=20):
        # with torch.no_grad():
        #     self.eval()
        #     x = x.to(my_device())
        
        #     # Шаг 1: Обработать начальную последовательность один раз
        #     emb = self.embedding(x)
        #     out, hidden = self.rnn(emb)  # получаем initial hidden state
        #     # Шаг 2: Генерировать токены один за другим
        #     for _ in range(max_new_tokens):
        #         # Берём логиты последнего токена
        #         logits = self.fc(out[:, -1, :])  # (batch, vocab_size)
        #         probs = torch.nn.functional.softmax(logits, dim=-1)
        #         next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (batch, 1)
                
        #         # Добавляем к последовательности
        #         x = torch.cat([x, next_token], dim=1)
    
        #         # Обрабатываем только новый токен
        #         emb = self.embedding(next_token)  # (batch, 1, hidden_dim)
        #         out, hidden = self.rnn(emb, hidden)  # используем предыдущий hidden
        
        #     return x