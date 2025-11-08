import torch
from torch.utils.data import Dataset

class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        
        self.inputs = []
        self.labels = []
        for line in texts:
            encodings = tokenizer(line, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
            
            token_ids = encodings["input_ids"].squeeze(0)
            attention_mask = encodings["attention_mask"].squeeze(0)

            input_ids = token_ids[:-1]
            label_ids = token_ids[1:]
            attn_mask = attention_mask[:-1]

            # Добавляем в список
            self.inputs.append({
                "input_ids": input_ids,
                "attention_mask": attn_mask,
            })
            self.labels.append(label_ids)
            
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx]["input_ids"],
            "attention_mask": self.inputs[idx]["attention_mask"],
            "labels": self.labels[idx]
        }