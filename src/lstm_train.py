from src.utils import my_device
import torch

def lstm_train(model, vocab_size, loader, optimizer,criterion):
        model.train()
        total_loss = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(my_device())
            target_ids = batch["labels"].to(my_device())

            optimizer.zero_grad()
            logits = model(input_ids)  # (batch, seq_len, vocab_size)
            
            # Сдвигаем таргеты, чтобы предугадывать следующий токен
            logits = logits[:, :-1, :].contiguous()
            target_ids = target_ids[:, 1:].contiguous()

            # Приводим к 1D
            loss = criterion(
                logits.view(-1, vocab_size),
                target_ids.view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        return avg_loss