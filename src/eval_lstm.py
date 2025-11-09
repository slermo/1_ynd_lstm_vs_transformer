import torch
import evaluate
from src.utils import my_device


def evaluate_lstm(model, loader, criterion, tokenizer):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    rogue = evaluate.load("rouge")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(my_device())
            labels = batch['labels'].to(my_device())
            
            
            logits = model(input_ids)
            # ??
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_tokens += labels.numel()
            # Считаем rogue только для одного элемента
            if batch_idx == 0:
                preds_text = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
                labels_text = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]
                rogue.add_batch(predictions=preds_text, references=labels_text)

           

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    rouge_results = rogue.compute()

    return avg_loss, accuracy, rouge_results["rouge1"]
