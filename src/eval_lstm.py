import torch
import evaluate
from src.utils import my_device

# @torch.no_grad() # декоратор
def evaluate_lstm(model, loader, criterion, tokenizer):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    rouge = evaluate.load("rouge")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(my_device())
            labels = batch['labels'].to(my_device())

            logits = model(input_ids)

            # Сдвигаем: предсказания для [:-1], цели для [1:]
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_tokens += labels.numel()
            
            # Считаем rogue для последовательности +1 
            preds_text = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
            labels_text = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]
            rouge.add_batch(predictions=preds_text, references=labels_text)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    rouge_results = rouge.compute()

    return avg_loss, accuracy, rouge_results
