import torch
from src.utils import my_device
import evaluate

def eval_transformer_pipeline(model, val_loader, tokenizer):
    model.eval()
    rouge = evaluate.load("rouge")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            inputs = batch['input_ids'].to(my_device())
            
            # Считаем лосс и сдвигаем токен на +1
            targets = inputs[:, 1:].contiguous()
            inputs = inputs[:, :-1].contiguous()
            outputs = model(input_ids=inputs, labels=targets)
            total_loss += outputs.loss.item()

            # Логика для rouge
            prompt_len = inputs.size(1) * 3 // 4
            prompts = inputs[:, :prompt_len]
            labels = inputs[:, prompt_len:]

            attention_mask = (prompts != tokenizer.pad_token_id).long().to(my_device())
            generated = model.generate(
                prompts,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=labels.size(1),
                do_sample=False,
            )

            preds = generated[:, prompt_len:]
            total_correct += (preds == labels).sum().item()
            total_tokens += labels.numel()

            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
            rouge.add_batch(predictions=pred_texts, references=labels_text)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    rouge_scores = rouge.compute()

    print_metrics(avg_loss, accuracy, rouge_scores)

    return avg_loss, accuracy, rouge_scores
    
def print_metrics(avg_loss, accuracy, rouge_scores):
    print('distilgpt2 Metrics on val dataset')
    print(f"|AvgLoss: {avg_loss:.4f} | "
          f"Acc: {accuracy:.4f} | "
          f"R1: {rouge_scores['rouge1']:.4f} | "
          f"R2: {rouge_scores['rouge2']:.4f} | "
          f"RL: {rouge_scores['rougeL']:.4f} |")