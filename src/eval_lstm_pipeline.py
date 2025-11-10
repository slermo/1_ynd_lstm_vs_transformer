from src.eval_lstm import evaluate_lstm
from src.lstm_train import lstm_train
import torch
from prettytable import PrettyTable
from src.utils import my_device

def eval_lstm_pipeline(config, model, tokenizer, train_loader, val_loader):
    num_epochs = config['train']['epoches']
    lr = float(config['train']['learning_rate'])
    wd = float(config['train']['weight_decay'])

    pad_token_id = tokenizer.pad_token_id
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_avg_loss = lstm_train(model, tokenizer.vocab_size, train_loader, optimizer, criterion)
        generate_seq(model, tokenizer)
        val_loss, val_acc, rog = evaluate_lstm(model, loader=val_loader, criterion = criterion, tokenizer=tokenizer)
        print_epoch_row(epoch + 1, num_epochs, train_avg_loss, val_loss, val_acc, rog)

        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/weights_lstm.pt")


def print_epoch_row(epoch, num_epochs, train_loss, val_loss, val_acc, rouge_scores):
    print(f"| {epoch:>2}/{num_epochs:<2} | "
          f"Train: {train_loss:.4f} | "
          f"Val: {val_loss:.4f} | "
          f"Acc: {val_acc:.4f} | "
          f"R1: {rouge_scores['rouge1']:.4f} | "
          f"R2: {rouge_scores['rouge2']:.4f} | "
          f"RL: {rouge_scores['rougeL']:.4f} |")
        
def generate_seq(model, tokenizer):
    prompt = "The meaning of life"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(my_device())
    output_ids = model.generate(input_ids, max_new_tokens=30, tokenizer= tokenizer)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(generated_text)
