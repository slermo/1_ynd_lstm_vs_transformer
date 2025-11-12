from src.eval_lstm import evaluate_lstm
from src.lstm_train import lstm_train
import torch
from prettytable import PrettyTable
from src.utils import my_device
import matplotlib.pyplot as plt

def eval_lstm_pipeline(config, model, tokenizer, train_loader, val_loader):
    num_epochs = config['train']['epoches']
    lr = float(config['train']['learning_rate'])
    wd = float(config['train']['weight_decay'])

    pad_token_id = tokenizer.pad_token_id
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_avg_loss = lstm_train(model, tokenizer.vocab_size, train_loader, optimizer, criterion)
        val_loss, val_acc, rog = evaluate_lstm(model, loader=val_loader, criterion = criterion, tokenizer = tokenizer)
        print_epoch_row(epoch + 1, num_epochs, train_avg_loss, val_loss, val_acc, rog)
        generate_seq(model, tokenizer)
        print('-' * 100)
        
        train_losses.append(train_avg_loss)
        val_losses.append(val_loss)

        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/weights_lstm.pt")
    
    plot_loss_curve(train_losses, val_losses)

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
    output_ids = model.generate(input_ids, max_new_tokens=10)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print('Generated example: ' + generated_text)

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()