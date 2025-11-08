from src.eval_lstm import evaluate_lstm
from src.lstm_train import lstm_train
import torch

def eval_transformer_pipeline(config, model, tokenizer, train_loader, val_loader):
    num_epochs = 10
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_avg_loss = lstm_train(model, tokenizer.vocab_size, train_loader, optimizer, criterion)
        val_loss, val_acc, rogl1 = evaluate_lstm(model, loader=val_loader, criterion = criterion, tokenizer=tokenizer)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | rog {rogl1:.4f}")