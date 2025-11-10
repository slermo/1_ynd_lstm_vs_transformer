import torch
from src.utils import my_device

def eval_transformer_pipeline(model, val_loader):
    model.eval()
      
    epoch_validation_loss = 0
    total_loss = 0
    with torch.no_grad():
        for batch in enumerate(val_loader):
            inputs = batch['input_ids'].squeeze(1).to(my_device())
            targets = inputs.clone()
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            total_loss += loss
            valid_iterator.set_postfix({'Validation Loss': loss.item()})
            epoch_validation_loss += loss.item()