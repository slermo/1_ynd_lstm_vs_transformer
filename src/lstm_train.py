
def lstm_train(model, vocab_size, loader, optimizer,criterion):

        model.train()
        total_loss = 0

        for batch in loader:
            input_ids = batch["input_ids"]
            target_ids = batch["labels"]

            optimizer.zero_grad()
            logits = model(input_ids)  # (batch, seq_len, vocab_size)

            loss = criterion(
                logits.view(-1, vocab_size),
                target_ids.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        # print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        return avg_loss