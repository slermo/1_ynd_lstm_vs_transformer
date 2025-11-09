# INIT

Update dep with:

```python
poetry install
```

Use this to generate `.venv`:

```python
poetry run jupyter notebook
```

weights_lstm_v1.pt - обучение заняло 248 минут
Процесс обучения:

```
Epoch 1/10 | Train Loss: 4.8766 | Val Loss: 4.5478 | Val Acc: 0.3235
Epoch 2/10 | Train Loss: 4.4416 | Val Loss: 4.4047 | Val Acc: 0.3348
Epoch 3/10 | Train Loss: 4.3367 | Val Loss: 4.3678 | Val Acc: 0.3367
Epoch 4/10 | Train Loss: 4.2761 | Val Loss: 4.3127 | Val Acc: 0.3428
Epoch 5/10 | Train Loss: 4.2250 | Val Loss: 4.2865 | Val Acc: 0.3458
Epoch 6/10 | Train Loss: 4.1971 | Val Loss: 4.3203 | Val Acc: 0.3401
Epoch 7/10 | Train Loss: 4.1742 | Val Loss: 4.2624 | Val Acc: 0.3491
Epoch 8/10 | Train Loss: 4.1559 | Val Loss: 4.2846 | Val Acc: 0.3455
Epoch 9/10 | Train Loss: 4.1397 | Val Loss: 4.2450 | Val Acc: 0.3512
Epoch 10/10 | Train Loss: 4.1213 | Val Loss: 4.2405 | Val Acc: 0.3521
```
