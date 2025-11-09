# 0 Инициализация проекта

Update dep with:

```python
poetry install
```

Use this to generate `.venv`:

```python
poetry run jupyter notebook
```

Добавть файл `raw_dataset.csv` в папку `data/`

# 1 Первая модель LSTM

## Характеристика

DataLoader: batch_size=256
torch.optim.Adam: lr=1e-3

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

Пример вывода:
Original: cjcroll why are you sad
Prompt: cjcroll why
Generated: cjcroll why

## Итог по первой версии

Модель переобучилась и стала запоминать и повторять раннее введенный текст `:(`
Сохранена как `weights_lstm_v1.pt`

# 2 Вторая модель LSTM

## Характеристика

1. Добавил dropout nn.LSTM = 0.3 (Раннее не было)
2. Добавил num_layers nn.LSTM = 2 (Раннее не было)
3. Добавил Инициализацию весов
4. Добавил измерение ROGUE для одного батча
5. Поменял количество эпох на 5, чтобы быстрее проверить результат

weights_lstm_v2.pt - обучение заняло 83 минуты
Процесс обучения:

```
Epoch 1/5 | Train Loss: 5.3885 | Val Loss: 4.9642 | Val Acc: 0.2878 | rog 0.0898
Epoch 2/5 | Train Loss: 4.8453 | Val Loss: 4.8204 | Val Acc: 0.2982 | rog 0.1031
Epoch 3/5 | Train Loss: 4.7550 | Val Loss: 4.7219 | Val Acc: 0.3073 | rog 0.1135
Epoch 4/5 | Train Loss: 4.7099 | Val Loss: 4.6936 | Val Acc: 0.3098 | rog 0.1176
Epoch 5/5 | Train Loss: 4.6927 | Val Loss: 4.6766 | Val Acc: 0.3113 | rog 0.1146
```

Краткий отчет:

- Стабильное снижение loss
- ROUGE растёт, но темп роста замедляется
- Accuracy растёт линейно

Пример вывода:

```
PROMPT: christinemc0828 yes i will admit i was pleasantly surprised with the quality of
LSTM result: christinemc0828 yes i will admit i was pleasantly surprised with the quality of�������������� petslu lower�!!
```

## Итог по второй версии

Возможно, что нехватка эпох обучения привела к тому, что модель генирирует случайный шум. Тут либо проблема в декдировании, либо малом обучении.
Еще есть предположение, что не стоит использовать `bert-base-uncased` в задачах на генерацию

# 3 Сравнение двух моделей

В данной работе сравниваются две архитектуры для генреации текста: LSTM и Transformer. LSTM оказалась не подходящей для этой задачи с данными настройками, а Transformer наоборот показывает хорошие метрики ROUGE, что говорит о совпадении с референсом

Это показывает, что архитектура self-attention лучше подходит для задач генерации текста
