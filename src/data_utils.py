import pandas as pd
import csv
import re
from sklearn.model_selection import train_test_split
# функция для "чистки" текстов
def clean_string(text):
    # приведение к нижнему регистру
    text = text.lower()
    # удаление всего, кроме латинских букв, цифр и пробелов
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # удаление дублирующихся пробелов, удаление пробелов по краям
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class DataUtils:
    @staticmethod
    def from_txt_to_csv(txt_path:str, csv_path:str):
        '''
        Вспомогательный метод для создания csv из txt
        '''
        with open(txt_path, 'r', encoding='utf-8') as in_file:
            lines = [line.strip() for line in in_file if line.strip()]

        with open(csv_path, 'w', newline='', encoding='utf-8') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(["text"]) 
            for line in lines:
                writer.writerow([line])

        print("Created ", csv_path)
    
    @staticmethod
    def _clean_df(df:pd.DataFrame) -> pd.DataFrame:
        '''
        Убраем из текста теги и ссылки
        Приводим к нижнему регистру
        '''
        df = df.copy()
        df["text"] = df["text"].apply(lambda x: clean_string(x))

        pattern = r'(https?://\S+|www\.\S+)'
        df["text"] = df["text"].apply(lambda x: re.sub(pattern, "[LINK]", x))

        pattern = r'@\w+'
        df["text"] = df["text"].apply(lambda x: re.sub(pattern, "[USER]", x))

        
        return df
    
    @staticmethod
    def samples_create(config: str):
        csv_path = config["path"]
        train_ratio = config["split"]["train"]
        val_ratio = config["split"]["val"]
        test_ratio = config["split"]["test"]

        df = pd.read_csv(csv_path + '/raw_dataset.csv')
        df_clean = DataUtils._clean_df(df)

        texts = df_clean["text"].tolist()
        train_val, test = train_test_split(texts, test_size=test_ratio)
        val_relative = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(train_val, test_size=val_relative)

        print(len(train), len(val), len(test))

        pd.DataFrame({"text": train}).to_csv(f"{csv_path}/train.csv", index=False)
        pd.DataFrame({"text": val}).to_csv(f"{csv_path}/val.csv", index=False)
        pd.DataFrame({"text": test}).to_csv(f"{csv_path}/test.csv", index=False)

        return







    