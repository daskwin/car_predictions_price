from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
from fastapi.responses import FileResponse

app = FastAPI()

# загружаем нашу модель
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# загружаем стандартизатор
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# колонки, на которых обучалась модель
model_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol',
              'seller_type_Individual', 'seller_type_Trustmark Dealer', 'transmission_Manual', 'owner_Fourth & Above Owner',
              'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner', 'seats_4', 'seats_5', 'seats_6', 'seats_7',
              'seats_8', 'seats_9', 'seats_10', 'seats_14']

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    # убираем ненужные символы и приводим столбцы к нужным типам данных
    df['mileage'] = df['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype(float)
    df['engine'] = df['engine'].str.replace(' CC', '').astype(int)
    df['max_power'] = df['max_power'].str.replace(' bhp', '').astype(float)

    if 'selling price' in df.columns:
        df = df.drop(columns=['selling price'])

    df = df.drop(columns=['torque', 'name'])

    df['seats'] = df['seats'].astype(int)

    # работа с категориальными переменными
    categorical_columns = list(df.select_dtypes(include=['object']).columns) + ['seats']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    miss_cols = list(set(model_cols) - set(df.columns))

    df[miss_cols] = 0
    df = df[model_cols]

    # стандартизация
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

    return df_scaled

# класс базового объекта
class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

# класс с коллецией объектов
class Items(BaseModel):
    objects: List[Item]

# метод для отправки json`а
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])
    processed_data = preprocess_data(df)
    prediction = model.predict(processed_data)
    return prediction

# метод для отправки csv файла, возвращает файл в директорию этого проекта
@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> FileResponse:
    df = pd.read_csv(file.file)
    processed_data = preprocess_data(df)
    predictions = model.predict(processed_data)
    df['predictions'] = predictions # столбец за место 'selling_price'

    df.to_csv("predictions.csv", index=False)

    return FileResponse(path="predictions.csv", filename="predictions.csv")
