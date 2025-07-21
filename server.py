#NEWEST today server.py
# server.py
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import joblib
import asyncio
import os
import numpy as np
import pandas as pd
from typing import List
from myparser import fetch_binance_ohlcv, fetch_moex_ohlcv

from fastapi import FastAPI, HTTPException
import joblib

app = FastAPI()

# глобальная переменная
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("models/pipeline_x.pkl")
        print("hooray-Модель загружена")
    except Exception as e:
        print("oh noo ошибка загрузки модели:", e)



DATABASE_URL = "sqlite:///./models.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------- DATABASE SCHEMA ----------------------
class ModelMeta(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    path = Column(String)

class ModelMetrics(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer)
    mae = Column(Float)
    rmse = Column(Float)
    accuracy = Column(Float)

Base.metadata.create_all(bind=engine)

# ---------------------- ML MODEL LOAD ----------------------
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# In-memory cache
active_model = None

@app.get("/parse/binance") #REST
async def parse_binance(symbol: str = Query("BTC/USDT")):
    data = await fetch_binance_ohlcv(symbol)
    return {"symbol": symbol, "ohlcv": data}

from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/binance")
async def ws_binance(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            symbol = await websocket.receive_text()  # получаем тикер
            data = await fetch_binance_ohlcv(symbol)
            await websocket.send_json({"symbol": symbol, "ohlcv": data})
    except WebSocketDisconnect:
        print("Binance WS disconnected")
        await websocket.close()

@app.websocket("/ws/moex")
async def ws_moex(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            ticker = await websocket.receive_text()
            data = await fetch_moex_ohlcv(ticker)
            await websocket.send_json({"ticker": ticker, "ohlcv": data})
    except WebSocketDisconnect:
        print("MOEX WS disconnected")
        await websocket.close()

# ---------------------- MOEX ENDPOINT ----------------------
@app.get("/parse/moex")
async def parse_moex(ticker: str = Query("GAZP")):
    data = await fetch_moex_ohlcv(ticker)
    return {"ticker": ticker, "ohlcv": data}

# ---------------------- MODEL UPLOAD ----------------------
from fastapi import UploadFile, File, APIRouter
import shutil
import os

@app.post("/model/upload")
async def upload_model(file: UploadFile = File(...)):
    model_name = file.filename
    model_path = f"models/{model_name}"

    # Сохраняем файл
    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Добавляем в SQLite
    db = SessionLocal()
    record = ModelMeta(name=model_name, path=model_path)
    db.add(record)
    db.commit()
    db.refresh(record)
    db.close()

    return {"status": "uploaded", "model_id": record.id, "path": model_path}
  
# ---------------------- MODEL LIST ----------------------
@app.get("/model/list")
def list_models():
    db = SessionLocal()
    models = db.query(ModelMeta).all()
    db.close()
    return [
        {
            "id": m.id,
            "name": m.name,
            "uploaded_at": m.uploaded_at,
            "path": m.path
        } for m in models
    ]
# ---------------------- MODEL INFO ----------------------
@app.get("/model/info/{model_id}")
def model_info(model_id: int):
    db = SessionLocal()
    model_meta = db.query(ModelMeta).filter(ModelMeta.id == model_id).first()
    db.close()

    if model_meta is None:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
      model = joblib.load(model_meta.path)
      if hasattr(model, "get_params"):  # обычная модель
        params = model.get_params()
      elif hasattr(model, "named_steps"):  # Pipeline
        params = model.named_steps["model"].get_params()
      else:
        params = {"error": "Unknown model type"}
    except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

    # try:
    #     model = joblib.load(model_meta.path)
    #     params = model.get_params() if hasattr(model, "get_params") else str(model)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    return {
        "model_id": model_id,
        "name": model_meta.name,
        "path": model_meta.path,
        "params": params
    }


# ---------------------- PREDICT ----------------------
class InputData(BaseModel):
    data: List[List[float]]
    
from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    data: List[List[float]]  # 2D список: [ [x1, x2, ..., xn], ...]

from fastapi import HTTPException, Query
import joblib
import numpy as np
from pydantic import BaseModel
import os
import sqlite3
from typing import List
from pydantic import BaseModel

class PredictRequest(BaseModel):
    data: List[List[float]]  # assuming list of samples

@app.post("/model/predict")
def predict(input: PredictRequest, model_id: int = Query(...)):
    # подключение к БД, получить путь к модели
    conn = sqlite3.connect("models.db")
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM ModelMeta WHERE id=?", (model_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Model ID {model_id} not found")

    model_path = row[0]

    # Проверка, существует ли файл
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file {model_path} not found")

    # Загрузка модели
    model = joblib.load(model_path)
    print("Model type:", type(model))

    try:
        input_data = np.array(input.data)
        predictions = model.predict(input_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
   
#доп НАДО ЕЩЕ  BACKTEST


from fastapi import APIRouter
from metrics import compute_metrics_regression
#описание Pydantic-модель входа 
from pydantic import BaseModel
from typing import List

@app.post("/model/metrics")
class MetricsRequest(BaseModel):
    y_true: List[float]
    y_pred: List[float]
  

def compute_metrics(data: MetricsRequest):
    y_true = np.array(data.y_true)
    y_pred = np.array(data.y_pred)
    acc = calculate_accuracy(y_pred, y_true)
    return {"accuracy": acc}



def model_metrics(request_data: dict):
    model = joblib.load("models/pipeline_x.pkl")
    x = pd.DataFrame(request_data["features"])
    y_true = request_data["target"]

    x_transformed = model.transform(x)
    y_pred = model.predict(x_transformed)

    return compute_metrics_regression(y_true, y_pred)

#@app.post("/model/backtest")
#def backtest_model(backtest_input: BacktestRequest):
    # 1. Load model
    #model = load_pipeline(backtest_input.model_id)
    # 2. Load data
    #data = fetch_crisis_data(symbol=backtest_input.symbol)
    # 3. Feature engineering (без target!)
    #features = build_features(data)
    # 4. Predict
    #predictions = model.predict(features)
    # 5. Eval metrics
    #ml_metrics = evaluate_ml(backtest_input.true_target, predictions)
    #financial_metrics = evaluate_financials(predictions)
    
    #return {"ml_metrics": ml_metrics, "financial_metrics": financial_metrics}

