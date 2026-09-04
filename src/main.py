from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.loaders.artifacts import (
    load_imputer,
    load_model,
    load_scaler_x,
    load_scaler_y,
)
from src.routes import prediction
from src.services.prediction import PredictionService

@asynccontextmanager
async def lifespan(app: FastAPI):

    model = load_model()
    scaler_x = load_scaler_x()
    scaler_y = load_scaler_y()
    imputer = load_imputer()

    app.state.prediction_service = PredictionService(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        imputer=imputer
    )

    yield

app = FastAPI(
    title="ML Inference Service",
    lifespan=lifespan
)
app.include_router(prediction.router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "The service is working fine."}
