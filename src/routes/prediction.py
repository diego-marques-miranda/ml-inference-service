from fastapi import APIRouter, Request
from src.schemas.prediction import FeaturesSchema, PredictionSchema

router = APIRouter(tags=["Prediction"])

@router.post("/predict", response_model=PredictionSchema)
def predict(features: FeaturesSchema, request: Request):
    service = request.app.state.prediction_service

    prediction = service.predict(features)

    return {"prediction": prediction}