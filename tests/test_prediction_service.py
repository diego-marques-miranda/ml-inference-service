import numpy as np
import torch

from src.schemas.prediction import FeaturesSchema
from src.services.prediction import PredictionService

class FakeModel:
    def eval(self):
        pass

    def __call__(self, tensor):
        return torch.tensor([[1.0]])


class FakeScalerX:
    def transform(self, features):
        return features


class FakeScalerY:
    def inverse_transform(self, prediction):
        return prediction * 100


class FakeImputer:
    def transform(self, features):
        return features


def create_service():
    return PredictionService(
        model=FakeModel(),
        scaler_x=FakeScalerX(),
        scaler_y=FakeScalerY(),
        imputer=FakeImputer(),
    )


def create_features():
    return FeaturesSchema(
        longitude=-122.23,
        latitude=37.88,
        housing_median_age=41,
        total_rooms=880,
        total_bedrooms=129,
        population=322,
        households=126,
        median_income=8.3252,
    )


def test_feature_mapping_preserves_expected_order():
    service = create_service()
    features = create_features()

    mapped_features = service.feature_mapping(features)

    assert mapped_features == [
        -122.23,
        37.88,
        41,
        880,
        129,
        322,
        126,
        8.3252,
    ]


def test_predict_returns_prediction_in_original_scale():
    service = create_service()
    features = create_features()

    prediction = service.predict(features)

    assert isinstance(prediction, (float, np.floating))
    assert prediction == 100.0