from pathlib import Path

import joblib
import torch


ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def load_model():
	"""Load the trained PyTorch model from the artifacts directory."""
	return torch.load(ARTIFACTS_DIR / "housing_model.pt", weights_only=False)


def load_scaler_x():
	"""Load the feature scaler from the artifacts directory."""
	return joblib.load(ARTIFACTS_DIR / "scaler_x.joblib")


def load_scaler_y():
	"""Load the target scaler from the artifacts directory."""
	return joblib.load(ARTIFACTS_DIR / "scaler_y.joblib")


def load_imputer():
	"""Load the feature imputer from the artifacts directory."""
	return joblib.load(ARTIFACTS_DIR / "imputer.joblib")
