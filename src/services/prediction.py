import numpy as np
import torch    

class PredictionService:
    def __init__(self, model, scaler_x, scaler_y, imputer):
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.imputer = imputer

    def feature_mapping(self, features):
        features_dict = features.model_dump()  # Convert Pydantic model to dictionary

        # Map the input features to the expected order and format
        feature_order = [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
        ]

        mapped_features = []

        for feature in feature_order:
            mapped_features.append(features_dict.get(feature))  

        return mapped_features

    def preprocess(self, features):
        # Convert to numpy array and reshape for a single sample
        features_array = np.array(features).reshape(1, -1)

        features_array = self.imputer.transform(features_array)  # Impute missing values

        # Scale the features using the pre-fitted scaler
        scaled_features = torch.tensor(
            self.scaler_x.transform(features_array), dtype=torch.float32
        )

        return scaled_features

    def inference(self, tensor):
        # Set the model to evaluation mode
        self.model.eval()

        # Disable gradient calculation during inference
        with torch.no_grad():
            prediction = self.model(tensor)

        return prediction

    def postprocess(self, prediction):
        # Convert the prediction to a numpy array and reshape for a single sample
        prediction_array = prediction.numpy().reshape(1, -1)

        # Inverse scale the prediction using the pre-fitted scaler
        inverse_scaled_prediction = self.scaler_y.inverse_transform(prediction_array)

        return inverse_scaled_prediction[0][0]  # Return the first element of the first sample

    def predict(self, features):
        # Step 1: Feature Mapping
        mapped_features = self.feature_mapping(features)

        # Step 2: Preprocessing
        preprocessed_features = self.preprocess(mapped_features)

        # Step 3: Inference
        prediction = self.inference(preprocessed_features)

        # Step 4: Postprocessing
        final_prediction = self.postprocess(prediction)

        return final_prediction