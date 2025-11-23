import sys
import os
import pandas as pd
from datetime import datetime

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        features: pandas DataFrame with the same columns used during training
                  (except the target 'oil').
        Returns a numpy array of predictions.
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # Load model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Ensure features is a DataFrame
            if not isinstance(features, pd.DataFrame):
                raise CustomException("features must be a pandas DataFrame", sys)

            # Preprocessor expects the same feature columns as training.
            # If date column exists, mimic the transformation used in training:
            if 'date' in features.columns:
                features = features.copy()
                features['date'] = pd.to_datetime(features['date'], errors='coerce')
                features['month'] = features['date'].dt.month
                features['dayofyear'] = features['date'].dt.dayofyear
                features.drop(columns=['date'], inplace=True)

            # Try to coerce numeric columns to numeric dtype (so imputer/scaler can work)
            numeric_cols = [
                "down_hole_presure",
                "down_hole_temperature",
                "production_pipe_pressure",
                "choke_size_pct",
                "well_head_presure",
                "well_head_temperature",
                "choke_size_pressure",
                "month",
                "dayofyear",
            ]
            for col in numeric_cols:
                if col in features.columns:
                    features[col] = pd.to_numeric(features[col], errors='coerce')

            # Transform and predict
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Helper to build a single-row DataFrame from raw input values for prediction.
    Field names must match training features (except 'oil' target).
    """

    def __init__(
        self,
        date: str | None = None,
        down_hole_presure: float | int | None = None,
        down_hole_temperature: float | int | None = None,
        production_pipe_pressure: float | int | None = None,
        choke_size_pct: float | int | None = None,
        well_head_presure: float | int | None = None,
        well_head_temperature: float | int | None = None,
        choke_size_pressure: float | int | None = None,
    ):
        self.date = date
        self.down_hole_presure = down_hole_presure
        self.down_hole_temperature = down_hole_temperature
        self.production_pipe_pressure = production_pipe_pressure
        self.choke_size_pct = choke_size_pct
        self.well_head_presure = well_head_presure
        self.well_head_temperature = well_head_temperature
        self.choke_size_pressure = choke_size_pressure

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame with a single row and columns matching the training features.
            If `date` is provided it will be parsed and 'month' and 'dayofyear' will be added.
        """
        try:
            data = {
                "date": [self.date],
                "down_hole_presure": [self.down_hole_presure],
                "down_hole_temperature": [self.down_hole_temperature],
                "production_pipe_pressure": [self.production_pipe_pressure],
                "choke_size_pct": [self.choke_size_pct],
                "well_head_presure": [self.well_head_presure],
                "well_head_temperature": [self.well_head_temperature],
                "choke_size_pressure": [self.choke_size_pressure],
            }

            df = pd.DataFrame(data)

            # If date present convert and create features like in training
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['month'] = df['date'].dt.month
                df['dayofyear'] = df['date'].dt.dayofyear
                df = df.drop(columns=['date'])

            # Convert numeric-like columns to numeric (coerce errors to NaN)
            numeric_cols = [
                "down_hole_presure",
                "down_hole_temperature",
                "production_pipe_pressure",
                "choke_size_pct",
                "well_head_presure",
                "well_head_temperature",
                "choke_size_pressure",
                "month",
                "dayofyear",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            raise CustomException(e, sys)
