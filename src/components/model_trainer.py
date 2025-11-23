import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training & test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(n_estimators=50),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(n_estimators=50),
                "CatBoost Regressor": CatBoostRegressor(verbose=False, iterations=50),
                "AdaBoost": AdaBoostRegressor(n_estimators=50)
            }

            # **SUPER SMALL param grids (FAST)**
            params = {
                "Random Forest": {"n_estimators": [50]},
                "Decision Tree": {"max_depth": [None, 5]},
                "Gradient Boosting": {"learning_rate": [0.1]},
                "Linear Regression": {},
                "XGBRegressor": {"learning_rate": [0.1]},
                "CatBoost Regressor": {"learning_rate": [0.1]},
                "AdaBoost": {"learning_rate": [0.1]}
            }

            logging.info("Evaluating models (FAST MODE)")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_name = max(model_report, key=lambda name: model_report[name]["test_r2"])
            best_model = models[best_model_name]

            logging.info(f"Best model selected: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            preds = best_model.predict(X_test)
            r2 = r2_score(y_test, preds)

            return r2

        except Exception as e:
            raise CustomException(e, sys)
