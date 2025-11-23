import os
import sys
import pickle
import dill
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save python object to disk using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load python object from disk. Try dill first, then pickle.
    """
    try:
        with open(file_path, "rb") as file_obj:
            try:
                # try dill (can handle more complex objects)
                return dill.load(file_obj)
            except Exception:
                file_obj.seek(0)
                return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    """
    Evaluate regression models using GridSearchCV and return a detailed report.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : arrays or dataframes
    models : dict -> {"model_name": sklearn_estimator, ...}
    params : dict -> {"model_name": {param_grid}, ...}

    Returns
    -------
    report : dict
      {"model_name": {
           "train_r2": float,
           "train_rmse": float,
           "test_r2": float,
           "test_rmse": float,
           "best_params": dict
       }, ...
      }
    """
    try:
        report = {}

        for name, model in models.items():
            param_grid = params.get(name, {})

            # Use RMSE-based scoring for GridSearch (neg_root_mean_squared_error)
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="neg_root_mean_squared_error",
                cv=3,
                n_jobs=1,  # <--- run single-threaded to avoid pickling issues
            )

            # fit grid-search on training data
            gs.fit(X_train, y_train)

            # Set best params on the original estimator and fit
            best_params = gs.best_params_
            model.set_params(**best_params)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = _rmse(y_train, y_train_pred)
            test_rmse = _rmse(y_test, y_test_pred)

            report[name] = {
                "train_r2": float(train_r2),
                "train_rmse": float(train_rmse),
                "test_r2": float(test_r2),
                "test_rmse": float(test_rmse),
                "best_params": best_params,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
