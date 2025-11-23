import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def _build_onehot_encoder(self):
        """
        Return an OneHotEncoder instance that works across sklearn versions:
        - new sklearn uses sparse_output
        - older sklearn used sparse
        """
        try:
            # preferred for newer sklearn
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # fallback for older sklearn versions
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def get_data_transformer_object(self, numeric_columns, categorical_columns):
        '''
        This function is responsible for creating and returning a preprocessing object
        (ColumnTransformer) for the supplied numeric and categorical columns.
        '''
        try:
            # numeric pipeline: impute missing values then scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # categorical pipeline: impute then one-hot encode then scale (with_mean=False for sparse-like output)
            ohe = self._build_onehot_encoder()

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", ohe),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Numerical columns: {numeric_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numeric_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                ],
                remainder="drop",
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Parse date column (if present) and create simple time-based features
            for df in (train_df, test_df):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    # create month and dayofyear as simple features (can be extended)
                    df['month'] = df['date'].dt.month
                    df['dayofyear'] = df['date'].dt.dayofyear
                    # drop the original date column
                    df.drop(columns=['date'], inplace=True)

            logging.info("Obtaining preprocessing object")

            # target for regression
            target_column_name = "oil"

            # check target exists
            if target_column_name not in train_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in train data", sys)
            if target_column_name not in test_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in test data", sys)

            # determine numeric and categorical columns automatically (exclude target)
            feature_columns = [c for c in train_df.columns if c != target_column_name]
            categorical_columns = [c for c in feature_columns
                                   if train_df[c].dtype == 'object' or str(train_df[c].dtype).startswith('category')]
            numeric_columns = [c for c in feature_columns if c not in categorical_columns]

            # get preprocessing object
            preprocessing_obj = self.get_data_transformer_object(numeric_columns, categorical_columns)

            # split input and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # fit_transform on train, transform on test
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # concatenate features and target to form arrays returned to model trainer
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
