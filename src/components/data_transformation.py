import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import sys
from dataclasses import dataclass
from src.logger.logging import logging
from src.exceptions.exceptions import customexception
from src.utility.utility import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self):
        try:
            logging.info('Data transformation initiated')

            # All columns except 'Outcome' are numerical
            numerical_cols = [
                'Pregnancies', 'Glucose', 'BloodPressure',
                'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age'
            ]

            logging.info('Pipeline initiated for numerical data')

            # Define a pipeline for numerical preprocessing
            num_pipeline = Pipeline(
                steps=[
                    # Handle missing values by imputing with the median
                    ('imputer', SimpleImputer(strategy='median')),
                    # Standardize numerical features
                    ('scaler', StandardScaler())
                ]
            )

            # Since there are no categorical columns in this dataset, only a numerical pipeline is needed
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info("Exception occurred in initiate_data_transformation")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')

            preprocessing_obj = self.initiate_data_transformation()

            # Target column is 'Outcome', and it is binary (0 or 1)
            target_column_name = 'Outcome'

            # Split features and target for train and test datasets
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply the preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing object on training and testing datasets")

            # Combine processed features with target arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing pickle file saved")

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Exception occurred in initialize_data_transformation")
            raise customexception(e, sys)