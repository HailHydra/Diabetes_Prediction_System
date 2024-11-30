import modin.pandas as pd
import ray
ray.shutdown()  # Properly shut down the Ray runtime before re-initializing
ray.init()      # Now you can call init again

import os
import sys
from src.exceptions.exceptions import customexception
from src.logger.logging import logging
from src.utility.utility import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Paths to artifacts (ensure these paths are correct)
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'xgboost_model.pkl')

            # Load preprocessor and model
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Check the loaded preprocessor and model
            print(f"Preprocessor loaded: {preprocessor}")
            print(f"Model loaded: {model}")

            # Preprocess the input data (ensure it matches training data format)
            scale_feat = preprocessor.transform(features)
            print(f"Processed features: {scale_feat}")

            # Use the trained model to make predictions
            pred = model.predict(scale_feat)

            # Debugging the prediction result
            print(f"Prediction result: {pred}")

            return pred

        except Exception as e:
            print(f"Error in prediction pipeline: {e}")
            raise customexception(e, sys)


class CustomData:
    def __init__(self,
                 Pregnancies: int,
                 Glucose: float,
                 BloodPressure: float,
                 SkinThickness: float,
                 Insulin: float,
                 BMI: float,
                 DiabetesPedigreeFunction: float,
                 Age: int):
        
        self.Pregnancies = Pregnancies
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age

    def get_data_as_dataframe(self):
        try:
            # Gather the input data into a dictionary
            custom_data_input_dict = {
                'Pregnancies': [self.Pregnancies],
                'Glucose': [self.Glucose],
                'BloodPressure': [self.BloodPressure],
                'SkinThickness': [self.SkinThickness],
                'Insulin': [self.Insulin],
                'BMI': [self.BMI],
                'DiabetesPedigreeFunction': [self.DiabetesPedigreeFunction],
                'Age': [self.Age]
            }
            # Convert dictionary to DataFrame
            df = pd.DataFrame(custom_data_input_dict)

            # Log and return the dataframe
            logging.info('Dataframe Gathered')
            print(f"Dataframe: {df}")
            return df

        except Exception as e:
            logging.error('Exception occurred while creating the dataframe')
            print(f"Error in dataframe creation: {e}")
            raise customexception(e, sys)