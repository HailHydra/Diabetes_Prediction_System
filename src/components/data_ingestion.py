import modin.pandas as pd
import ray
ray.shutdown() # Shutdown ray if it was initialized before
ray.init() # Initialize ray for modin.pandas

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.model_selection import train_test_split 
# sklearnex doesnot support train_test_split, so it fallbacks to original Scikit-learn

import os
import sys
from dataclasses import dataclass
from src.logger.logging import logging
from src.exceptions.exceptions import customexception

# @dataclass decorator automatically generates special methods for the class such as __init__().
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Update the path to point to the diabetes dataset
            data = pd.read_csv(r'https://raw.githubusercontent.com/HailHydra/Diabetes_Prediction_System/refs/heads/main/diabetes.csv')
            logging.info("Successfully read the diabetes dataset")

            # Create directories for saving raw and processed data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw dataset saved in artifact folder")

            # Split the dataset into training and testing sets
            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
            logging.info("Train-test split completed")

            # Save training and testing data to respective paths
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Training and testing datasets saved")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise customexception(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()