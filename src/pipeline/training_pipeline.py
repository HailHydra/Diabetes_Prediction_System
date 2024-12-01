import sys
from src.logger.logging import logging
from src.exceptions.exceptions import customexception
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

class TrainingPipeline:
    
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()
        
    def start_training(self):
        try:
            # Data Ingestion
            logging.info("Starting data ingestion...")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()

            # Data Transformation (Scaling, Encoding)
            logging.info("Starting data transformation...")
            train_arr, test_arr = self.data_transformation.initialize_data_transformation(train_data_path, test_data_path)

            # Model Training (Using XGBoost)
            logging.info("Starting model training...")
            self.model_trainer.initiate_model_training(train_arr, test_arr)

            # Model Evaluation
            logging.info("Starting model evaluation...")
            self.model_evaluation.initiate_model_evaluation(train_arr, test_arr)

        except Exception as e:
            logging.error(f"Error in the training pipeline: {str(e)}")
            raise customexception(e, sys)