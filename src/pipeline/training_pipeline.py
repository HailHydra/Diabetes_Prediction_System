import sys
from src.logger.logging import logging
from src.exceptions.exceptions import customexception
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

try:
    # Data Ingestion
    obj = DataIngestion()

    # Initiating data ingestion to load training and testing data
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    # Data Transformation (Scaling, Encoding)
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)

    # Model Training (Using XGBoost)
    model_trainer_obj = ModelTrainer()
    model_trainer_obj.initiate_model_training(train_arr, test_arr)

    # Model Evaluation
    model_eval_obj = ModelEvaluation()
    model_eval_obj.initiate_model_evaluation(train_arr, test_arr)

except Exception as e:
    logging.error(f"Error in the training pipeline: {str(e)}")
    raise customexception(e, sys)