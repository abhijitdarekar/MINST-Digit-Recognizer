import os 
import sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.training.train import Model

def main():
    try:
        logging.info("MAIN : Data Ingestion")
        data_ingestion = DataIngestion()
        data_ingestion._initate_data_ingestion()

        logging.info("MAIN : Data Transformation")
        data_transformation = DataTransformation()
        train_loader, test_loader = data_transformation._initate_data_transformation()

        logging.info("MAIN : Model Training")
        model_training = Model(train_loader,test_loader)
        model_training.initiate_traning()

        print("Model Training Completed")
        logging.info("MODEL : Model Training Completed.")
    except Exception as e:
        logging.info(f"Caught Exception : {e}")
        raise CustomException(e,sys)


if __name__=="__main__":
    main()



