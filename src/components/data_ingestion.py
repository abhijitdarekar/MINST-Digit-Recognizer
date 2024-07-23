import sys
import os
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from src.config import CONFIG

@dataclass
class DataIngestionConfig:
    data_folder :str = os.path.join("notebooks","data")

    train_data_path :str = CONFIG['TRAIN_FILE_PATH']
    test_data_path :str = CONFIG['TEST_FILE_PATH']

class DataIngestion:
    def __init__(self,):
        self.ingestion_config = DataIngestionConfig()

    def _initate_data_ingestion(self):
        try:
            logging.info("Loading Data for ingestion.")
            train_df = pd.read_csv(os.path.join(self.ingestion_config.data_folder,'train.csv'))

            logging.info("Splitting data to train and test.")
            
            train_df = train_df.sample(frac=1).reset_index(drop=True)

            total_len = len(train_df)
            split_size = CONFIG['TRAIN_SIZE']
            train_size = int(total_len*split_size)
           
            train_data = train_df.iloc[:train_size]
            test_Data = train_df.iloc[train_size:]

            os.makedirs(CONFIG['ARTIFACTS'],exist_ok=True)
            train_data.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )
            test_Data.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )
            logging.info("Saved data to artifacts folder.")
        except Exception as e:
            logging.error(f"Caught Ingestion :{e}")
            raise CustomException(e,sys)
