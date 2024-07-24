import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.config import CONFIG
from src.utils import CustomDataset, get_transforms

from torch.utils.data import DataLoader

import pandas as pd
from dataclasses import dataclass

@dataclass
class DataTransfromationConfig:
    test_data_path : str=CONFIG['TEST_FILE_PATH']
    train_data_path :str = CONFIG['TRAIN_FILE_PATH']

class DataTransformation:
    def __init__(self,):
        self.data_transformation_config = DataTransfromationConfig()

    def _initate_data_transformation(self):
        try:
            train_transform, test_transform = get_transforms()
            logging.info("Loaded train and test transformations")
            
            train_data = pd.read_csv(self.data_transformation_config.train_data_path)
            test_data = pd.read_csv(self.data_transformation_config.test_data_path)
            logging.info("Loaded train and test data from artifacts")

            train_dataset = CustomDataset(train_data,train_transform)
            test_dataset = CustomDataset(test_data,test_transform)
            logging.info("Created Custom Dataset Modules.")
            
            train_loader = DataLoader(train_dataset,batch_size=CONFIG["BATCH_SIZE"],shuffle=True)
            test_loader  = DataLoader(test_dataset,batch_size=CONFIG['BATCH_SIZE'],shuffle=True)
            logging.info("Dataloader for train and test created.")

            return (
                train_loader,
                test_loader
            )


        except Exception as e:
            logging.error(f"Caught Ingestion :{e}")
            raise CustomException(e,sys)

