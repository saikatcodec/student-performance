'''
Read the data from various source and save into the system
'''

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exceptions import CustomException

logger = logging.getLogger(__name__)

@dataclass
class DataIngestionConfig:
    parent_path: str = 'artifacts'
    train_data_path: str = os.path.join(parent_path, 'train_data.csv')
    test_data_path: str = os.path.join(parent_path, 'test_data.csv')
    raw_data_path: str = os.path.join(parent_path, 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            dataset = pd.read_csv('notebooks/data/stud.csv')
            logger.info('Data are retrieve from the source')

            train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=42)
            logger.info('Splitting the dataset into train and test sets')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            dataset.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logger.info('Saved the train, test, and raw datasets successfully')

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
                self.data_ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    data_ingest = DataIngestion()
    data_ingest.initiate_data_ingestion()