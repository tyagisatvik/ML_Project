import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
        train_data_path: str=os.path.join("artifacts", "train.csv")
        test_data_path: str=os.path.join("artifacts", "test.csv")
        raw_data_path: str=os.path.join("artifacts", "raw.csv")

class DataIngestion:
        def __init__(self):
                self.ingestion_config=DataIngestionConfig()
        
        def initiate_data_ingestion(self):
                logging.info("Entered the data ingestion method or component")
                try:
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        csv_path = os.path.join(project_root, 'notebook', 'data', 'stud.csv')
                        df=pd.read_csv(csv_path)
                        logging.info('Read the dataset as dataframe')

                        # Create the 'artifacts' directory if it does not already exist.
                        # This directory will store the output files (raw, train, test).
                        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

                        # Save the original, raw DataFrame to a new CSV file.
                        df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)

                        logging.info("Train test split initiated")
                        # Split the DataFrame into training and testing sets.
                        # 80% of the data will be for training, 20% for testing.
                        # random_state ensures reproducibility of the split.
                        train_set, test_set= train_test_split(df, test_size=0.2, random_state=42)

                        # Save the training set to a CSV file.
                        train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

                        # Save the test set to a CSV file.
                        test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

                        logging.info("Ingestion of the data is completed")
                        
                        # Return the paths of the newly created train, test, and raw data files.
                        return (
                                        self.ingestion_config.train_data_path,
                                        self.ingestion_config.test_data_path,
                                        self.ingestion_config.raw_data_path
                        )
                  
                except Exception as e: 
                        raise CustomException(e, sys)
                        
                

if __name__=="__main__":
        obj=DataIngestion()
        obj.initiate_data_ingestion()