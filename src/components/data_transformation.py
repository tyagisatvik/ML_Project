from dataclasses import dataclass
import os
from src.logger import logging
import sys
from src.exception import CustomException
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.utils import save_object

from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer


## dataclass is used to store configuration settings in a structured way.
@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    """
    preprocessor_obj_file_path=os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Class for data transformation.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):

        """
        This function creates a data transformation pipeline.
        
        It combines preprocessing for numerical and categorical features using a ColumnTransformer.
        Numerical features are imputed with the median and scaled.
        Categorical features are imputed with the most frequent value, one-hot encoded, and scaled.
        
        Raises:
            CustomException: If an error occurs during the process.
        
        Returns:
            preprocessor: A ColumnTransformer object for data preprocessing.
        """
           
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            ## Create the numerical pipeline
            num_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), # robust to outliers
                    ("scaler", StandardScaler()), # calculate mean and std deviation for each column and use it to scale the data
                ]
            )

            ## Create the categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), # fill missing values with most frequent value
                    ("one_hot_encoder", OneHotEncoder()), # convert categorical variables into numerical variables
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("Numerical and categorical pipelines created successfully.")

            ## Combine the numerical and categorical pipelines into a single preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            logging.info("Preprocessor created successfully.")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        
        try:            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully.")

            logging.info("Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target column name
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate the training dataframe into input features and the target feature
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate the test dataframe into input features and the target feature
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")
            
            # Fit and transform the training data, and transform the test data
            # 1. Learn from the training data and transform it
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            
            # 2. Apply the SAME transformation rules to the test data
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            # 3. Combine the processed features with the target variable
            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                # Argument 1: WHERE to save the file
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                
                # Argument 2: WHAT to save
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    # Step 1: Data Ingestion
    data_ingestion = DataIngestion()
    train_data_path, test_data_path, _ = data_ingestion.initiate_data_ingestion()

    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    logging.info("Data transformation script executed successfully.")

    # Step 3: Model Training
    model_trainer = ModelTrainer()
    r2_square = model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)
    logging.info(f"Model training completed. Best model R2 score: {r2_square}")






