import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    """
    Save an object to a file using numpy's save function.
    
    Args:
        file_path (str): The path where the object will be saved.
        obj: The object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise Exception(f"Error saving object: {e}") from e
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """    
    Evaluate multiple regression models and return their R2 scores.

    Args:
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training target variable.
        X_test (np.ndarray): Testing feature set.
        y_test (np.ndarray): Testing target variable.
        models (dict): Dictionary of model names and their corresponding model instances.

    Returns:
        dict: A dictionary containing model names as keys and their R2 scores as values.
    """
    try:
        model_report = {}
        for model_name, model in models.items():
            # Fit the model
            model.fit(X_train, y_train)
            # Predict on test data
            y_pred = model.predict(X_test)
            # Calculate R2 score
            r2_square = r2_score(y_test, y_pred)
            model_report[model_name] = r2_square
        return model_report
    
    except Exception as e:
        raise CustomException(e,sys)
    