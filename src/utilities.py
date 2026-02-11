import pickle
import os
import sys

from src.exceptions import CustomException 

def save_object(file_path, object):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(object, file)
    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys)