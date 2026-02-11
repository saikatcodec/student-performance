import sys
import pandas as pd
import numpy as np

from src.utilities import load_object

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, dataframe):
        preprocessor_path = 'artifacts/preprocessor.pkl'
        model_path = 'artifacts/model.pkl'

        preprocessor = load_object(file_path=preprocessor_path)
        model = load_object(file_path=model_path)

        processed_data = preprocessor.transform(dataframe)

        results = model.predict(processed_data)
        return results[0]


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ) -> None:
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def create_dataframe(self):
        data_dict = {
            'gender': self.gender,
            'race_ethnicity': self.race_ethnicity,
            'parental_level_of_education': self.parental_level_of_education,
            'lunch': self.lunch,
            'test_preparation_course': self.test_preparation_course,
            'reading_score': self.reading_score,
            'writing_score': self.writing_score
        }

        dataframe = pd.DataFrame(data_dict)
        return dataframe