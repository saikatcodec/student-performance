'''
Transforms all the data and prepare for the model training
'''

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exceptions import CustomException
from src import utilities

logger = logging.getLogger(__name__)


@dataclass
class DataTransformerConfig:
    parent: str = 'artifacts'
    preprocessor_file_path = os.path.join(parent, 'preprocessor.pkl')


class DataTransformer:
    def __init__(self):
        self.data_transformer_config = DataTransformerConfig()

    def __transformer_pipeline(self):
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            logger.info(f'Numerical features are: {numerical_features}')
            logger.info(f'Categorical features are: {categorical_features}')

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot', OneHotEncoder(drop='first')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical', num_pipeline, numerical_features),
                    ('categorical', cat_pipeline, categorical_features)
                ],
                remainder='drop'
            )
            logger.info('Preprocessor is initiated for numerical and categorical features')

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def prepare_dataset(self, train_set_path, test_set_path):
        try:
            train_set = pd.read_csv(train_set_path)
            test_set = pd.read_csv(test_set_path)
            logger.info('Successfully read the train/test dataset')

            train_input_features = train_set.drop(columns=['math_score'])
            train_target_features = train_set['math_score']

            test_input_features = test_set.drop(columns=['math_score'])
            test_target_features = test_set['math_score']
            logger.info('dataset divided into depended and independent features')

            preprocessor = self.__transformer_pipeline()
            logger.info('Preprocessor object created')

            train_input_arr = preprocessor.fit_transform(train_input_features)
            test_input_arr = preprocessor.transform(test_input_features)
            logger.info('Performs the preprocessing on train and test data')

            train_arr = np.c_[np.array(train_input_arr), np.array(train_target_features)]
            test_arr = np.c_[np.array(test_input_arr), np.array(test_target_features)]
            logger.info('Concatenate the target and feature into array')

            utilities.save_object(
                file_path=self.data_transformer_config.preprocessor_file_path,
                object=preprocessor
            )
            logger.info('Successfully saved the preprocessor')

            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
