'''
Used for model training
'''

import os
import sys
import warnings
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exceptions import CustomException
from src.utilities import save_object

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class ModelTrainerConfig:
    parent: str = 'artifacts'
    model_path: str = os.path.join(parent, 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def __find_best_model(self, models: dict, params: dict, X_train, y_train, X_test, y_test) -> dict[str, float]:
        model_scores: dict[str, float] = {}

        for name, model in models.items():
            best_params: dict = {}
            if params.get(name, None):
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params[name],
                    scoring='r2',
                    n_jobs=-1,
                    refit=True,
                    cv=5,
                    verbose=False
                )
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_

            if best_params:
                model.set_params(**best_params)

            model.fit(X_train, y_train)
            logger.info(f'Training on model')
            y_pred = model.predict(X_test)

            score = r2_score(y_test, y_pred)
            model_scores[name] = score
            logger.info('Saving the model R Square score')

        return model_scores

    def evaluate_best_model(self, train_arr, test_arr) -> float:
        try:
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            logger.info('Test train split')

            models = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(), 
                'SVR': SVR(kernel='rbf'),
                'KNN Regressor': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Adaboost': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Xgboost': XGBRegressor(),
                'Catboost': CatBoostRegressor(verbose=False)
            }

            params = {
                "Linear Regression":{},
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Adaboost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Xgboost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            best_models: dict[str, float] = self.__find_best_model(models, params, X_train, y_train, X_test, y_test)
            best_model = sorted(best_models.items(), key=lambda item: item[1], reverse=True)[0]

            model = models[best_model[0]]
            model.fit(X_train, y_train)
            logger.info('Successfully train the best model')

            y_pred = model.predict(X_test)

            save_object(self.model_trainer_config.model_path, object=model)
            logger.info('Model saved successfully')

            return r2_score(y_test, y_pred)

        except Exception as e:
            raise CustomException(e, sys)