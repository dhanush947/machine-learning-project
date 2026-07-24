# Basic Import
import numpy as np
import pandas as pd

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from src.utils import print_evaluated_results
from src.utils import model_metrics

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            xtrain, ytrain, xtest, ytest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(max_iter=10000),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=1),
                "XGBRegressor": XGBRegressor(n_estimators=100, random_state=42, n_jobs=1), 
                "CatBoosting Regressor": CatBoostRegressor(iterations=100, depth=4, learning_rate=0.1, thread_count=1,verbose=False),
                "GradientBoosting Regressor":GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(xtrain,ytrain,xtest,ytest,models)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                logging.info('Best model has r2 Score less than 60%')
                raise CustomException('No Best Model Found')
            
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            logging.info('Hyperparameter tuning started for catboost')

            # Hyperparameter tuning on Catboost
            # Initializing catboost
            cbr = CatBoostRegressor(iterations=100,depth=4,learning_rate=0.1,thread_count=1,verbose=False)
            
            # Fit the model
            cbr.fit(xtrain, ytrain)

            # Print the tuned parameters and score
            print(f'Best Catboost parameters : {cbr.get_params()}')
            print(f'Best Catboost Score : {cbr.score(xtest, ytest)}')
            print('\n====================================================================================\n')

            logging.info('Hyperparameter tuning complete for Catboost')

            logging.info('Hyperparameter tuning started for KNN')

            # Initialize knn
            knn = KNeighborsRegressor()

            # parameters
            k_range = [3, 5, 7, 9, 11, 13, 15]
            param_grid = dict(n_neighbors=k_range)

            # Fitting the cvmodel
            grid = GridSearchCV(knn, param_grid, cv=3, scoring='r2',n_jobs=1)
            grid.fit(xtrain, ytrain)

            # Print the tuned parameters and score
            print(f'Best KNN Parameters : {grid.best_params_}')
            print(f'Best KNN Score : {grid.best_score_}')
            print('\n====================================================================================\n')

            best_knn = grid.best_estimator_

            logging.info('Hyperparameter tuning Complete for KNN')

            logging.info('Voting Regressor model training started')

            # Creating final Voting regressor
            er = VotingRegressor([('cbr',cbr),('xgb',XGBRegressor()),('knn',best_knn)], weights=[3,2,1])
            er.fit(xtrain, ytrain)
            print('Final Model Evaluation :\n')
            print_evaluated_results(xtrain,ytrain,xtest,ytest,er)
            logging.info('Voting Regressor Training Completed')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = er
            )
            logging.info('Model pickle file saved')
            # Evaluating Ensemble Regressor (Voting Classifier on test data)
            ytest_pred = er.predict(xtest)

            mae, rmse, r2 = model_metrics(ytest, ytest_pred)
            logging.info(f'Test MAE : {mae}')
            logging.info(f'Test RMSE : {rmse}')
            logging.info(f'Test R2 Score : {r2}')
            logging.info('Final Model Training Completed')
            
            return mae, rmse, r2
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)



