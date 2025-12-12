import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib

def load_preprocessed_data(filepath):
    return joblib.load(filepath)

def model_training(model, param_grid, X_train, y_train):
    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    model = GridSearchCV(model, 
                         param_grid, 
                         cv = cv, 
                         scoring = {"neg_mse": "neg_mean_squared_error", 
                                    "r2": "r2"}, 
                         refit = "neg_mse",
                         verbose = 3)
    model.fit(X_train, y_train)

    print(model.best_params_)
    print(-model.best_score_)
    print(-model.cv_results_['mean_test_neg_mse'])
    print()
    return model 

def save_model(model, filepath):
    joblib.dump(model, filepath)


if __name__ == "__main__":
    # load preprocessed data 
    X_train, X_test, y_train, y_test = load_preprocessed_data("preprocessed_data.pkl")

    # for decision tree 
    param_grid_dtr = {'max_depth': [3, 4, 5], 
                      'min_samples_split': [3, 4, 5], 
                      'min_samples_leaf': [3, 4, 5]}
    model_dtr = model_training(DecisionTreeRegressor(random_state = 42), param_grid_dtr, X_train, y_train)

    # for random forest 
    param_grid_rfr = {'n_estimators': [20, 30, 40], 
                      'max_depth': [3, 4, 5], 
                      'min_samples_split': [2, 3, 4, 5], 
                      'min_samples_leaf': [1, 2, 3, 4, 5]}
    model_rfr = model_training(RandomForestRegressor(random_state = 42), param_grid_rfr, X_train, y_train)

    # for gradient boosting 
    param_grid_gbr = {'n_estimators': [20, 30, 40], 
                      'learning_rate': [0.1, 0.01, 0.001, 1], 
                      'max_depth': [3, 4, 5],
                      'min_samples_split': [3, 4, 5], 
                      'min_samples_leaf': [3, 4, 5]}
    model_gbr = model_training(GradientBoostingRegressor(random_state = 42), param_grid_gbr, X_train, y_train)

    # for extra trees
    param_grid_etr = {'n_estimators': [20, 30, 40], 
                      'max_depth': [3, 4, 5], 
                      'min_samples_split': [3, 4, 5], 
                      'min_samples_leaf': [3, 4, 5]}
    model_etr = model_training(ExtraTreesRegressor(random_state = 42), param_grid_etr, X_train, y_train)

    # save model 
    save_model(model_dtr, "model_dtr.pkl")
    save_model(model_rfr, "model_rfr.pkl")
    save_model(model_gbr, "model_gbr.pkl")
    save_model(model_etr, "model_etr.pkl")



