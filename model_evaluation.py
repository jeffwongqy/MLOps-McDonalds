import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
import joblib

def load_preprocessed_data(filepath):
    return joblib.load(filepath)

def load_model(filepath):
    return joblib.load(filepath)

def model_evaluation(model, X_train, X_test, y_train, y_test):
    predict_y_train = model.predict(X_train)
    r2Score_train = r2_score(y_train, predict_y_train)
    mae_train = mean_absolute_error(y_train, predict_y_train)
    mse_train = mean_squared_error(y_train, predict_y_train)
    rmse_train = root_mean_squared_error(y_train, predict_y_train)

    predict_y_test = model.predict(X_test)
    r2Score_test = r2_score(y_test, predict_y_test)
    mae_test = mean_absolute_error(y_test, predict_y_test)
    mse_test = mean_squared_error(y_test, predict_y_test)
    rmse_test = root_mean_squared_error(y_test, predict_y_test)

    results_train = {'R2': r2Score_train, 'MAE': mae_train, 'MSE': mse_train, 'RMSE': rmse_train}
    results_test = {'R2': r2Score_test, 'MAE': mae_test, 'MSE': mse_test, 'RMSE': rmse_test}

    display_results = pd.DataFrame({'Train': results_train, 'Test': results_test})

    return display_results

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_preprocessed_data("preprocessed_data.pkl")

    model_dtr = load_model("model_dtr.pkl")
    results_dtr = model_evaluation(model_dtr, X_train, X_test, y_train, y_test)
    print("Decision Tree: ")
    print(results_dtr)
    

    model_rfr = load_model("model_rfr.pkl")
    results_rfr = model_evaluation(model_rfr, X_train, X_test, y_train, y_test)
    print("\nRandom Forest: ")
    print(results_rfr)
    

    model_gbr = load_model("model_gbr.pkl")
    results_gbr = model_evaluation(model_gbr, X_train, X_test, y_train, y_test)
    print("\nGradient Boosting: ")
    print(results_gbr)

    model_etr = load_model("model_etr.pkl")
    results_etr = model_evaluation(model_etr, X_train, X_test, y_train, y_test)
    print("\nExtra Trees: ")
    print(results_etr)