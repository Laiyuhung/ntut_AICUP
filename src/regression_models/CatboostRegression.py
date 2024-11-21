import joblib
from catboost import CatBoostRegressor
import os

def catboost_regression_modal(NowDateTime, X_train, y_train):
    # Initialize and train the CatBoost model
    model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0)
    model.fit(X_train, y_train)
    
    # Ensure the 'models' directory exists
    os.makedirs('./models', exist_ok=True)
    
    # Save the CatBoost model using joblib with a .joblib extension
    model_filename = f'./models/CatBoostRegression.joblib'
    joblib.dump(model, model_filename)
    
    print(f"CatBoost model saved.")
