from xgboost import XGBRegressor
import joblib
import os

def xgboost_regression_modal(NowDateTime, X_train, y_train):
    # Initialize and train the XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # Ensure the 'models' directory exists
    os.makedirs('./models', exist_ok=True)
    
    # Save the XGBoost model using joblib with a .joblib extension
    model_filename = f'./models/XGBoostRegression.joblib'
    joblib.dump(model, model_filename)
    
    print(f"XGBoost model saved.")
