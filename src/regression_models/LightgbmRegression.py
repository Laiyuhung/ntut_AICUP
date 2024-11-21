import joblib
from lightgbm import LGBMRegressor
import os

def lightgbm_regression_modal(NowDateTime, X_train, y_train):
    # Initialize and train the LightGBM model
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # Ensure the 'models' directory exists
    os.makedirs('./models', exist_ok=True)
    
    # Save the LightGBM model using joblib with a .joblib extension
    model_filename = f'./models/LightGBMRegression.joblib'
    joblib.dump(model, model_filename)
    
    print(f"LightGBM model saved.")
