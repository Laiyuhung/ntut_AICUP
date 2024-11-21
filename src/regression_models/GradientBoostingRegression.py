from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime
import joblib
import os

def create_modal(Regression_X_train, Regression_y_train):
    # Initialize and train the GradientBoosting model
    GradientBoostingModel = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    GradientBoostingModel.fit(Regression_X_train, Regression_y_train)

    return GradientBoostingModel

def gradient_boosting_regression_modal(NowDateTime, Regression_X_train, Regression_y_train):
    # Create the GradientBoosting model
    RegressionModel = create_modal(Regression_X_train, Regression_y_train)
    
    # Ensure the 'models' directory exists
    os.makedirs('./models', exist_ok=True)
    
    # Save the GradientBoosting model using joblib with a .joblib extension
    model_path = f'./models/GradientBoostingRegression.joblib'
    joblib.dump(RegressionModel, model_path)

    print(f'Model Saved: {model_path}')
    
    # Output the R squared score
    print('Gradient Boosting R squared:', RegressionModel.score(Regression_X_train, Regression_y_train))
