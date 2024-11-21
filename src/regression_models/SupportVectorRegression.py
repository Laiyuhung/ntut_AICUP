from sklearn.svm import SVR
from datetime import datetime
import joblib
import os

def create_modal(Regression_X_train, Regression_y_train):
    # Initialize and train the Support Vector Regressor (SVR) model
    SVRModel = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    SVRModel.fit(Regression_X_train, Regression_y_train)

    return SVRModel

def support_vector_regression_modal(NowDateTime, Regression_X_train, Regression_y_train):
    # Create the SVR model
    RegressionModel = create_modal(Regression_X_train, Regression_y_train)
    
    # Ensure the 'models' directory exists
    os.makedirs('./models', exist_ok=True)
    
    # Save the SVR model using joblib with a .joblib extension
    model_path = f'./models/SupportVectorRegression.joblib'
    joblib.dump(RegressionModel, model_path)

    print(f'Model Saved: {model_path}')
    
    # Output the R squared score
    print('SVR R squared:', RegressionModel.score(Regression_X_train, Regression_y_train))
