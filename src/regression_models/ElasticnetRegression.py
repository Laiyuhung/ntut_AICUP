import joblib
from sklearn.linear_model import ElasticNet
import os

def elasticnet_regression_modal(NowDateTime, X_train, y_train):
    # Initialize and train the ElasticNet model
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X_train, y_train)
    
    # Ensure the 'models' directory exists
    os.makedirs('./models', exist_ok=True)
    
    # Save the ElasticNet model using joblib with a .joblib extension
    model_filename = f'./models/ElasticNetRegression.joblib'
    joblib.dump(model, model_filename)
    
    print(f"ElasticNet model saved.")

