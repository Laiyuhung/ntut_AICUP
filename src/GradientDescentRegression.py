import numpy as np
from joblib import dump
import os

def gradient_descent_regression_modal(timestamp, AllOutPut, X_train, y_train, learning_rate=0.01, max_iters=1000, tolerance=1e-6):
    
    # Ensure X_train is a 2D array
    if len(X_train.shape) != 2:
        raise ValueError("X_train should be a 2D array with shape (num_samples, num_features).")
    
    # Initialize weights and bias
    num_features = X_train.shape[1]
    weights = np.zeros(num_features)
    bias = 0
    
    # Training loop
    for i in range(max_iters):
        # Compute predictions
        y_pred = np.dot(X_train, weights) + bias
        
        # Compute gradient
        error = y_pred - y_train
        gradient_weights = (2 / X_train.shape[0]) * np.dot(X_train.T, error)
        gradient_bias = (2 / X_train.shape[0]) * np.sum(error)
        
        # Diagnostic print for shapes
        print(f"Iteration {i+1}:")
        print(f"X_train shape: {X_train.shape}")
        print(f"weights shape: {weights.shape}")
        print(f"gradient_weights shape: {gradient_weights.shape}")
        
        # Update weights and bias
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
        
        # Check convergence
        if np.linalg.norm(gradient_weights) < tolerance and abs(gradient_bias) < tolerance:
            print(f"Gradient Descent converged at iteration {i+1}")
            break

    # Save the trained model
    model_filename = f'./models/GradientDescentRegression_{timestamp}.joblib'
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    dump((weights, bias), model_filename)
    print(f"Gradient Descent Regression model saved to {model_filename}")
