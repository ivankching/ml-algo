import numpy as np
import warnings

class SimpleLinearRegression:
    """Simple Linear regression with 1 dimension"""
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, x, y):
        """Fit linear regression using ordinary least squares
        
        Args:
            x (array-like): Vector of floats
            y (array-like): Vector of floats
        
        Returns:
            self (object): Fitted estimator
        """
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        numerator = np.sum(x_centered * y_centered)
        denominator = np.sum(np.square(x_centered))

        self.slope = numerator / denominator
        self.intercept = np.mean(y) - (self.slope * np.mean(x))

        return self

    def predict(self, x):
        """Predict y from given x using fitted estimator"""
        if not self.slope or not self.intercept:
            raise Exception("Fit model first")
        return (x * self.slope) + self.intercept
    
class LinearRegression:
    """Multiple Linear Regression with n dimension"""
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        """Fit linear regression using ordinary least squares
        
        Args:
            x (array-like): Matrix of floats -- Dimension: (n_samples, n_features)
            y (array-like): Vector of floats --- Dimension: (n_samples,)
        
        Returns:
            self (object): Fitted estimator
        """
        X = np.hstack((np.ones((np.shape(X)[0], 1)), X))  # Add a bias/intercept to first column
        y = y.T   # Change from row vector to column vector

        try:
            self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
        except:
            warnings.warn("Singular matrix. Using pinv")
            self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y

        return self

    def predict(self, X):
        """Predict y from given X using fitted estimator"""
        if not self.weights:
            raise Exception("Fit model first")
        X = np.hstack(np.ones((np.shape(X)[0], 1)), X)  # Add a bias/intercept to first column
        y = X @ self.weights

        return y.reshape(-1)   # Return back a 1-D array of dimension (n_samples,)
    

class Lasso:
    """Linear Regresion with L1 prior as regularizer (Lasso)"""
    def __init__(self, alpha=1.0, tol=1e-4, max_iter=1000):
        
        self.weights = None
        self.alpha = alpha        # L1 Regularization Term
        self.tol = tol       # Tolerance for optimization
        self.max_iter = max_iter    # Maximum number of iterations

    def fit(self, X, y):
        """Fit linear regression using lasso coordinate descent
        
        Args:
            x (array-like): Matrix of floats -- Dimension: (n_samples, n_features)
            y (array-like): Vector of floats --- Dimension: (n_samples,)
        
        Returns:
            self (object): Fitted estimator
        """
        X = np.hstack((np.ones((np.shape(X)[0], 1)), X))  # Add a bias/intercept to first column