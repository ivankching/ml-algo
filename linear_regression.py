import numpy as np

class SimpleLinearRegression:
    """Simple Linear regression with 1 dimension"""
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, x, y):
        """Fit linear regression using ordinary least squares
        
        Args:
            x (array-like): Vector of integers
            y (array-like): Vector of integers
        
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