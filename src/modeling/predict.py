import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class FootballModel:
    """Predicts winner, spread coverage, and team totals."""
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
