import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ADD_ON_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]

TENURE_BINS = [-1, 6, 12, 24, 48, 60, np.inf]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-60", "61-72"]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # blank TotalCharges only occurs for tenure=0 customers with no billing history yet
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)
        X["tenure_group"] = pd.cut(X["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
        X["num_add_on_services"] = (X[ADD_ON_COLUMNS] == "Yes").sum(axis=1)
        return X
