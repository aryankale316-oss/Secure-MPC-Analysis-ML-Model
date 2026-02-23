import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_local_model(df: pd.DataFrame):
    """Train a scaled local logistic regression model and return scaler + parameters."""
    if "target" not in df.columns:
        raise ValueError("Input dataframe must contain a 'target' column.")

    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_scaled, y)

    return scaler, model.coef_, model.intercept_
