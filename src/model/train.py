import pandas as pd
from src.model.model import create_model

def train_model(data_path):

    data = pd.read_csv(data_path)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = create_model()

    model.fit(X, y)

    return model
