import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_model(model, data_path):

    data = pd.read_csv(data_path)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    return accuracy
