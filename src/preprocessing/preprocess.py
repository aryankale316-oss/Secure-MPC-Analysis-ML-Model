import pandas as pd

def preprocess_data(data: pd.DataFrame):

    # Remove null values
    data = data.dropna()

    # Example normalization (optional)
    # data = (data - data.mean()) / data.std()

    return data
