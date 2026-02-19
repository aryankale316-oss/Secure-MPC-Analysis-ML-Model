import pandas as pd
from config.config import DATA_PATH

def load_dataset():
    data = pd.read_csv(DATA_PATH)
    return data
