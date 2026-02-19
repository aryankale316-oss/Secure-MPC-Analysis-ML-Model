import numpy as np
import pandas as pd
from config.config import PROCESSED_DATA_PATHS

def split_into_clients(data: pd.DataFrame, num_clients=3):

    shuffled = data.sample(frac=1).reset_index(drop=True)

    splits = np.array_split(shuffled, num_clients)

    for i, split in enumerate(splits):
        split.to_csv(PROCESSED_DATA_PATHS[i], index=False)

    print("Data split into hospitals successfully.")
