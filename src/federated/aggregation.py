import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def aggregate_models(models):

    # average coefficients
    avg_coef = np.mean([m.coef_ for m in models], axis=0)

    # average intercepts
    avg_intercept = np.mean([m.intercept_ for m in models], axis=0)

    # create dummy model and fit once to initialize properly
    dummy_X = np.zeros((2, models[0].coef_.shape[1]))
    dummy_y = np.array([0, 1])

    global_model = LogisticRegression(max_iter=1000)

    global_model.fit(dummy_X, dummy_y)

    # now safely overwrite weights
    global_model.coef_ = avg_coef
    global_model.intercept_ = avg_intercept
    global_model.classes_ = models[0].classes_

    return global_model
