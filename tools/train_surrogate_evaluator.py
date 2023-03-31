from __future__ import annotations

import hashlib
import pickle
import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


param_names = ["lr", "weight_lr_ratios", "batch_size", "momentum"]
model_path = "model.pkl"


def main():
    files = glob.glob("./reports/*.csv")
    trials: dict[str, tuple[list[float], float]] = {}
    n_duplicated_suggestions = 0
    for f in files:
        df = pd.read_csv(f)

        n_trials = len(df)
        for i in range(n_trials):
            params = [df[param_name][i] for param_name in param_names]
            value = float(df["objective"][i])
            key = hashlib.md5(f"{params}:{value}".encode('utf-8')).hexdigest()
            if key in trials:
                assert params == trials[key][0]
                assert value == trials[key][1]
                n_duplicated_suggestions += 1
            else:
                trials[key] = (params, value)

    print("Trials")
    print(len(trials), n_duplicated_suggestions)

    X = np.empty(shape=(len(trials), len(param_names)))
    y = np.empty(shape=len(trials))
    for i, key in enumerate(trials):
        t = trials[key]
        X[i] = t[0]
        y[i] = t[1]

    print(f"Train RandomForestRegressor and dump to {model_path}")
    regr = RandomForestRegressor(random_state=1)
    regr.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(regr, f)

    check_model()


def check_model():
    print("Check pickle object")
    with open(model_path, "rb") as f:
        regr_loaded = pickle.load(f)

    predicted = regr_loaded.predict([[8.28097061e-02, 1.00000000e-03, 1.48000000e+02, 8.77722145e-01]])
    print(predicted)


if __name__ == '__main__':
    main()
