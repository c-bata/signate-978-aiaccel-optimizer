import os.path
import pickle

import numpy as np
from aiaccel.util import aiaccel


param_names = ["lr", "weight_lr_ratios", "batch_size", "momentum"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main(p):
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    x = np.array([p[n] for n in param_names])
    y = model.predict([x])
    return y[0]


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
