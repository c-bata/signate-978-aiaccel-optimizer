import optuna
import glob

import optuna_dashboard
import pandas as pd
from optuna import create_trial
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState


search_space = {
    "lr": FloatDistribution(1.0e-4, 1.0),
    "weight_lr_ratios": FloatDistribution(1.0e-6, 1.0e-3),
    "batch_size": IntDistribution(64, 256),
    "momentum": FloatDistribution(0.8, 1.0),
}


def main():
    storage = optuna.storages.InMemoryStorage()
    files = glob.glob("./reports/*.csv")
    print(files)
    all_study = optuna.create_study(study_name="all", storage=storage)
    for f in files:
        study = optuna.create_study(study_name=f, storage=storage)
        df = pd.read_csv(f)
        trials = []

        n_trials = len(df)
        for i in range(n_trials):
            params = {param_name: df[param_name][i] for param_name in search_space}
            value = float(df["objective"][i])
            user_attrs = {
                "rank": df['objective'][:i+1].rank()[i]
            }

            trial = create_trial(
                state=TrialState.COMPLETE,
                params=params,
                value=value,
                distributions=search_space,
                user_attrs=user_attrs,
            )
            trials.append(trial)
        study.add_trials(trials)
        all_study.add_trials(trials)
    optuna_dashboard.run_server(storage)


if __name__ == '__main__':
    main()
