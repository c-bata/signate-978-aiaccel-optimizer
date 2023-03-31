"""
Custom Optuna sampler for this competition.

* Initialization with Sobol's quasi-random sequences (10 trials).
* Sample by BoTorchSampler with 2 candidate functions, qEI and LogEI (15 trials).
* Calculate the trust region, a subset of search space, against high-fANOVA scored parameters (15 trials).
* Calculate the trust region, a subset of search space, against high-fANOVA scored parameters (10 trials).
* Return the best hyperparameters (50 trials)
"""
from __future__ import annotations

from typing import Optional, Container

import optuna
from optuna.distributions import FloatDistribution, BaseDistribution
from optuna.samplers import BaseSampler, QMCSampler, TPESampler
from optuna.trial import FrozenTrial, TrialState
from optuna.study import StudyDirection, Study
from optuna.importance import get_param_importances, FanovaImportanceEvaluator

from botorch_sampler import BoTorchSampler, qei_candidates_func, logei_candidates_func


N_STARTUP = 10
USE_BEST_FROM = 50
IMPORTANCE_THRESHOLD = 0.2
TRUST_REGION_HALF_WIDTH = 0.1


class SwitchingSampler(BaseSampler):
    def __init__(self, sobol_initial_search_space: dict[str, BaseDistribution], seed=None):
        self._seed = seed
        self._sobol_initial_search_space = sobol_initial_search_space
        self._sobol_sampler = QMCSampler(qmc_type="sobol", seed=seed, scramble=True)
        self._tpe_sampler = TPESampler(seed=seed, n_startup_trials=N_STARTUP)
        self._botorch_sampler = BoTorchSampler(seed=seed, candidates_func=qei_candidates_func, n_startup_trials=N_STARTUP)

        self._candidate_idx = 0
        self._candidates_func_list = [
            qei_candidates_func,
            logei_candidates_func,
        ]

    def infer_relative_search_space(self, study, trial):
        if trial.number < N_STARTUP:
            self._sobol_sampler._initial_search_space = self._sobol_initial_search_space
            return self._sobol_initial_search_space
        else:
            return self._botorch_sampler.infer_relative_search_space(study, trial)

    @property
    def n_candidate_functions(self) -> int:
        return len(self._candidates_func_list)

    def set_next_candidates_func(self) -> None:
        next_candidates_func = self._candidates_func_list[self._candidate_idx]
        self._botorch_sampler._candidates_func = next_candidates_func
        self._candidate_idx = (self._candidate_idx + 1) % self.n_candidate_functions

    def sample_relative(self, study: optuna.Study, trial: FrozenTrial, search_space):
        if trial.number < N_STARTUP:
            return self._sobol_sampler.sample_relative(study, trial, search_space)

        if trial.number < 40 + N_STARTUP:
            self.set_next_candidates_func()
            return self._botorch_sampler.sample_relative(study, trial, search_space)

        self.set_next_candidates_func()
        return self.sample_from_trust_region(study, trial, search_space)

    def sample_from_trust_region(self, study: optuna.Study, trial: FrozenTrial, search_space):
        param_importances = calculate_param_importance(study)

        trust_region = {}
        top_10_percent_trials = get_best_n_trial(study, int(trial.number * 0.1))

        # first stage
        for param_name, importance in param_importances.items():
            if importance < IMPORTANCE_THRESHOLD:
                continue

            distribution = search_space[param_name]
            assert isinstance(distribution, FloatDistribution)
            best_param_value = study.best_params[param_name]

            half_width = TRUST_REGION_HALF_WIDTH
            lower_bound = max(distribution.low, best_param_value - half_width)
            upper_bound = min(distribution.high, best_param_value + half_width)

            top_10_percent_param_values = [t.params[param_name] for t in top_10_percent_trials]
            lower_bound = min(lower_bound, min(top_10_percent_param_values))
            upper_bound = max(upper_bound, max(top_10_percent_param_values))

            trust_region[param_name] = FloatDistribution(low=lower_bound, high=upper_bound)

        # second stage
        if trial.number >= 70:
            top_trials_param_importances = calculate_param_importance(study, n_trials=trial.number // 3)
            for param_name, importance in top_trials_param_importances.items():
                if param_name in trust_region:
                    # Skip first stage params
                    continue

                if importance < IMPORTANCE_THRESHOLD:
                    continue

                distribution = search_space[param_name]
                assert isinstance(distribution, FloatDistribution)
                best_param_value = study.best_params[param_name]

                half_width = TRUST_REGION_HALF_WIDTH
                lower_bound = max(distribution.low, best_param_value - half_width)
                upper_bound = min(distribution.high, best_param_value + half_width)

                top_10_percent_param_values = [t.params[param_name] for t in top_10_percent_trials]
                lower_bound = min(lower_bound, min(top_10_percent_param_values))
                upper_bound = max(upper_bound, max(top_10_percent_param_values))

                trust_region[param_name] = FloatDistribution(low=lower_bound, high=upper_bound)

        self._botorch_sampler.set_trust_region(trust_region)
        relative_params = self._botorch_sampler.sample_relative(study, trial, search_space)
        self._botorch_sampler.set_trust_region(None)
        return relative_params

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._tpe_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )


def is_feasible(params: dict[str, float], importance_constraints: dict[str, FloatDistribution]) -> bool:
    for name, distribution in importance_constraints.items():
        if not distribution._contains(params[name]):
            return False
    return True


def get_best_n_trial(study: Study, n: int) -> list[FrozenTrial]:
    all_trials = study.get_trials(deepcopy=True, states=[TrialState.COMPLETE])
    all_trials = sorted(all_trials, key=lambda t: t.value)
    if study.direction == StudyDirection.MAXIMIZE:
        all_trials = reversed(all_trials)

    assert len(all_trials) >= n
    return all_trials[:n]


class StudyWrapperForImportance(Study):
    def __init__(self, study: Study, return_top_n_trials: int) -> None:
        super().__init__(study_name=study.study_name, storage=study._storage)
        self.__study = study
        self.__return_top_n_trials = return_top_n_trials

    @property
    def trials(self) -> list[FrozenTrial]:
        trials = get_best_n_trial(self.__study, self.__return_top_n_trials)
        trials = sorted(trials, key=lambda t: t.number)
        return trials

    def get_trials(
        self,
        deepcopy: bool = True,
        states: Optional[Container[TrialState]] = None,
    ) -> list[FrozenTrial]:
        trials = get_best_n_trial(self.__study, self.__return_top_n_trials)
        trials = sorted(trials, key=lambda t: t.number)
        return trials


def calculate_param_importance(study: Study, n_trials: int = -1) -> dict[str, float]:
    if n_trials > 0:
        study = StudyWrapperForImportance(study, n_trials)

    try:
        param_importances = get_param_importances(
            study, evaluator=FanovaImportanceEvaluator(seed=1), normalize=True
        )
    except RuntimeError:
        # Handling RuntimeError("Encountered zero total variance in all trees.")
        param_importances = {}
    return param_importances
