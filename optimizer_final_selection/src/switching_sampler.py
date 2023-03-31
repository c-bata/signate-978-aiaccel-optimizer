"""
Custom Optuna sampler for this competition.

* Initialization with Sobol's quasi-random sequences (7 trials).
* Switch 4 samplers per each: BoTorch qEI, Multivariate TPE, BoTorch qEI with Trust Region, and BoTorch GIBBON (43 trials).
* Return the best hyperparameters (50 trials)
"""
from __future__ import annotations

import optuna
from optuna.distributions import FloatDistribution
from optuna.samplers import BaseSampler, QMCSampler, TPESampler
from optuna.trial import FrozenTrial

from botorch_sampler import BoTorchSampler, qei_candidates_func, gibbon_candidates_func


USE_BEST_FROM = 50
N_STARTUP = 8


class SwitchingSampler(BaseSampler):
    def __init__(self, seed=None):
        self._seed = seed
        self._sobol_sampler = QMCSampler(qmc_type="sobol", seed=seed, scramble=True)
        self._mv_tpe_sampler = TPESampler(multivariate=True, seed=seed, n_startup_trials=N_STARTUP)
        self._botorch_sampler_qei = BoTorchSampler(seed=seed, candidates_func=qei_candidates_func, n_startup_trials=N_STARTUP)
        self._botorch_sampler_gibbon = BoTorchSampler(seed=seed, candidates_func=gibbon_candidates_func, n_startup_trials=N_STARTUP)

    def infer_relative_search_space(self, study, trial):
        return self._botorch_sampler_qei.infer_relative_search_space(study, trial)

    def sample_relative(self, study: optuna.Study, trial: FrozenTrial, search_space):
        if trial.number < N_STARTUP:
            return self._sobol_sampler.sample_relative(study, trial, search_space)

        if trial.number < USE_BEST_FROM:
            if trial.number % 4 == 0:
                self._botorch_sampler_qei.set_trust_region(None)
                return self._botorch_sampler_qei.sample_relative(study, trial, search_space)
            elif trial.number % 4 == 1:
                return self._mv_tpe_sampler.sample_relative(study, trial, search_space)
            elif trial.number % 4 == 2:
                return self.sample_from_trust_region(self._botorch_sampler_qei, study, trial, search_space)
            else:
                return self._botorch_sampler_gibbon.sample_relative(study, trial, search_space)
        return study.best_params

    def sample_from_trust_region(self, sampler: BoTorchSampler, study: optuna.Study, trial: FrozenTrial, search_space):
        trust_region = {}
        for param_name, distribution in search_space.items():
            assert isinstance(distribution, FloatDistribution)
            best_param_value = study.best_params[param_name]

            trust_region_width_multiplier = max((USE_BEST_FROM - trial.number) / 100, 0.001)
            half_width = (distribution.high - distribution.low) * trust_region_width_multiplier / 2

            lower_bound = max(distribution.low, best_param_value - half_width)
            upper_bound = min(distribution.high, best_param_value + half_width)
            trust_region[param_name] = FloatDistribution(low=lower_bound, high=upper_bound)
            trust_region[param_name] = FloatDistribution(low=lower_bound, high=upper_bound)

        sampler.set_trust_region(trust_region)
        return sampler.sample_relative(study, trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._mv_tpe_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
