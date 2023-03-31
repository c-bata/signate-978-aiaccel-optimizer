"""
A sampler using BoTorch.
Code were taken from Optuna (MIT License) and modified to include following changes:

* Add trust-region support.
* Add GIBBON, LogEI, qNoisyEI candidates functions.
* Add Botorch>=0.8.0 support and remove the use of deprecated functions.
* Simplify the source code by removing constraints support that is not required in this competition.

Link:
* https://github.com/optuna/optuna/blob/v3.1.0/LICENSE
* https://github.com/optuna/optuna/blob/v3.1.0/optuna/integration/botorch.py.
"""

from __future__ import annotations
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import numpy
import optuna

from optuna import logging
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution, FloatDistribution
from optuna.samplers import BaseSampler, IntersectionSearchSpace
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import manual_seed
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.sampling.normal import SobolQMCNormalSampler
import torch


_logger = logging.get_logger(__name__)


def qei_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qEI.")
    train_y = train_obj
    best_f = train_obj.max()
    train_x = normalize(train_x, bounds=bounds)
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    acqf = qExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size((256,))),
    )
    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=100,
        raw_samples=1024,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    candidates = unnormalize(candidates.detach(), bounds=bounds)
    return candidates


def qei_noisy_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qEI.")
    train_y = train_obj
    train_x = normalize(train_x, bounds=bounds)
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    acqf = qNoisyExpectedImprovement(
        model,
        train_x,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size((256,))),
    )
    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    candidates = unnormalize(candidates.detach(), bounds=bounds)
    return candidates


def logei_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    train_y = train_obj
    best_f = train_obj.max()
    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    acqf = LogExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size((256,))),
        objective=None,
    )
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=100,
        raw_samples=1024,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    return candidates


def gibbon_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qEI.")

    train_y = train_obj
    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    candidate_set = torch.rand(
        1000,
        bounds.size(1),
        device=bounds.device,
        dtype=bounds.dtype
    )
    qGIBBON = qLowerBoundMaxValueEntropy(model, candidate_set)
    candidates, _ = optimize_acqf(
        acq_function=qGIBBON,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
    )
    return candidates


class BoTorchSampler(BaseSampler):
    def __init__(
        self,
        *,
        n_startup_trials: int = 10,
        seed: Optional[int] = None,
        candidates_func = None
    ):
        self._search_space = IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        self._seed = seed
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._candidates_func = candidates_func or qei_candidates_func
        self._trust_region: Optional[dict[str, BaseDistribution]] = None

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> Dict[str, BaseDistribution]:
        search_space: Dict[str, BaseDistribution] = OrderedDict()
        for name, distribution in self._search_space.calculate(study, ordered_dict=True).items():
            if distribution.single():
                # built-in `candidates_func` cannot handle distributions that contain just a
                # single value, so we skip them. Note that the parameter values for such
                # distributions are sampled in `Trial`.
                continue
            search_space[name] = distribution

        return search_space

    def set_trust_region(self, trust_region: Optional[dict[str, BaseDistribution]]) -> None:
        self._trust_region = trust_region

    def get_trials(self, study: Study) -> list[FrozenTrial]:
        trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if self._trust_region is None:
            return trials

        filtered_trials = []
        for t in trials:
            is_feasible = True
            for param_name, distribution in self._trust_region.items():
                assert isinstance(distribution, FloatDistribution)
                param_value = t.params[param_name]
                if not distribution._contains(param_value):
                    is_feasible = False
                    break
            if is_feasible:
                filtered_trials.append(t)
        return filtered_trials

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        assert isinstance(search_space, OrderedDict)
        assert len(study.directions) == 1

        if trial.number < self._n_startup_trials:
            return {}
        if len(search_space) == 0:
            return {}

        if self._trust_region:
            search_space = self._trust_region
        trans = _SearchSpaceTransform(search_space)

        trials = self.get_trials(study)
        n_trials = len(trials)

        values: Union[numpy.ndarray, torch.Tensor] = numpy.empty((n_trials, 1), dtype=numpy.float64)
        params: Union[numpy.ndarray, torch.Tensor] = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
        bounds: Union[numpy.ndarray, torch.Tensor] = trans.bounds

        for trial_idx, trial in enumerate(trials):
            params[trial_idx] = trans.transform(trial.params)
            assert len(study.directions) == len(trial.values)

            for obj_idx, (direction, value) in enumerate(zip(study.directions, trial.values)):
                assert value is not None
                if direction == StudyDirection.MINIMIZE:  # BoTorch always assumes maximization.
                    value *= -1
                values[trial_idx, obj_idx] = value

        values = torch.from_numpy(values).to(self._device)
        params = torch.from_numpy(params).to(self._device)
        bounds = torch.from_numpy(bounds).to(self._device)

        bounds.transpose_(0, 1)

        with manual_seed(self._seed):
            # `manual_seed` makes the default candidates functions reproducible.
            # `SobolQMCNormalSampler`'s constructor has a `seed` argument, but its behavior is
            # deterministic when the BoTorch's seed is fixed.
            candidates = self._candidates_func(params, values, bounds)
            if self._seed is not None:
                self._seed += 1

        if not isinstance(candidates, torch.Tensor):
            raise TypeError("Candidates must be a torch.Tensor.")
        if candidates.dim() == 2:
            if candidates.size(0) != 1:
                raise ValueError(
                    "Candidates batch optimization is not supported and the first dimension must "
                    "have size 1 if candidates is a two-dimensional tensor. Actual: "
                    f"{candidates.size()}."
                )
            # Batch size is one. Get rid of the batch dimension.
            candidates = candidates.squeeze(0)
        if candidates.dim() != 1:
            raise ValueError("Candidates must be one or two-dimensional.")
        if candidates.size(0) != bounds.size(1):
            raise ValueError(
                "Candidates size must match with the given bounds. Actual candidates: "
                f"{candidates.size(0)}, bounds: {bounds.size(1)}."
            )

        return trans.untransform(candidates.cpu().numpy())

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        if self._seed is not None:
            self._seed = numpy.random.RandomState().randint(2**60)
