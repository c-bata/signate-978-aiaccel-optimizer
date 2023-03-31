from __future__ import annotations

from typing import Optional, Any

import aiaccel.parameter
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from optuna.study import StudyDirection
import optuna

from switching_sampler import SwitchingSampler, N_STARTUP


class MyOptimizer(AbstractOptimizer):
    def __init__(self, options: dict) -> None:
        """Initial method of TpeOptimizer.

        Args:
            options (dict): A file name of a configuration.
        """
        super().__init__(options)
        self.parameter_pool = {}
        self.parameter_list = []
        self.study_name = "signate-hpo-study"
        self.study = None
        self.trial_pool = {}
        self.randseed = self.config.randseed.get()
        self.optuna_distributions: Optional[OptunaDistribution] = None

    def pre_process(self) -> None:
        """Pre-Procedure before executing optimize processes.
        """
        super().pre_process()

        self.parameter_list = self.params.get_parameter_list()
        self.optuna_distributions = OptunaDistribution(self.params)
        self.create_study()

    def post_process(self) -> None:
        """Post-procedure after executed processes.
        """
        self.check_result()
        super().post_process()

    def check_result(self) -> None:
        """Check the result files and add it to sampler object.

        Returns:
            None
        """

        del_keys = []
        for trial_id, param in self.parameter_pool.items():
            objective = self.storage.result.get_any_trial_objective(trial_id)
            if objective is not None:
                trial = self.trial_pool[trial_id]
                self.study.tell(trial, objective)
                del_keys.append(trial_id)

        for key in del_keys:
            self.parameter_pool.pop(key)
            self.logger.info(f'trial_id {key} is deleted from parameter_pool')

        self.logger.debug(f'current pool {[k for k, v in self.parameter_pool.items()]}')

    def is_startup_trials(self) -> bool:
        """Is a current trial startup trial or not.

        Returns:
            bool: Is a current trial startup trial or not.
        """
        return self.num_of_generated_parameter < N_STARTUP

    def generate_parameter(self, number: Optional[int] = 1) -> Optional[list[dict[str, Any]]]:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.
        """
        self.check_result()
        self.logger.debug(f'number: {number}, pool: {len(self.parameter_pool)} losses')

        # TPE has to be sequential.
        if (
            (not self.is_startup_trials()) and
            (len(self.parameter_pool) >= 1)
        ):
            return None

        if len(self.parameter_pool) >= self.config.num_node.get():
            return None

        new_params = []
        trial = self.study.ask(self.optuna_distributions.normalize_distributions)

        for param in self.params.get_parameter_list():
            new_param = {
                'parameter_name': param.name,
                'type': param.type,
                'value': self.optuna_distributions.norm_to_dist(
                    param.name,
                    trial.params[param.name]
                )
            }
            new_params.append(new_param)
        trial_id = self.trial_id.get()
        self.parameter_pool[trial_id] = new_params
        self.trial_pool[trial_id] = trial
        self.logger.info(f'newly added name: {trial_id} to parameter_pool')

        return new_params

    def generate_initial_parameter(self):

        enqueue_trial = {}
        for hp in self.params.hps.values():
            if hp.initial is not None:
                enqueue_trial[hp.name] = self.optuna_distributions.dist_to_norm(
                    hp.name, hp.initial
                )

        # all hp.initial is None
        if len(enqueue_trial) == 0:
            return self.generate_parameter()

        self.study.enqueue_trial(enqueue_trial)
        trial = self.study.ask(self.optuna_distributions.normalize_distributions)

        new_params = []

        for name, value in trial.params.items():
            new_param = {
                'parameter_name': name,
                'type': self.params.hps[name].type,
                'value': self.optuna_distributions.norm_to_dist(
                    name,
                    trial.params[name]
                )
            }
            new_params.append(new_param)

        trial_id = self.trial_id.get()
        self.parameter_pool[trial_id] = new_params
        self.trial_pool[trial_id] = trial
        self.logger.info(f'newly added name: {trial_id} to parameter_pool')
        return new_params

    def create_study(self) -> None:
        """Create the optuna.study object and store it.

        Returns:
            None
        """
        if self.study is None:
            direction = None
            if self.config.goal.get().lower() == "minimize":
                direction = StudyDirection.MINIMIZE
            elif self.config.goal.get().lower() == "maximize":
                direction = StudyDirection.MAXIMIZE
            assert direction is not None
            assert self.optuna_distributions is not None
            sampler = SwitchingSampler(self.optuna_distributions.normalize_distributions, seed=self.randseed)
            self.study = optuna.create_study(
                sampler=sampler,
                study_name=self.study_name,
                direction=direction
            )


class OptunaDistribution:
    def __init__(self, parameters: aiaccel.parameter.HyperParameterConfiguration) -> None:
        self.distributions = self.create_distributions(parameters)
        self.normalize_distributions = self.create_normalize_distributions(parameters)

    def create_distributions(
        self, parameters: aiaccel.parameter.HyperParameterConfiguration
    ) -> dict:
        """Create an optuna.distributions dictionary for the parameters.

        Args:
            parameters(aiaccel.parameter.HyperParameterConfiguration): A
                parameter configuration object.

        Returns:
            (dict): An optuna.distributions object.
        """
        distributions = {}

        for p in parameters.get_parameter_list():
            if p.type == 'FLOAT':
                distributions[p.name] = optuna.distributions.FloatDistribution(p.lower, p.upper, log=p.log)

            elif p.type == 'INT':
                distributions[p.name] = optuna.distributions.IntDistribution(p.lower, p.upper, log=p.log)
            else:
                raise 'Unsupported parameter type'

        return distributions

    def create_normalize_distributions(
        self, parameters: aiaccel.parameter.HyperParameterConfiguration
    ) -> dict:
        distributions = {}

        for p in parameters.get_parameter_list():
            if p.type == 'FLOAT':
                distributions[p.name] = optuna.distributions.FloatDistribution(0.0, 1.0, log=p.log)
            elif p.type == 'INT':
                distributions[p.name] = optuna.distributions.FloatDistribution(0.0, 1.0, log=p.log)
            else:
                raise 'Unsupported parameter type'

        return distributions

    # a-b -> 0-1
    def dist_to_norm(self, param_name: str, value: float):
        low = self.distributions[param_name].low
        high = self.distributions[param_name].high
        return (value - low) / (high - low)

    # 0-1 -> a-b
    def norm_to_dist(self, param_name: str, value: float):
        low = self.distributions[param_name].low
        high = self.distributions[param_name].high
        return value * (high - low) + low
