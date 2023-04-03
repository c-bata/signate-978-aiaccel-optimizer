# signate-978-aiaccel-optimizer

A solution ([aiaccel](https://github.com/aistairc/aiaccel) optimizer) for the SIGNATE HPO competition.

## Solution

### `optimizer_final_selection`

This folder contains the source code that I selected as the final submission.
It performed worse on the private leaderboard though.
The main idea of this method is as follows:

1. Select a trust region, which is expected to contain promising solutions based on past trial results, and sample from this subspace.
   The trust region is centered around the parameters that produced the best evaluation value and is calculated with a width of `trust_region_width = (upper_bound - lower_bound) * max((50 - trial_number) / 100, 0.01)`.
   The basic idea comes from TuRBO, which samples several parameters from within the trust region and expands (explores) it if the best evaluation value improves, and shrinks (exploits) it if it does not improve.
   We designed this algorithm to focus more on exploitation (shrinking) towards the end of the search when the number of evaluations is fixed (in this competition, it was 100).
2. Use Sobol quasi-random sequences instead of random sampling for initial sampling. The low-discrepancy property of Sobol sequences reduces the bias of initial search points and improves search performance stability.
3. Increase robustness to unknown objective functions by alternating between multiple optimization methods with different strengths for solving different types of problems.

### `optimizer_fanova_trust_region`

This folder contains the source code for an algorithm that did not perform well on the public leaderboard, but showed promising results on local benchmarking.

This method narrows down the trust region of high-scoring hyperparameters according to the fANOVA score, which intuitively results in an effect similar to step-wise tuning.
Step-wise tuning, as explained in the following blog on Optuna's LightGBMTuner, tunes the best hyperparameters in order of importance.
This allows for an early and significant reduction of the search space, enabling the discovery of better hyperparameters with fewer evaluations.

> LightGBM Tuner: New Optuna Integration for Hyperparameter Optimization
> https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258

As further improvement of the algorithm, the fANOVA score is calculated in two stages to build a trust region.

* In the first stage, the fANOVA score is calculated using all past trial results to determine the trust region.
* In the second stage, the fANOVA score is calculated using only the top-performing trials to further narrow down the trust region.

This is because the fANOVA score only scores the contribution of each parameter to the objective value, so a parameter with a high fANOVA score, such as the learning rate, can unfairly lower the score of other hyperparameters.
Thus, in the second stage, the top-N trials are used to calculate the fANOVA score with these parameters fixed, and the trust region is determined accordingly.

## Tools

During the competition, I created following tools to smoothly evaluate the performance and analyse the algorithm's behavior.

### `train_surrogate_evaluator.py`

This script trains a surrogate model of the objective function from multiple aiaccel execution results (`reports.csv`) using `RandomForest`.
By using the model to evaluate the algorithm, the evaluation of the algorithm can be done very quickly.

### `tools/optuna_dashboard.py`

A script to create an Optuna Study from the execution results of aiaccel (reports.csv) and visualize/analyze it using Optuna Dashboard.

