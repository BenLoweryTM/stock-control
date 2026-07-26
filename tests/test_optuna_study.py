"""
Usage
-----------------
    uv run ./tests/test_optuna_study.py --jobs 8 --trials 100

"""

import argparse
import threading
from functools import partial
from multiprocessing import cpu_count
import gymnasium as gym
import inventorygyms  # noqa: F401
import inventorygyms.wrappers.transhipment.lookahead as LA
import numpy as np
import optuna
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed instance definition (edit here to change the scenario)
# ---------------------------------------------------------------------------
BASE_INSTANCE = {
    "periods": 14,
    "stores": 5,
    "lead_time": [1, 1, 0],
    "warehouse_capacity": 250,
    "cluster_assignment": [1, 1, 1, 1, 1],
    "ts_cost_for_cluster": {1: 4.0},
    "dfw_cost": 0,
    "penalty": 18,
    "holding_warehouse": 1,
    "holding_store": 3,
    "initial_inventory": [[19, 0], [10, 0], [10, 0], [10, 0], [4, 0], [4, 0]],
    "online_demand_params": [0 for _ in range(14)],
    "store_demand_params": [[5 for _ in range(14)] for _ in range(3)] + [[2 for _ in range(14)] for _ in range(2)],
    "demand_distribution": ["Poisson" for _ in range(6)],
    "dfw_chance": 0.2,
}

# Number of Monte-Carlo replications per Optuna trial (higher = less noise, slower)
N_SIMS = 5000

STUDY_NAME = "inventory_tuning"

# Thread-local storage ensures each thread owns its own gym environment,
# avoiding any shared-state issues between concurrent trials.
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
def _get_env(instance: dict) -> LA.ts_la:
    """Return a thread-local wrapped environment, creating it on first use."""
    if not hasattr(_thread_local, "wrapped_env"):
        env = gym.make("inventorygyms/TwoEchelonPLSTS-v0", **instance)
        _thread_local.wrapped_env = LA.ts_la(env)
        _thread_local.wrapped_env.reset()
    return _thread_local.wrapped_env


def run_simulation(instance: dict, warehouse_order_up_to: int, seed: int = 42) -> float:
    """
    Run N_SIMS replications of the DES and return the mean per-period cost.

    Parameters
    ----------
    instance:
        Environment kwargs forwarded to gym.make.
    warehouse_order_up_to:
        The warehouse replenishment target passed to generate_action.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    float
        Mean per-period cost (lower is better).
    """
    wrapped_env = _get_env(instance)
    wrapped_env.reset(seed=seed)

    all_period_costs: list[float] = []

    for _ in range(N_SIMS):
        sim_costs = []
        terminated = False
        while not terminated:
            action = wrapped_env.generate_action(warehouse_order_up_to, True, "RegBS")
            _, reward, terminated, _, _ = wrapped_env.step(action)
            sim_costs.append(-reward)
        all_period_costs.append(float(np.sum(sim_costs)))
        wrapped_env.reset()

    # Find the 95% credible interval of the mean total cost
    lower_bound = np.mean(all_period_costs) - 1.96 * np.std(all_period_costs) / np.sqrt(N_SIMS)
    upper_bound = np.mean(all_period_costs) + 1.96 * np.std(all_period_costs) / np.sqrt(N_SIMS)
    print(f"95% confidence interval for {warehouse_order_up_to}: [{lower_bound}, {upper_bound}]")

    
    # Return the mean per-period cost
    # (lower is better)
    return np.mean(all_period_costs)

    return float(np.mean(all_period_costs))

# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial, instance_params: dict) -> float:
    """
    Optuna objective function.

    Tune
    ----
    warehouse_order_up_to : int  – warehouse order-up-to level [0, 200]

    Extend with additional trial.suggest_* calls to tune other parameters,
    e.g. store base-stock levels, penalty costs, holding costs.
    """
    warehouse_order_up_to = trial.suggest_int("warehouse_order_up_to", 20, 40)

    # --- optional: tune instance-level parameters ---
    # penalty = trial.suggest_float("penalty", 5.0, 40.0)
    # instance = {**BASE_INSTANCE, "penalty": penalty}
    instance = instance_params

    # Use trial.number as seed — unique per trial, deterministic across runs
    return run_simulation(instance, warehouse_order_up_to, seed=trial.number)


# ---------------------------------------------------------------------------
# Study entry-point
# ---------------------------------------------------------------------------
def run_study(
    n_trials: int = 50,
    n_jobs: int = 1,
    study_name: str = STUDY_NAME,
    instance_params: dict = BASE_INSTANCE,
) -> optuna.Study:
    """
    Create and run an Optuna study using n_jobs parallel threads.

    Parameters
    ----------
    n_trials:
        Total number of Optuna trials to run.
    n_jobs:
        Number of parallel threads passed to study.optimize().
        1 = sequential, -1 = use all available cores.
    study_name:
        Human-readable name for the study.

    Returns
    -------
    optuna.Study
        The completed study (inspect .best_params, .best_value, etc.).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=sampler,
    )

    effective_jobs = cpu_count() if n_jobs == -1 else n_jobs
    print(f"Running {n_trials} trials with n_jobs={effective_jobs}")

    study.optimize(
        partial(objective, instance_params=instance_params),
        n_trials=n_trials,
        n_jobs=n_jobs,
        gc_after_trial=True,
        show_progress_bar=(n_jobs == 1),  # tqdm is not thread-safe with n_jobs > 1
    )

    return study


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def print_study_summary(study: optuna.Study) -> None:
    print("\n=== Optuna Study Summary ===")
    print(f"  Best value  : {study.best_value:.4f}  (mean per-period cost)")
    print(f"  Best params : {study.best_params}")
    print(f"  Trials run  : {len(study.trials)}")


def trials_to_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Return all completed trials as a tidy DataFrame."""
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rows.append({"trial": t.number, "value": t.value, **t.params})
    return pd.DataFrame(rows).sort_values("value").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna inventory tuning study")
    parser.add_argument("--trials", type=int, default=50, help="Total number of trials")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel threads for study.optimize() (default: 1, use -1 for all cores)",
    )
    parser.add_argument("--study", type=str, default=STUDY_NAME, help="Study name")
    args = parser.parse_args()

    study = run_study(
        n_trials=args.trials,
        n_jobs=args.jobs,
        study_name=args.study,
        instance_params=BASE_INSTANCE,  # Eventually we will replace with actual parameter instances we generated
    )

    print_study_summary(study)

    df = trials_to_dataframe(study)
    print("\n Trial results:")
    print(df.head(5))
    df.to_csv('../results/test.csv')
