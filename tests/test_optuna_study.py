"""
Usage
-----------------
    uv run ./tests/test_optuna_study.py --jobs 8 --trials 100

"""

import argparse
from multiprocessing import cpu_count

import gymnasium as gym
import inventorygyms  # noqa: F401
import inventorygyms.wrappers.transhipment.lookahead as LA
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

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
    "store_demand_params": [[5 for _ in range(14)] for _ in range(3)]
    + [[2 for _ in range(14)] for _ in range(2)],
    "demand_distribution": ["Poisson" for _ in range(6)],
    "dfw_chance": 0.2,
}

# Number of Monte-Carlo replications per Optuna trial (higher = less noise, slower)
N_SIMS = 5000

STUDY_NAME = "inventory_tuning"


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
def _create_env(instance: dict) -> LA.ts_la:
    """Create a fresh wrapped environment (safe for multiprocessing)."""
    env = gym.make("inventorygyms/TwoEchelonPLSTS-v0", **instance)
    wrapped_env = LA.ts_la(env)
    wrapped_env.reset()
    return wrapped_env


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
    wrapped_env = _create_env(instance)
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
    # lower_bound = np.mean(all_period_costs) - 1.96 * np.std(all_period_costs) / np.sqrt(N_SIMS)
    # upper_bound = np.mean(all_period_costs) + 1.96 * np.std(all_period_costs) / np.sqrt(N_SIMS)
    # print(f"95% confidence interval for {warehouse_order_up_to}: [{lower_bound}, {upper_bound}]")

    # Return the mean per-period cost
    # (lower is better)
    return np.mean(all_period_costs)


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
    warehouse_order_up_to = trial.suggest_int("warehouse_order_up_to", 0, 200)

    # --- optional: tune instance-level parameters ---
    # penalty = trial.suggest_float("penalty", 5.0, 40.0)
    # instance = {**BASE_INSTANCE, "penalty": penalty}
    instance = instance_params

    # Use trial.number as seed — unique per trial, deterministic across runs
    return run_simulation(instance, warehouse_order_up_to, seed=trial.number)


# ---------------------------------------------------------------------------
# Parallel trial executor (module-level for pickling with joblib)
# ---------------------------------------------------------------------------
def run_replication_chunk(
    chunk_id: int,
    warehouse_order_up_to: int,
    instance_params: dict,
    trial_seed: int,
    chunk_size: int,
    n_sims: int,
) -> list[float]:
    """
    Run a chunk of Monte Carlo replications in parallel.

    This must be at module level (not nested) to be picklable by joblib.

    Parameters
    ----------
    chunk_id : int
        Which chunk (0, 1, 2, ...) to identify which replications to run.
    warehouse_order_up_to : int
        Parameter value for this trial.
    instance_params : dict
        Instance configuration.
    trial_seed : int
        Base seed for reproducibility.
    chunk_size : int
        Number of replications per chunk.
    n_sims : int
        Total replications (to handle last chunk correctly).

    Returns
    -------
    list[float]
        Total costs for each replication in this chunk.
    """
    costs = []
    env = _create_env(instance_params)
    n_reps = (
        chunk_size
        if chunk_id < (n_sims // chunk_size)
        else n_sims - (chunk_id * chunk_size)
    )
    env.reset(seed=trial_seed + chunk_id)
    for _ in range(n_reps):
        sim_costs = []
        terminated = False
        while not terminated:
            action = env.generate_action(warehouse_order_up_to, True, "RegBS")
            _, reward, terminated, _, _ = env.step(action)
            sim_costs.append(-reward)
        costs.append(float(np.sum(sim_costs)))
        env.reset()

    return costs


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
    Create and run an Optuna study with joblib multiprocessing.

    Strategy:
    - Optuna's TPE sampler suggests parameter values (n_trials times)
    - Each parameter suggestion is evaluated using joblib to parallelize
      the N_SIMS Monte Carlo replications
    - This allows true multiprocessing while respecting Optuna's optimization

    Parameters
    ----------
    n_trials:
        Total number of Optuna trials to run.
    n_jobs:
        Number of parallel processes for replications (1 = sequential, -1 = all cores).
    study_name:
        Human-readable name for the study.
    instance_params:
        Instance configuration dict.

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

    print(f"Running {n_trials} trials")
    print(
        f"Parallelizing {N_SIMS} replications per trial across {effective_jobs} processes"
    )

    def parallel_objective(trial: optuna.Trial) -> float:
        """
        Objective function: Optuna controls trial parameters,
        we parallelize replication sampling with joblib.
        """
        warehouse_order_up_to = trial.suggest_int("warehouse_order_up_to", 0, 200)

        if effective_jobs == 1:
            # Sequential: just run normally
            return run_simulation(
                instance_params, warehouse_order_up_to, seed=trial.number
            )
        else:
            # Parallel: split N_SIMS replications across processes
            chunk_size = max(1, N_SIMS // effective_jobs)
            n_chunks = effective_jobs

            # Parallelize replication sampling using module-level function
            replication_results = Parallel(
                n_jobs=effective_jobs,
                backend="multiprocessing",
            )(
                delayed(run_replication_chunk)(
                    chunk_id=i,
                    warehouse_order_up_to=warehouse_order_up_to,
                    instance_params=instance_params,
                    trial_seed=trial.number,
                    chunk_size=chunk_size,
                    n_sims=N_SIMS,
                )
                for i in range(n_chunks)
            )

            # Flatten and compute mean
            all_costs = [cost for chunk in replication_results for cost in chunk]
            return float(np.mean(all_costs))

    # Optuna controls everything: trial sampling, feedback loop, optimization
    study.optimize(
        parallel_objective,
        n_trials=n_trials,
        gc_after_trial=True,
        n_jobs=2,
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
    parser = argparse.ArgumentParser(
        description="Run Optuna inventory tuning study with joblib multiprocessing"
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Total number of trials to run"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel processes for joblib (1 = sequential, -1 = all cores)",
    )
    parser.add_argument("--study", type=str, default=STUDY_NAME, help="Study name")
    args = parser.parse_args()

    study = run_study(
        n_trials=args.trials,
        n_jobs=args.jobs,
        study_name=args.study,
        instance_params=BASE_INSTANCE,
    )

    print_study_summary(study)

    df = trials_to_dataframe(study)
    print("\nTop 5 trial results:")
    print(df.head(5))
    df.to_csv("../results/test.csv")
