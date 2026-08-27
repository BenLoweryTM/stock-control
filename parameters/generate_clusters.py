import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

TIME_HORIZON = 16
NUM_STORES = [5, 10, 20]
N_TRAJECTORIES = 5
N_CLUSTERS = 3
DEMAND_RATE_RANGE = (2, 10)
DEMAND_DECLINE_RATE = 0.10  # 10% decline every T/5 periods
DEFAULT_RANDOM_STATE = 42
OUTPUT_PATH = "./parameters/cluster_instances.pkl"


def generate_declining_poisson_demand(
    n_stores,
    periods=TIME_HORIZON,
    demand_rate_range=DEMAND_RATE_RANGE,
    decline_rate=DEMAND_DECLINE_RATE,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Generate store demand from NumPy Poisson rates selected uniformly from demand_rate_range.

    Each store starts with an integer Poisson rate sampled uniformly from the
    inclusive range [2, 10]. The rate declines by 10% every T/5 periods.
    """
    np.random.seed(random_state)

    min_rate, max_rate = demand_rate_range
    initial_rates = np.random.randint(min_rate, max_rate + 1, size=n_stores)
    periods_per_decline = periods / 5

    demand = []
    for rate in initial_rates:
        store_demand = []
        for period in range(periods):
            decline_steps = int(period // periods_per_decline)
            declined_rate = rate * ((1 - decline_rate) ** decline_steps)
            store_demand.append(max(int(np.random.poisson(declined_rate)), 1))
        demand.append(store_demand)

    return demand


def generate_stores_with_demand(
    n_stores,
    n_clusters=N_CLUSTERS,
    cluster_std=0.05,
    periods=TIME_HORIZON,
    demand_rate_range=DEMAND_RATE_RANGE,
    decline_rate=DEMAND_DECLINE_RATE,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Generate clustered store locations and declining Poisson demand.

    Returns a dictionary containing:
    - cluster_locations: tuple of store location tuples
    - demand: list of per-store demand lists
    - true_assigned_cluster: list of true cluster ids for each store
    """
    np.random.seed(random_state)

    centers = np.random.uniform(0, 1, size=(n_clusters, 2))
    locations, cluster_assignment = make_blobs(
        n_samples=n_stores,
        centers=centers,
        n_features=2,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    locations = np.clip(locations, 0, 1)

    demand = generate_declining_poisson_demand(
        n_stores=n_stores,
        periods=periods,
        demand_rate_range=demand_rate_range,
        decline_rate=decline_rate,
        random_state=random_state,
    )
    mean_demands = 2 * (sum(sum(store_demand) for store_demand in demand) / periods)

    return {
        "cluster_locations": tuple(map(tuple, locations.tolist())),
        "demand": demand,
        "online_demand": mean_demands,
        "true_assigned_cluster": cluster_assignment.tolist(),
    }


def generate_cluster_instances(
    num_stores=NUM_STORES,
    n_trajectories=N_TRAJECTORIES,
    n_clusters=N_CLUSTERS,
    random_state=DEFAULT_RANDOM_STATE,
):
    """Generate cluster instances for each configured trajectory and store count."""
    instances = {}
    for trajectory in range(n_trajectories):
        instances[trajectory] = {}
        for stores in num_stores:
            instances[trajectory][stores] = generate_stores_with_demand(
                n_stores=stores,
                n_clusters=n_clusters,
                random_state=random_state + trajectory * max(num_stores) + stores,
            )

    return instances


if __name__ == "__main__":
    cluster_instances = generate_cluster_instances()
    pd.to_pickle(cluster_instances, OUTPUT_PATH)
