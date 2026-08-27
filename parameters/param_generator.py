# base instance parameters
import itertools

import numpy as np
import pandas as pd

BASE_INSTANCE = {
    "periods": 16,
    "stores": 0,
    "lead_time": [1, 1, 0],
    "warehouse_capacity": 250,
    "cluster_assignment": [],
    "ts_cost_for_cluster": {},
    "dfw_cost": 0,
    "penalty": 18,
    "holding_warehouse": 1,
    "holding_store": 3,
    "initial_inventory": [[19, 0], [10, 0], [10, 0], [10, 0], [4, 0], [4, 0]],
    "online_demand_params": [6 for _ in range(16)],
    "store_demand_params": [],
    "demand_distribution": ["Poisson" for _ in range(6)],
    "dfw_chance": 0.2,
}

# Variables we are varing
num_stores = [5, 10, 20]
transhipment_costs = [1, 3]
dfw_proportion = [0.2, 0.5, 0.8]
holding_warehouse = [1, 3]

instances = []
for stores, ts_cost, dfw, holding in itertools.product(
    num_stores, transhipment_costs, dfw_proportion, holding_warehouse
):
    instance = BASE_INSTANCE.copy()
    instance["stores"] = stores
    instance["cluster_assignment"] = [1 for i in range(stores)]
    instance["ts_cost_for_cluster"] = {1: ts_cost}
    instance["dfw_chance"] = dfw
    instance["store_demand_params"] = [
        [max(np.random.poisson(10), 1) for _ in range(16)] for _ in range(stores)
    ]
    instance["holding_warehouse"] = holding

    instances.append(instance)

# Save to a pickle file
pd.to_pickle(instances, "./parameters/instances.pkl")
