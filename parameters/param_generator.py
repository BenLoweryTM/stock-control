# base instance parameters
import itertools

import numpy as np
import pandas as pd
import pickle as pkl

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
    "initial_inventory": [],
    "online_demand_params": [],
    "store_demand_params": [],
    "demand_distribution": ["Poisson" for _ in range(6)],
    "dfw_chance": 0.2,
}

# Variables we are varing
num_stores = [5, 10, 20]
transhipment_costs = [1, 3]
dfw_proportion = [0.2, 0.5, 0.8]
holding_warehouse = [1, 3]

# Import the clusters
with open('./parameters/cluster_instances.pkl', 'rb') as f:
    cluster_instances = pkl.load(f)

instances = []
for stores, ts_cost, dfw, holding in itertools.product(
    num_stores, transhipment_costs, dfw_proportion, holding_warehouse
):
    for trajectory in range(5):
        # If its 5 stores then the cluster arrangement is just all the same
        if stores == 5:
            instance = BASE_INSTANCE.copy()
            instance["stores"] = stores
            instance['trajectory"'] = trajectory
            instance["cluster_method"] = "None"
            instance["cluster_assignment"] = [1 for i in range(stores)]
            instance["ts_cost_for_cluster"] = {1: ts_cost}
            instance["dfw_chance"] = dfw
            instance["store_demand_params"] = cluster_instances[trajectory][stores]['demand']
            instance["initial_inventory"] = [cluster_instances[trajectory][stores]['online_demand']+sum(cluster_instances[trajectory][stores]['demand'][i][0] for i in range(stores))] + [cluster_instances[trajectory][stores]['demand'][i][0] for i in range(stores-1)]
            instance["online_demand_params"] = [cluster_instances[trajectory][stores]['online_demand'] for _ in range(instance['periods'])]
            instance["holding_warehouse"] = holding
            instances.append(instance)
        else:
            # KMEANS variant
            instance = BASE_INSTANCE.copy()
            instance["stores"] = stores
            instance['trajectory"'] = trajectory
            instance['cluster_method'] = "KM"
            instance['cluster_assignment'] = cluster_instances[trajectory][stores]['KMeans Clusters']
            instance["ts_cost_for_cluster"] = {cluster: ts_cost for cluster in np.unique(instance["cluster_assignment"])} 
            instance["dfw_chance"] = dfw
            instance["online_demand_params"] = [cluster_instances[trajectory][stores]['online_demand'] for _ in range(instance['periods'])]
            instance["store_demand_params"] = cluster_instances[trajectory][stores]['demand']
            instance["initial_inventory"] = [cluster_instances[trajectory][stores]['online_demand']+sum(cluster_instances[trajectory][stores]['demand'][i][0] for i in range(stores))] + [cluster_instances[trajectory][stores]['demand'][i][0] for i in range(stores-1)]
            instance["holding_warehouse"] = holding

            instances.append(instance)

            # Griffin variant
            instance = BASE_INSTANCE.copy()
            instance["stores"] = stores
            instance['trajectory"'] = trajectory
            instance['cluster_method'] = "Griffin"
            instance['cluster_assignment'] =  cluster_instances[trajectory][stores]['Griffin Clusters']
            instance["ts_cost_for_cluster"] = {cluster: ts_cost for cluster in np.unique(instance["cluster_assignment"])}
            instance["dfw_chance"] = dfw
            instance["store_demand_params"] = cluster_instances[trajectory][stores]['demand']
            instance["online_demand_params"] = [cluster_instances[trajectory][stores]['online_demand'] for _ in range(instance['periods'])]
            instance["initial_inventory"] = [cluster_instances[trajectory][stores]['online_demand']+sum(cluster_instances[trajectory][stores]['demand'][i][0] for i in range(stores))] + [cluster_instances[trajectory][stores]['demand'][i][0] for i in range(stores-1)]
            instance["holding_warehouse"] = holding

            instances.append(instance)

# Save to a pickle file
pd.to_pickle(instances, "./parameters/instances.pkl")
