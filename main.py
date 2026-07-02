import gymnasium as gym
import inventorygyms  # noqa: F401
import inventorygyms.wrappers.transhipment.lookahead as LA
import numpy as np
import scipy.stats as sp
import pandas as pd
import sys

def run_la_instance(instance):
        # Create the environment
        env_all = gym.make('inventorygyms/TwoEchelonPLSTS-v0', **instance)

        # Wrap with transhipment policy (in this case, ESR: but we will have no transhipments)
        wrapped_env = LA.ts_la(env_all)
        wrapped_env.reset()
        # Run Iterations
        sims = 50
        info_pd = []
        for sim in range(sims):
            print(sim)
            overall_costs = []
            terminated = False
            sim_cost = []
            T = 0


            while not terminated:
                out_all = {
                    'warehouse': 0,
                }
                # Generate action
                action = wrapped_env.generate_action(out_all['warehouse'], True,'RegBS')
                # Take a step in time.
                observation, reward, terminated, truncated, info = wrapped_env.step(action)
                info['run'] = sim+1
                info['period'] = T
                sim_cost.append(-reward)
                T+=1
                info_pd.append(info)
            overall_costs.append(np.sum(sim_cost))
            wrapped_env.reset()

        # DF runs
        df_run_infos = pd.DataFrame(info_pd)
        df_run_infos['Starting Inv. Store'] = [[d[i][0] for i in range(instance['stores'])] for d in df_run_infos['Starting Inv. Store']]
        df_run_infos['Ending Inventory Store'] = [[d[i][0] for i in range(instance['stores'])] for d in df_run_infos['Ending Inventory Store']]
        df_run_infos['Starting Inv. Warehouse'] = [d[0]for d in df_run_infos['Starting Inv. Warehouse']]
        df_run_infos['Ending Inventory Warehouse'] = [d[0]for d in df_run_infos['Ending Inventory Warehouse']]
        return df_run_infos



if __name__ == "__main__":
    instance = {
        "periods": 16,
        "stores": 5,
        "lead_time": [1, 1, 0],
        "warehouse_capacity": 250,
        "cluster_assignment": [0,1,1,1,0],
        "ts_cost_for_cluster": {0: 1.0,1: 3.0},
        "dfw_cost": 0,
        "penalty": 18,
        "holding_warehouse": 2,
        "holding_store": 3,
        "initial_inventory": [[50, 0], [22,0], [20,0], [18,0], [16,0], [14,0]],
        "online_demand_params": [12 for i in range(16)],
        "store_demand_params": [[6 for i in range(16)] for _ in range(5)],
        "demand_distribution": ['Poisson' for p in range(6)],
        "dfw_chance": 0.8,
    }
    res = run_la_instance(instance)

    print(res)

