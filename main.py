# Test the dfw model w/ Transhipments works: namely ESR
import gymnasium as gym
import inventorygyms
import inventorygyms.wrappers.transhipment.ESR_rust as ESR
import numpy as np

params = {
    'periods': 10, 
    'stores': 4,
    'lead_time': [1,1,0],
    'ts_cost': 1,
    'penalty': 9,
    'initial_inventory': [[10,10],[12,0],[7,0],[6,0], [9,0]],
    'online_demand_params': [0 for i in range(10)],
    'store_demand_params': [[(6,0.375) for i in range(10)] for t in range(4)],
    'dfw_chance': 0.8,
    'demand_distribution': ['Poisson'] + ['Negative Binomial' for i in range(4)],
}
out = {'warehouse': 100, 'store': [10,10,7,8], 'r': [6,7,8,9]}
env_2 = gym.make('inventorygyms/TwoEchelonPLSTS-v0', **params)

trans_env = ESR.ts_ESR(env_2)

terminated = False
cost = []
j=0
while not terminated:
    print('Period: {}'.format(j))
    j+=1
    # Generate action
    action = trans_env.generate_action('Capped', out, True)

    # Take a step in time.
    observation, reward, terminated, truncated, info = trans_env.step(action)