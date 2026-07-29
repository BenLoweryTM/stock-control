import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generate_stores_with_demand(n_stores, n_clusters, cluster_std=0.05, 
                                demand_mean_range=(50, 500), 
                                demand_std_range=(5, 50),
                                random_state=42):
    """
    Generate store locations on a unit grid with different demand parameters.
    
    Parameters:
    -----------
    n_stores : int
        Number of stores to generate
    n_clusters : int
        Number of clusters (blobs) to distribute stores across
    cluster_std : float, default=0.05
        Standard deviation of store locations within clusters
    demand_mean_range : tuple, default=(50, 500)
        Range for store's mean daily demand
    demand_std_range : tuple, default=(5, 50)
        Range for store's demand volatility (std dev)
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    stores_df : DataFrame
        Store information with columns: store_id, location_x, location_y, 
        demand_mean, demand_std, cluster_id
    X : ndarray of shape (n_stores, 2)
        Store locations
    y : ndarray of shape (n_stores,)
        Cluster labels for each store
    """
    np.random.seed(random_state)
    
    # Generate random cluster centers on unit grid [0, 1]
    centers = np.random.uniform(0, 1, size=(n_clusters, 2))
    
    # Generate location clusters with stores distributed across clusters
    X, y = make_blobs(n_samples=n_stores, centers=centers, n_features=2,
                      cluster_std=cluster_std, random_state=random_state)
    X = np.clip(X, 0, 1)
    
    # Generate demand parameters for each store
    demand_means = np.random.uniform(demand_mean_range[0], demand_mean_range[1], n_stores)
    demand_stds = np.random.uniform(demand_std_range[0], demand_std_range[1], n_stores)
    
    # Create stores dataframe
    stores_df = pd.DataFrame({
        'store_id': range(1, n_stores + 1),
        'location_x': X[:, 0],
        'location_y': X[:, 1],
        'cluster_id': y,
        'demand_mean': demand_means,
        'demand_std': demand_stds
    })
    
    return stores_df, X, y
