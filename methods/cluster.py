# Implementation of Griffin et. al (2023) medoid clustering algorithm
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from math import radians


class GriffinMedoid:
    def __init__(self, locations, demand_corr, weight=0.5, clusters=3):
        """
        Implement the Griffin-Medoid clustering algorithm.

        :param locations: A Nx2 numpy array of store locations  longitude and latitude
        :param demand_corr: A NxN array of demand correlations
        :param weight: the weighting factor between locations and demand correlations
        """

        # Calculate distances
        self.M = clusters
        self.N = locations.shape[0]
        d_ij = np.zeros((self.N, self.N))
        rad_locations = np.zeros((self.N, 2))
        rho_ij = demand_corr

        for i in range(locations.shape[0]):
            rad_locations[i, :] = [radians(_) for _ in locations[i, :]]
        for i in range(self.N):
            for j in range(i):
                if i == j:
                    d_ij = 0
                else:
                    d_ij[i, j] = haversine_distances(
                        [rad_locations[i, :], rad_locations[j, :]]
                    )[0][1]
                    d_ij[j, i] = d_ij[i, j]

        # Normalise (add a mask to ignore diagonal values)
        mask = np.ones(d_ij.shape, dtype=bool)
        d_ij = (d_ij - d_ij.min()) / (
            d_ij.max() - d_ij[mask].min()
        )  # Diagonals are zero but this is fine
        rho_ij = (rho_ij + 1) / 2

        # Calculate D_ij
        self.D = (1 - weight) * d_ij + weight * rho_ij

        # Environmental Variables
        self.K = 1e11

    def run_algorithm(self):
        """
        Run the algorithm:
        usage:
            loc = # longitude and lattitude data #
            dc = # correlation matrix of demand correlations #
            GM = GriffinMedoid
        """
        unassigned_cluster = [i for i in range(self.N)]

        medoids = [0 for i in range(self.M)]
        cluster = [0 for i in range(self.N)]

        A = np.zeros(self.N)
        for m in range(self.M):
            for j in range(self.N):
                A[j] = np.sum([self.D[i, j] for i in unassigned_cluster])
                if j in medoids:
                    A[j] = self.K
            medoids[m] = np.argmin(A)
            unassigned_cluster.remove(medoids[m])
        for m in medoids:
            cluster[m] = m
        for p in range(self.N - self.M):
            for j in range(self.N):
                if j in unassigned_cluster:
                    A[j] = np.sum([self.D[i, j] for i in unassigned_cluster])
                else:
                    A[j] = self.K
            s = np.argmin(A)
            cost = [0 for i in range(self.N)]
            for w in medoids:
                for f in range(self.N):
                    if cluster[f] == cluster[w]:
                        cost[w] += self.D[s, f]
            valid_idx = np.ma.MaskedArray(np.array(cost), np.array(cost) <= 0)
            a = np.argmin(valid_idx)
            cluster[s] = cluster[a]
            unassigned_cluster.remove(s)
        return cluster
