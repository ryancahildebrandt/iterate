#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 02:51:48 PM EDT 2023 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import random
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from attrs import define
from itertools import compress

random.seed(42)

# helper functions
def mean_sils(data: np.array, labels: np.array) -> np.array:
    """
    Calculates mean silhouette score for clustered data

    Args:
        data (np.array): Datapoints being clustered via sklearn.cluster clusterer
        labels (np.array): Labels returned from trained sklearn.cluster clusterer

    Returns:
        np.array: Array of cluster labels and their corresponding mean silhouette score
    """    
    label_set = list(set(labels))
    if len(label_set) == 1:
        sils = [None]
    else:
        sils = list(map(lambda l: np.mean(silhouette_samples(data, labels)[labels == l]), label_set))
    out = np.stack((label_set, sils), axis = 1)
    return out

def id_noise_cluster(silhouette_scores: np.array) -> int:
    """
    Identifies cluster with the worst silhouettte score from a provided set of clusters

    Args:
        silhouette_scores (np.array): Array of cluster labels and silhouette scores, from the mean_sils function

    Returns:
        int: Label corresponding to the most dispersed (noise) cluster
    """    
    noise_ind = np.argmin(silhouette_scores, axis = 0)[0]
    out = int(silhouette_scores[noise_ind, 0])
    return out

def empty_results_df() -> pd.DataFrame:
    """
    Creates an empty dataframe to store results from the iterative clustering process

    Returns:
        pd.DataFrame: Results dataframe with necessary columns
    """    
    out = pd.DataFrame(columns = ["name", "data", "round", "label", "cluster_size", "silhouette_score", "mean_silhouette_score"])
    return out

# iterative clusterer class
@define
class IterativeClusterer:
    """
    Object to handle the iterative clustering process
    
    Args:
        base_clusterer (sklearn.cluster clusterer): Clusterer with any necessary parameters to tweak the subsequent clustering
        data (np.array): Dataset to cluster
        names (list, optional): Names or identifiers to associate with each datapoint. Defaults to integer identifiers
    
    Attributes:
    results (pd.DataFrame): Results of iterative clustering, grouped by cluster
    results_exploded (pd.DataFrame): Results of iterative clustering, for each datapoint
    _round_counter (int): Tracks number of clustering iterations performed
    _current_noise_n (int): Tracks number of datapoints still unclustered
    _previous_noise_n (int): Tracks number of datapoints still unclustered as of previous iteration
    _noise_subset (np.array): Stores subset of original data still unclustered
    _noise_names (list): Stores names for unclustered datapoints
    _is_singular (bool): Tracks whether latest iteration of clustering was unable to find clusters with given parameters

    """    
    base_clusterer: any
    data: np.array
    names: list = []
    results: pd.DataFrame = empty_results_df()
    results_exploded: pd.DataFrame = empty_results_df()
    _round_counter: int = 0
    _current_noise_n: int = 0
    _previous_noise_n: int = 0
    _noise_subset: np.array = None
    _noise_names: list = None
    _is_singular: bool = False

    def fit(self, data: np.array):
        """
        Wrapper to handle fitting a dataset with the base clusterer
        Shortcut for self.base_clusterer.fit(data) while still allowing a single fit iteration in the same object

        Args:
            data (np.array): Dataset to cluster

        Returns:
            Clusterer fit to provided data
        """        
        return self.base_clusterer.fit(data)

    def _single_iteration(self) -> None:
        """
        Single iteration of the iterative clustering
        Updates noise/clustered data information and appends to results as necessary

        Returns:
            None
        """        
        if self.names == []:
            self.names = list(range(len(self.data)))

        if self._round_counter == 0:
            data_to_cluster = self.data
        else:
            data_to_cluster = self._noise_subset 
            
        self.fit(data_to_cluster)
        
        noise_cluster_id = id_noise_cluster(mean_sils(data_to_cluster, self.base_clusterer.labels_))
        clustered_subset = data_to_cluster[self.base_clusterer.labels_ != noise_cluster_id, :]
        clustered_labels = list(compress(self.base_clusterer.labels_, self.base_clusterer.labels_ != noise_cluster_id))
        clustered_names = list(compress(self.names, self.base_clusterer.labels_ != noise_cluster_id))

        if len(set(clustered_labels)) <= 1:
            self._is_singular = True
        else:
            self._previous_noise_n = self._current_noise_n
            self._noise_subset = data_to_cluster[self.base_clusterer.labels_ == noise_cluster_id, :]
            self._current_noise_n = len(self._noise_subset)
            self._noise_names = list(compress(self.names, self.base_clusterer.labels_ == noise_cluster_id))

            iteration_results = pd.DataFrame({
                "name" : clustered_names,
                "data" : clustered_subset.tolist(),
                "label" : clustered_labels,
                "silhouette_score" : silhouette_samples(clustered_subset, clustered_labels)
            }).groupby(["label"]).agg(lambda x: list(x)).reset_index()
            iteration_results["round"] = self._round_counter
            iteration_results["cluster_size"] = iteration_results["data"].map(len)
            iteration_results["mean_silhouette_score"] = iteration_results["silhouette_score"].map(np.mean)
            print(f"Round {self._round_counter} completed, {self._current_noise_n} observations in noise cluster")
            self._round_counter += 1
            self.results = pd.concat([self.results, iteration_results])
        return None

    def run_iterative_clustering(self, target_noise_proportion = 0.1) -> list:
        """
        Runs iterations of the clustering algorithm 
        Stops upon reaching the target noise proportion or when data cannot be clustered any further with the provided parameters

        Args:
            target_noise_proportion (float, optional): Proportion of original data which can remain unclustered , aka noise. Defaults to 0.1.

        Returns:
            list: Dataframes of the results, aggregated by cluster as well as split out into individual datapoints for convenience
        """        
        self._single_iteration()
        while (
            self._current_noise_n >= np.ceil(target_noise_proportion * len(self.data))
            and 
            self._previous_noise_n != self._current_noise_n
            and not
            self._is_singular
            and not
            self._noise_subset is None
            ):
            self._single_iteration()
            
        self.results["label"] = range(len(self.results))
        
        noise_row = pd.DataFrame({
            "name" : [self._noise_names],
            "data" : [self._noise_subset],
            "round" : self._round_counter - 1, 
            "label" : -1,
            "cluster_size" : self._current_noise_n,
            "mean_silhouette_score" : [-1],
            "silhouette_score" : [[-1] * self._current_noise_n]
        })
        self.results = pd.concat([self.results, noise_row])
        self.results_exploded = self.results.explode(["data", "silhouette_score", "name"])
        
        return [self.results, self.results_exploded]
