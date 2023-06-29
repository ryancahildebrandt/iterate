# Iterative Clustering Wrapper for Scikit-Learn Clustering Models

---

[*Open*](https://gitpod.io/#https://github.com/ryancahildebrandt/iterate) *in gitpod*

## *Purpose*

This is a small wrapper around scikitlearn clustering models to allow for re-clustering of "noise" datapoints. In many cases, unclustered datapoints returned from a single iteration of a clustering algorithm can themselves be further combined into useful clusters using the same clustering parameters. This wrapper provides a reproducible and tunable algorithm for re-applying a given clustering algorithm to otherwise unclustered datapoints and combining the results from each clustering iteration.

---

## Dataset
The datasets used in the examples for the current project were pulled from the [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets) package, namely the iris, wine, and news datasets

---

## Outputs
- A quickstart [example](https://github.com/ryancahildebrandt/iterate/blob/master/iterative_clusterer.py) notebook to show the basic usage and functionality of the iterative clusterers

**Note**
*Some of the algorithms contained in the sklearn.cluster module may be practically or theoretically incompatible with the present project. You should tune each algorithm to the data you're working with and select the algorithm based on your understanding of the data as much as possible*