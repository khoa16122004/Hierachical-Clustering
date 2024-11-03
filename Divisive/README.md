# Divisive Clustering Methods

This folder implements Divisive Clustering in three ways:

1. **Hierarchical Clustering**:
   - Uses the `fcluster` function to perform divisive clustering.
   - Implemented in [LibraryFCluster.py](Divisive/LibraryFCluster.py).

2. **Bisecting K-Means**:
   - Custom implementation of the first approach mentioned in the book *Data Mining: Concepts and Techniques* (Section 8.3.3, Divisive Hierarchical Clustering).
   - Implemented in [BisectingKMeans.py](Divisive/BisectingKMeans.py).

3. **Spanning Tree**:
   - Custom implementation of the second approach using the Minimum Spanning Tree (MST) method mentioned in the book *Data Mining: Concepts and Techniques* (Section 8.3.3, Divisive Hierarchical Clustering).
   - Implemented in [SpanningTree.py](Divisive/SpanningTree.py).

## How to Run

To run the divisive clustering methods, execute the `DivisiveClustering.ipynb` notebook.