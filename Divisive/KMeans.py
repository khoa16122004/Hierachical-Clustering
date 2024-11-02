import pandas as pd
from sklearn.cluster import KMeans, BisectingKMeans
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Function to create a GIF showing clustering results for a range of K values
def create_knn_clustering_gif(X, min_k=2, max_k=10, filename="knn_clustering.gif"):
    fig, ax = plt.subplots(figsize=(8, 5))

    def update(k):
        ax.clear()
        kmeans = KMeans(n_clusters=k, max_iter=50, random_state=0)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        # Plot clusters and centroids
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"K = {k}")
        ax.legend()
        ax.grid(True)

    # Generate animation
    ani = FuncAnimation(fig, update, frames=range(min_k, max_k + 1), repeat=False)
    ani.save(filename, writer=PillowWriter(fps=1))