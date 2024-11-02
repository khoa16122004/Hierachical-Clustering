import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from ipywidgets import interact, IntSlider
from matplotlib.animation import FuncAnimation, PillowWriter
import ipywidgets as widgets

class InteractiveHierarchicalClustering2D:
    def __init__(self, data=None, max_clusters=10, linkage_method='ward', p=3):
        self.max_clusters = max_clusters
        self.linkage_method = linkage_method
        self.data = data if data is not None else np.random.rand(100, 2)
        self.linkage_matrix = linkage(self.data, method=self.linkage_method)
        self.p = p
    
    def apply_clustering(self, n_clusters):
        """Apply hierarchical clustering with a given number of clusters."""
        return fcluster(self.linkage_matrix, t=n_clusters, criterion='maxclust')
    
    def calculate_centroids(self, labels):
        """Calculate the centroids of each cluster based on current labels."""
        centroids = []
        for label in np.unique(labels):
            cluster_points = self.data[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
        return np.array(centroids)
    
    def plot_dendrogram(self):
        """Plot the dendrogram for an overview of the clustering structure."""
        plt.figure(figsize=(10, 6))
        dendrogram(self.linkage_matrix, truncate_mode='level', p=self.p, show_leaf_counts=False, no_labels=True)
        plt.title('Dendrogram for Hierarchical Clustering')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.show()
    
    def interactive_clustering(self, n_clusters):
        """Interactive plot to update clustering based on slider input."""
        plt.figure(figsize=(8, 6))
        labels = self.apply_clustering(n_clusters)
        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
        centroids = self.calculate_centroids(labels)
        
        # Plot centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
        plt.legend()
        plt.title(f'Hierarchical Clustering with {n_clusters} Clusters')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
    
    def save_clustering_animation(self, save_path='clustering_process.gif'):
        """Save the clustering process as an animated GIF."""
        fig, ax = plt.subplots(figsize=(8, 6))

        def update(frame):
            ax.clear()
            labels = self.apply_clustering(n_clusters=frame + 1)
            ax.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
            centroids = self.calculate_centroids(labels)
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
            ax.set_title(f'Clustering with {frame + 1} Clusters')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.legend()

        # Create the animation using FuncAnimation
        ani = FuncAnimation(fig, update, frames=range(1, self.max_clusters + 1), repeat=False)

        # Save the animation as a GIF
        ani.save(save_path, writer=PillowWriter(fps=1))
        print(f"Animation saved as {save_path}")
        plt.close(fig)  # Close the figure to prevent display if running in a notebook
