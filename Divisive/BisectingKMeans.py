import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import clear_output
from ipywidgets import interact, IntSlider

class BisectingKmeansClustering:
    def __init__(self, X, max_clusters=5):
        self.X = X
        self.max_clusters = max_clusters
        self.clusters = [X]
        self.labels_history = [np.zeros(X.shape[0], dtype=int)]
        self.sse_history = []

    def calculate_average_sse(self, cluster):
        kmeans = KMeans(n_clusters=1, random_state=0).fit(cluster)
        sse = np.sum((cluster - kmeans.cluster_centers_) ** 2)
        return sse / len(cluster)

    def split_cluster(self):
        cluster_scores = [(self.calculate_average_sse(cluster), idx) for idx, cluster in enumerate(self.clusters)]
        max_sse_value, max_sse_index = max(cluster_scores, key=lambda x: x[0])
        max_sse_cluster = self.clusters[max_sse_index]

        kmeans = KMeans(n_clusters=2, random_state=0).fit(max_sse_cluster)
        new_clusters = [max_sse_cluster[kmeans.labels_ == i] for i in range(2)]

        self.clusters.pop(max_sse_index)
        self.clusters.extend(new_clusters)
        self.update_labels()

    def update_labels(self):
        labels = np.zeros(self.X.shape[0], dtype=int)
        label_counter = 0
        for cluster in self.clusters:
            cluster_indices = np.where((self.X[:, None] == cluster).all(-1).any(1))[0]
            labels[cluster_indices] = label_counter
            label_counter += 1
        self.labels_history.append(labels.copy())

    def fit(self):
        while len(self.clusters) < self.max_clusters:
            self.split_cluster()

    def plot_step(self, step):
        """Plot clusters and SSE at a specific step in the clustering process, including centroids."""
        # Move figure creation outside the condition to enforce single-instance display
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(figsize=(8, 6))  # Create fig and ax here

        # Ensure clear output right before plotting
        clear_output(wait=True)

        if step < len(self.labels_history):
            # Plot clusters with centroids
            labels = self.labels_history[step]
            unique_labels = np.unique(labels)
            for label in unique_labels:
                cluster_points = self.X[labels == label]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}", s=70, alpha=0.8)

                # Calculate the centroid for the current cluster and plot it
                centroid = cluster_points.mean(axis=0)
                ax.scatter(centroid[0], centroid[1], color='red', s=200, marker='X')

            ax.set_xlabel("X", fontsize=14)
            ax.set_ylabel("Y", fontsize=14)
            ax.set_title(f"Divisive Clustering Step {step}", fontsize=16)
            ax.legend()

        plt.show()
        plt.close(fig)  # Close the figure to prevent duplicate display


    def interactive(self):
        """Display an interactive plot to view clustering steps."""
        interact(self.plot_step, step=IntSlider(min=0, max=len(self.labels_history)-1, step=1, value=0))

    def create_gif(self, filename="BisectingKmeansClustering_SmillingFace.gif"):
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(figsize=(8, 6))

        def update(step):
            ax.clear()  # Clear ax instead of creating a new figure each time
            labels = self.labels_history[step]
            unique_labels = np.unique(labels)
            
            # Plot clusters with centroids
            for label in unique_labels:
                cluster_points = self.X[labels == label]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}", s=70, alpha=0.8)
                centroid = cluster_points.mean(axis=0)
                ax.scatter(centroid[0], centroid[1], color='red', s=200, marker='X')

            ax.set_xlabel("X", fontsize=14)
            ax.set_ylabel("Y", fontsize=14)
            ax.set_title(f"Step {step}", fontsize=16)
            ax.legend()
            plt.tight_layout()

        ani = FuncAnimation(fig, update, frames=len(self.labels_history), repeat=False)
        ani.save(filename, writer=PillowWriter(fps=1))
        plt.close(fig)  # Close the figure after saving to GIF