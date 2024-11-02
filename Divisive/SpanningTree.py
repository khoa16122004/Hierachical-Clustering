import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import matplotlib.cm as cm
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.animation import FuncAnimation, PillowWriter

class MSTDivisiveClustering:
    def __init__(self, data, max_clusters=10, max_cuts=5):
        self.data = data
        self.max_clusters = max_clusters
        self.max_cuts = max_cuts
        self.steps, self.pos = self.precompute_divisive_clustering_steps()

    def minimum_spanning_tree_once(self):
        """Construct the initial minimum spanning tree of the data points once."""
        distance_matrix = squareform(pdist(self.data))
        G = nx.from_numpy_array(distance_matrix)
        mst = nx.minimum_spanning_tree(G)
        return mst, distance_matrix

    def calculate_centroid(self, cluster):
        """Calculate the centroid of a given cluster."""
        if len(cluster) == 0:
            return None
        return np.mean(self.data[list(cluster)], axis=0)

    def plot_initial_data(self):
      """Plot the initial data points without any edges or MST."""
      plt.figure(figsize=(8, 4))
      plt.scatter(self.data[:, 0], self.data[:, 1], color='purple', alpha=0.7)
      plt.title("Initial Data Points (No MST)")
      plt.xlabel("X-axis")
      plt.ylabel("Y-axis")

      # Set the limits based on the data
      plt.xlim(np.min(self.data[:, 0]) - 0.1, np.max(self.data[:, 0]) + 0.1)
      plt.ylim(np.min(self.data[:, 1]) - 0.1, np.max(self.data[:, 1]) + 0.1)

      plt.grid(True)  # Optional: Add grid lines for better visibility
      plt.tight_layout()  # Adjust layout to fit all elements
      plt.show()

    def plot_mst_and_clusters(self, mst, clusters, step):
      """Plot the MST and current clusters without weights."""
      plt.figure(figsize=(8, 4))

      # Draw edges of the MST without labels
      nx.draw_networkx_edges(mst, self.pos, alpha=0.5)

      # Define a colormap
      colors = cm.get_cmap('viridis', len(clusters))

      # Draw clusters and centroids with scatter
      for idx, cluster in enumerate(clusters):
          cluster_color = colors(idx)  # Get color for the cluster
          cluster_points = np.array([self.pos[node] for node in cluster])

          # Plot each cluster with scatter
          plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_color, label=f"Cluster {idx}", alpha=0.7)

          # Calculate and plot the centroid of the cluster
          centroid = self.calculate_centroid(cluster)
          
          if centroid is not None:
              plt.scatter(*centroid, color='red', s=200, marker='X')

      plt.title(f"Step {step}: Clusters after removing an edge" if step >= 1 else "Data with MST")
      plt.xlabel("X")
      plt.ylabel("Y")
      
      # Set the limits based on the data
      plt.xlim(np.min(self.data[:, 0]) - 0.1, np.max(self.data[:, 0]) + 0.1)
      plt.ylim(np.min(self.data[:, 1]) - 0.1, np.max(self.data[:, 1]) + 0.1)

      plt.grid(True)  # Optional: Add grid lines for better visibility
      plt.tight_layout()  # Adjust layout to fit all elements
      plt.legend()
      plt.show()



    def precompute_divisive_clustering_steps(self):
        """Precompute all clustering steps in the divisive hierarchical clustering using MST."""
        mst, distance_matrix = self.minimum_spanning_tree_once()
        clusters = [set(np.arange(len(self.data)))]
        steps = [('initial', clusters.copy(), 0)]
        steps.append((mst.copy(), clusters.copy(), 0))
        current_cluster_index = len(self.data)

        sorted_edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

        cuts = 0

        while cuts < self.max_cuts and len(clusters) < self.max_clusters and sorted_edges:
            max_edge = sorted_edges.pop(0)
            mst.remove_edge(max_edge[0], max_edge[1])

            components = list(nx.connected_components(mst))
            clusters = [set(component) for component in components]

            steps.append((mst.copy(), clusters.copy(), cuts + 1))
            cuts += 1

        return steps, {i: self.data[i] for i in range(len(self.data))}



    def divisive_clustering_mst_interactive(self):
        """Interactive function to visualize divisive clustering steps using precomputed steps."""
        def display_step(step_index):
            clear_output(wait=True)
            if self.steps[step_index][0] == 'initial':
                self.plot_initial_data()
            else:
                mst, clusters, step = self.steps[step_index]
                self.plot_mst_and_clusters(mst, clusters, step)

        slider = widgets.IntSlider(value=0, min=0, max=len(self.steps)-1, step=1, description="Step")
        display(widgets.interactive(display_step, step_index=slider))


    def create_gif(self, filename="mst.gif"):
        """Create a GIF of the MST divisive clustering process."""
        fig, ax = plt.subplots(figsize=(8, 4))

        def update(frame):
            ax.clear()
            if self.steps[frame][0] == 'initial':
                ax.scatter(self.data[:, 0], self.data[:, 1], color='purple', alpha=0.7)
                ax.set_title("Initial Data Points (No MST)")
            else:
                mst, clusters, step = self.steps[frame]
                nx.draw_networkx_edges(mst, self.pos, alpha=0.5, ax=ax)
                colors = cm.get_cmap('viridis', len(clusters))

                for idx, cluster in enumerate(clusters):
                    cluster_color = colors(idx)
                    cluster_points = np.array([self.pos[node] for node in cluster])
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_color, alpha=0.7)
                    centroid = self.calculate_centroid(cluster)

                    if centroid is not None:
                        ax.scatter(*centroid, color='red', s=200, marker='X')

                ax.set_title(f"Step {step}: Clusters after removing an edge" if step >= 1 else "Data with MST")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        ani = FuncAnimation(fig, update, frames=len(self.steps), repeat=False)
        ani.save(filename, writer=PillowWriter(fps=1))
        plt.close(fig)