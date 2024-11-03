import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from PIL import Image
from tqdm import tqdm
from sklearn import cluster, datasets
import numpy as np

def create_gif(image_paths, output_path, duration=5):
    images = [Image.open(img_path) for img_path in image_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

def plot2d_data(df, clusters, title, path):
    unique_clusters = np.unique(clusters)
    plt.figure(figsize=(8, 6))
    plt.scatter(df[0], df[1], c=unique_clusters, cmap='gist_rainbow', s=50)
    plt.title(title)
    plt.savefig(path)
    plt.close()

def save_dendrogram(linkage_matrix, method):
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title(f"Dendrogram ({method})")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.savefig(f"{method}_dendrogram.png")
    plt.close()

def hierarchical_clustering(df: pd.DataFrame, linkage_method: str = 'ward'):
    linkage_matrix = linkage(df, method=linkage_method)
    return linkage_matrix

def animate_clustering(df, linkage_matrix, n_clusters, linkage_method):
    num_merges = linkage_matrix.shape[0]
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    paths = []
    
    for i in tqdm(range(num_merges)):
        clusters = fcluster(linkage_matrix, t=num_merges - i, criterion='maxclust')
        
        title = f"Step {i + 1} - Clusters: {num_merges - i}"
        plot_path = os.path.join(frames_dir, f"frame_{i + 1}.png")
        plot2d_data(df, clusters, title, plot_path)
        paths.append(plot_path)
        
        if num_merges - i == n_clusters:
            plot2d_data(df, clusters, title, f"{linkage_method}.png")
            break
        
    create_gif(paths, f"{linkage_method}.gif")
    
    for path in paths:
        os.remove(path)

def get_dataset(name: str) -> pd.DataFrame:
    if name == "aggregations":
        df = pd.read_csv("D:/Hierachical-Clustering/2D_Data/Aggregation.csv", sep=' ', header=None)
    elif name == "smileface":
        df = pd.read_csv("../2D_Data/SmileFace.csv", sep=',', header=None)
    elif name == "t4.8k":
        df = pd.read_csv("../2D_Data/t4.8k.csv", sep=' ', header=None)
    return df

if __name__ == "__main__":
    # df = get_dataset("aggregations")
    # print(df.shape)
    n_samples = 1500
    df, Y = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=170
    )
    print(df.shape)
    
    linkage_methods = ["single", "complete", "average", "ward"]
    for linkage_method in linkage_methods:
        linkage_matrix = hierarchical_clustering(df, linkage_method=linkage_method)
        animate_clustering(df, linkage_matrix, n_clusters=2, linkage_method=linkage_method)
    
        save_dendrogram(linkage_matrix, linkage_method)
    
