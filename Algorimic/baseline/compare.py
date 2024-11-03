import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from PIL import Image
from tqdm import tqdm
from sklearn import cluster, datasets
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_gif(image_paths, output_path, duration=10):  # Duration in milliseconds
    if not image_paths:
        raise ValueError("No image paths provided")
        
    images = []
    for img_path in image_paths:
        try:
            images.append(Image.open(img_path))
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue
            
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
    else:
        raise ValueError("No valid images found")

def plot2d_data(data, clusters, title, path):
    plt.figure(figsize=(8, 6))
    unique_clusters = np.unique(clusters)
    scatter = plt.scatter(data[:, 0], data[:, 1], c=clusters, 
                         cmap='rainbow', s=50)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def save_dendrogram(linkage_matrix, method, output_dir="."):
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title(f"Dendrogram ({method})")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    output_path = os.path.join(output_dir, f"{method}_dendrogram.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def hierarchical_clustering(data: np.ndarray, linkage_method: str = 'ward'):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return linkage(data, method=linkage_method)

def animate_clustering(data, dataset_name, linkage_matrix, n_clusters, linkage_method, output_dir="."):
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    num_merges = len(data) - 1
    paths = []
    
    try:
        for i in tqdm(range(num_merges)):
            current_n_clusters = num_merges - i
            clusters = fcluster(linkage_matrix, t=current_n_clusters, criterion='maxclust')
            
            title = f"Step {i + 1} - Clusters: {current_n_clusters}"
            plot_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            plot2d_data(data, clusters, title, plot_path)
            paths.append(plot_path)
            
            if current_n_clusters == n_clusters:
                final_plot_path = os.path.join(output_dir, f"{dataset_name}_{linkage_method}_final.png")
                plot2d_data(data, clusters, f"Final Clustering ({n_clusters} clusters)", 
                           final_plot_path)
                break
        
        # Create GIF
        gif_path = os.path.join(output_dir, f"{dataset_name}_{linkage_method}.gif")
        create_gif(paths, gif_path)
    
    finally:
        # Clean up temporary files
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            os.rmdir(frames_dir)
        except OSError:
            pass

def get_dataset(name: str, data_dir: str = "../2D_Data") -> pd.DataFrame:
    dataset_paths = {
        "aggregations": "Aggregation.csv",
        "smileface": "SmileFace.csv",
        "t4.8k": "t4.8k.csv"
    }
    
    if name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(dataset_paths.keys())}")
    
    file_path = os.path.join(data_dir, dataset_paths[name])
    sep = ',' if name == "smileface" else ' '
    
    try:
        return pd.read_csv(file_path, sep=sep, header=None)
    except Exception as e:
        raise RuntimeError(f"Error loading dataset {name}: {e}")

if __name__ == "__main__":
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "circles": datasets.make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=170),
        "moons": datasets.make_moons(n_samples=100, noise=0.05, random_state=170),
        "blobs": datasets.make_blobs(n_samples=100, random_state=170),
        "aniso": datasets.make_blobs(n_samples=100, random_state=170),
        "no_structure": (np.random.rand(100, 2), None)
    }

    X, y = datasets["aniso"]
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    datasets["aniso"] = (X_aniso, y)

    linkage_methods = ["single", "complete", "average", "ward"]

    for dataset_name, (data, _) in datasets.items():
        for linkage_method in linkage_methods:
            try:
                linkage_matrix = hierarchical_clustering(data, linkage_method=linkage_method)
                animate_clustering(data, dataset_name, linkage_matrix, n_clusters=2, 
                                   linkage_method=linkage_method, output_dir=output_dir)
                save_dendrogram(linkage_matrix, linkage_method, output_dir)
            except Exception as e:
                print(f"Error during clustering with dataset {dataset_name} and method {linkage_method}: {e}")
