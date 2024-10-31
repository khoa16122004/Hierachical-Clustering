import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

def plot2d_data(df, i):
    if (len(df.columns) > 3):
        print("Plot Waring: more than 2-Dimensions!")
    df.plot(kind='scatter', c=df['cluster'], cmap='gist_rainbow', x=0, y=1)
    plt.show(block=False)
    
    path = str(i) + ".png"
    plt.savefig(path)

def create_gif(image_paths, output_path, duration=500):
    images = [Image.open(img_path) for img_path in image_paths]
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    
