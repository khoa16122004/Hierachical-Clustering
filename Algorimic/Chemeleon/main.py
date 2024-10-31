import pandas as pd

from visualization import *
from chameleon import *
from config import *

def get_dataset(name: str) -> pd.DataFrame:
    if name == "aggregations":
        df = pd.read_csv("../2D_Data/Aggregation.csv", sep=' ', header=None)
        
    elif name == "smileface":
        df = pd.read_csv("../2D_Data/SmileFace.csv", sep=',', header=None),
    
    elif name == "t4.8k":
        df = pd.read_csv("../2D_Data/t4.8k.csv", sep=' ', header=None)
        
    return df


if __name__ == "__main__":
    # get a set of data points
    df = get_dataset(DATASET)
    res = cluster("Aggregation.gif", df, 7, knn=25, m=40, alpha=2.0, plot=True)
    plot2d_data(res, "aggregation.png")
