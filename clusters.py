import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# Read the data
df = pd.read_csv('data/metadata.csv')

def lat_lon_to_mercator(df):
    lat, lon = df['lat'], df['lon']
    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x/lon
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lat * (np.pi/180.0)/2.0)) * scale
    #create new columns in the dataframe
    df['x_mercator'] = x
    df['y_mercator'] = y
    return df


def plot_clusters(df):
    # Create a scatter plot
    plt.scatter(df['lat'], df['lon'], alpha=0.5)
    plt.show()
    return

#Have the optimal number of clusters
# Create a list of inertia values
def get_optimal_number_cluster(df):
    inertia = []
    for k in range(1, 40):
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)
        
        # Fit model to samples
        model.fit(df[['x_mercator','y_mercator']])
        
        # Append the inertia to the list of inertias
        inertia.append(model.inertia_)
        
        # Plot ks vs inertias
    plt.plot(range(1, 40), inertia, '-+')
    plt.show()
    return 

def get_clusters(k, df):
    kmeans_model = KMeans(n_clusters=k, random_state=1)
    # Fit the model to the data
    kmeans_model.fit(df[['x_mercator','y_mercator']])
    # Get the cluster assignments
    labels = kmeans_model.predict(df[['x_mercator','y_mercator']])
    return labels

def plot_clusters(df, labels):
    # Plot the points with seaborn
    #sns.scatterplot(df['x_mercator'], df['y_mercator'], hue=labels)
    plt.scatter(df['x_mercator'], df['y_mercator'], c=labels, alpha=0.5)
    plt.show()
    return


df = lat_lon_to_mercator(df)
#get_optimal_number_cluster(df)
nb_cluster = 20
plot_clusters(df, get_clusters(nb_cluster, df))

