from sklearn.cluster import KMeans
import pandas as pd

def cluster_drivers(df, n_clusters=4):
    """
    Clusters drivers based on telemetry features.
    Returns DataFrame with assigned clusters and the clustering model.
    """
    feats = ['avg_lap_time', 'elo'] if 'avg_lap_time' in df.columns and 'elo' in df.columns else df.select_dtypes('number').columns.tolist()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[feats])
    return df, kmeans

def analyze_cluster_strengths(df):
    """
    Analyzes and prints statistics about each cluster.
    """
    if 'cluster' in df.columns:
        print(df.groupby('cluster')['Position'].mean())
