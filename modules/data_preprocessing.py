import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Cleans and preprocesses the raw race DataFrame.
    Handles missing values, data types, and basic normalization.
    """
    if df.empty:
        return df
    # Drop irrelevant columns
    drop_cols = [col for col in ['Weather', 'Location', 'TrackStatus'] if col in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')
    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # Fill missing categorical with mode
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df
