from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def train_random_forest(X, y, task='classification'):
    """
    Trains a RandomForest model for classification or regression.
    Returns the fitted model.
    """
    if task == 'regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
