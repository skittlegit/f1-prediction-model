from modules.modeling import train_random_forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def train_models(df):
    """
    Trains all necessary models and returns a dict of models and metrics.
    """
    features = [col for col in df.columns if col not in ['Position', 'Driver', 'Team', 'Status']]
    X = df[features]
    y = df['Position']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Race results model (regression)
    race_results_model = train_random_forest(X_train, y_train, task='regression')
    y_pred = race_results_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Tyre degradation (stub)
    tyre_model = train_random_forest(X_train, y_train, task='regression')

    # Pit stop model (stub)
    pit_stop_model = train_random_forest(X_train, y_train, task='regression')

    models = {
        "race_results_model": {"model": race_results_model, "metrics": {"mse": mse}},
        "tyre_model": {"model": tyre_model, "metrics": {}},
        "pit_stop_model": {"model": pit_stop_model, "metrics": {}}
    }
    return models

def predict(model, X):
    """Predicts using the provided model and feature set."""
    return model.predict(X)
