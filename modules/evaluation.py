from sklearn.metrics import accuracy_score, mean_squared_error

def evaluate_model(model, X_test, y_test, task='regression'):
    """
    Evaluates the model on test data.
    """
    y_pred = model.predict(X_test)
    if task == 'classification':
        return {"accuracy": accuracy_score(y_test, y_pred)}
    else:
        return {"mse": mean_squared_error(y_test, y_pred)}
