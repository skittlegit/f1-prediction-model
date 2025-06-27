import numpy as np

def monte_carlo_simulation(model, X, n_simulations=1000):
    """
    Runs Monte Carlo simulations for race outcomes using the given model.
    """
    results = []
    for _ in range(n_simulations):
        pred = model.predict(X)
        results.append(pred)
    return np.array(results)
