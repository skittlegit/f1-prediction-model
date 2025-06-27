import pandas as pd

def predict_final_standings(simulation_results):
    """
    Aggregates simulation results to predict final driver and team standings.
    """
    avg_positions = simulation_results.mean(axis=0)
    driver_standings = pd.Series(avg_positions).sort_values().reset_index()
    team_standings = driver_standings.copy() # Stub, group by team if teams are present
    return driver_standings, team_standings
