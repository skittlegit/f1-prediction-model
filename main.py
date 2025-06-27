import logging
from modules.data_collection import fetch_race_data
from modules.data_preprocessing import preprocess_data
from modules.feature_engineering import generate_features
from modules.predictive_modeling import train_models, predict
from modules.evaluation import evaluate_model
from modules.clustering import cluster_drivers, analyze_cluster_strengths
from modules.visualization import (
    visualize_season_standings,
    visualize_driver_performance,
    visualize_lap_by_lap,
    visualize_pit_stops,
    visualize_tire_performance,
    visualize_overtaking_difficulty
)
from modules.strategy_optimization import optimize_strategy
from modules.race_control_analysis import predict_race_control_events
from modules.monte_carlo_simulations import monte_carlo_simulation
from modules.standings_prediction import predict_final_standings
from modules.deployment import deploy_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("main.log", mode='a')
    ]
)

def main():
    logging.info("Starting main script for season-long analysis.")

    # Step 1: Fetch data for all years (2018-2025), prioritize 2025
    years = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018]
    logging.info("Fetching data for years: %s", years)
    raw_data = fetch_race_data(years)

    if raw_data.empty:
        logging.error("No data was fetched for any year. Exiting script.")
        return

    # Step 2: Preprocess data
    logging.info("Preprocessing data.")
    preprocessed_data = preprocess_data(raw_data)

    if preprocessed_data.empty:
        logging.error("Data preprocessing resulted in an empty dataset. Exiting script.")
        return

    # Step 3: Generate features and labels
    logging.info("Generating features.")
    enriched_data, labels = generate_features(preprocessed_data)

    if enriched_data.empty or labels is None or labels.empty:
        logging.error("Feature generation resulted in empty features or labels. Exiting script.")
        return

    # Step 4: Train predictive models
    logging.info("Training predictive models.")
    models = train_models(enriched_data)

    # Log model evaluation metrics
    for model_name, model_info in models.items():
        logging.info("Evaluating model: %s", model_name)
        evaluation_metrics = model_info["metrics"]
        logging.info("Metrics for %s:\n%s", model_name, evaluation_metrics)

    # Extract specific models for simulation and optimization steps
    tyre_degradation_model = models["tyre_model"]["model"]
    pit_stop_model = models["pit_stop_model"]["model"]
    race_results_model = models["race_results_model"]["model"]

    # Step 5: Deploy models
    logging.info("Deploying models for batch and real-time inference.")
    deploy_model(tyre_degradation_model, endpoint_type="batch")
    deploy_model(pit_stop_model, endpoint_type="real-time")
    deploy_model(race_results_model, endpoint_type="batch")

    # Step 6: Cluster drivers
    logging.info("Clustering drivers based on telemetry data.")
    clustered_data, clustering_model = cluster_drivers(enriched_data)
    analyze_cluster_strengths(clustered_data)

    # Step 7: Race strategy optimization
    logging.info("Optimizing race strategy.")
    optimized_strategy = optimize_strategy(preprocessed_data, tyre_degradation_model, pit_stop_model)
    logging.info(f"Optimized Strategy: {optimized_strategy}")

    # Step 8: Predict race control events
    logging.info("Predicting race control events.")
    race_control_model = predict_race_control_events(preprocessed_data)

    # Step 9: Monte Carlo simulations for race outcomes
    logging.info("Running Monte Carlo simulations for race outcomes.")
    simulation_results = monte_carlo_simulation(race_results_model, enriched_data)

    # Step 10: Predict final driver and team standings
    logging.info("Predicting final standings.")
    driver_standings, team_standings = predict_final_standings(simulation_results)
    logging.info(f"Predicted Driver Standings:\n{driver_standings}")
    logging.info(f"Predicted Team Standings:\n{team_standings}")

    # Step 11: Generate visualizations
    logging.info("Generating visualizations.")
    visualize_season_standings(preprocessed_data)
    visualize_driver_performance(preprocessed_data)
    visualize_lap_by_lap(preprocessed_data, race_number=5)
    visualize_pit_stops(preprocessed_data, race_number=5)
    visualize_tire_performance(preprocessed_data)
    visualize_overtaking_difficulty(preprocessed_data)

    logging.info("Main script completed successfully.")

if __name__ == "__main__":
    main()
