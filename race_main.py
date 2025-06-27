import logging
from modules.data_collection import fetch_race_data
from modules.data_preprocessing import preprocess_data
from modules.feature_engineering import generate_features
from modules.predictive_modeling import train_models, predict
from modules.visualization import visualize_lap_by_lap
from modules.race_details import get_race_details
from modules.deployment import deploy_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("race_main.log", mode='a')
    ]
)

def race_main():
    logging.info("Starting race-specific analysis...")

    # User input for race number
    try:
        race_number = int(input("Enter the race number (e.g., 6 for Race 6): "))
        logging.info(f"Race number selected: {race_number}")
    except ValueError:
        logging.error("Invalid input for race number. Please enter an integer.")
        return

    # Step 1: Fetch data for all years (2018-2025), prioritize 2025
    logging.info("Fetching and prioritizing data...")
    years = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018]
    raw_data = fetch_race_data(years)

    if raw_data.empty:
        logging.error("No data was fetched for any year. Exiting script.")
        return

    # Step 2: Preprocess data
    logging.info("Preprocessing data...")
    preprocessed_data = preprocess_data(raw_data)

    if preprocessed_data.empty:
        logging.error("Data preprocessing resulted in an empty dataset. Exiting script.")
        return

    # Step 3: Get race-specific details from race_details.py
    logging.info("Loading race-specific details...")
    race_details = get_race_details(race_number)

    # Step 4: Generate features and labels
    logging.info("Generating features and labels...")
    enriched_data, labels = generate_features(preprocessed_data, race_details)

    if enriched_data.empty or labels is None or labels.empty:
        logging.error("Feature generation resulted in empty features or labels. Exiting script.")
        return

    # Step 5: Train predictive models and extract specific models
    logging.info("Training predictive models...")
    models = train_models(enriched_data)

    if not models:
        logging.error("Model training failed. Exiting script.")
        return

    # Extract the race results model for predictions
    race_results_model = models["race_results_model"]["model"]

    # Step 6: Make predictions for the specific race
    logging.info("Making predictions for the specific race...")
    predictions = predict(race_results_model, enriched_data)

    # Step 7: Deploy the race results model (optional for real-time or batch inference)
    logging.info("Deploying the race results model...")
    deploy_model(race_results_model, endpoint_type="real-time")

    # Step 8: Visualize lap-by-lap positional changes and pit stops
    logging.info("Visualizing race results...")
    visualize_lap_by_lap(preprocessed_data, predictions, race_number, race_details)

    logging.info("Race-specific analysis completed successfully.")

if __name__ == "__main__":
    race_main()
