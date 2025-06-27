# F1 Race Result Prediction Pipeline

This repository contains a modular end-to-end machine learning pipeline for predicting Formula 1 race results, championship standings, and related metrics using multi-year data, circuit stats, and real-time FastF1 API augmentation.

## Features

- Data collection and integration from 2018–2025 (FastF1 + manual sources)
- Feature engineering: ELO ratings, circuit/weather stats, DNF, pitstops, penalties, upgrades, clustering, etc.
- Modular ML models for qualifying, race results, DNF, pitstops, penalties, and more (RandomForest/XGBoost)
- Championship and points prediction, including Monte Carlo simulations
- Visualization of results, standings, and driver/team metrics
- Automated reporting and robust error handling
- Ready for batch or real-time deployment

## Repository Structure

- `main.py` — Main entry point for full-season analysis and predictions
- `race_main.py` — Script for predicting a single race (by race number)
- `modules/` — All feature, model, data, and utility code (see below)
- `requirements.txt` — Python dependencies
- `README.md` — This file

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download/setup FastF1 cache** if you want rapid repeated access.

3. **Run the main pipeline:**
   ```bash
   python main.py
   ```

   Or, to predict a specific race:
   ```bash
   python race_main.py
   ```

4. **Check outputs and logs** for results, predictions, and visualizations.

## Notes

- All main logic is in the `modules/` folder.
- You can easily extend the pipeline with new features or models (see `modules/` for entry points).
- See `main.py` for the full workflow and how modules are orchestrated.

---

## License

MIT License
