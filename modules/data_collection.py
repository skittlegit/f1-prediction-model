import fastf1
import pandas as pd
import logging

def fetch_race_data(years):
    """
    Fetches and combines race data for the given list of years using FastF1.
    Returns a pandas DataFrame with all race data.
    """
    all_races = []
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
            for _, event in schedule.iterrows():
                try:
                    session = fastf1.get_session(year, event['RoundNumber'], 'R')
                    session.load()
                    df = session.laps
                    df['Year'] = year
                    df['Round'] = event['RoundNumber']
                    all_races.append(df)
                except Exception as e:
                    logging.warning(f"Failed to load session for {year} round {event['RoundNumber']}: {e}")
        except Exception as e:
            logging.error(f"Failed to get schedule for {year}: {e}")
    if all_races:
        return pd.concat(all_races, ignore_index=True)
    return pd.DataFrame()
