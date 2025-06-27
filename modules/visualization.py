import matplotlib.pyplot as plt
import seaborn as sns

def visualize_season_standings(df):
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x='Round', y='Position', hue='Driver')
    plt.title('Season Standings')
    plt.ylabel('Position')
    plt.gca().invert_yaxis()
    plt.show()

def visualize_driver_performance(df):
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='Driver', y='avg_lap_time')
    plt.title('Average Lap Time by Driver')
    plt.ylabel('Average Lap Time (s)')
    plt.show()

def visualize_lap_by_lap(df, predictions=None, race_number=None, race_details=None):
    plt.figure(figsize=(12,6))
    if race_number:
        df = df[df['Round'] == race_number]
    sns.lineplot(data=df, x='LapNumber', y='Position', hue='Driver')
    plt.title(f"Lap-by-Lap Position Changes (Race {race_number})")
    plt.ylabel('Position')
    plt.gca().invert_yaxis()
    plt.show()

def visualize_pit_stops(df, race_number=None):
    if race_number:
        df = df[df['Round'] == race_number]
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x='PitStop', hue='Driver')
    plt.title(f"Pit Stops Distribution (Race {race_number})")
    plt.show()

def visualize_tire_performance(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x='Compound', y='LapTime')
    plt.title('Tyre Compound Performance')
    plt.show()

def visualize_overtaking_difficulty(df):
    plt.figure(figsize=(10,6))
    sns.histplot(df['Overtakes'], bins=20)
    plt.title('Overtaking Difficulty Distribution')
    plt.xlabel('Number of Overtakes')
    plt.show()
