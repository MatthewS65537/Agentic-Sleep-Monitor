import numpy as np
import csv
from datetime import datetime, timedelta

def read_sleep_data(file_path):
    sleep_data = []
    
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        
        for row in csv_reader:
            timestamp = int(row['Timestamp'])
            label = row['Label']
            
            # Convert timestamp to datetime object
            date_time = datetime.fromtimestamp(timestamp)
            
            sleep_data.append({
                'timestamp': timestamp,
                'datetime': date_time,
                'label': label
            })
    
    return sleep_data

# Usage
file_path = '../data/audio_logs/audio_classification_log_AUG13DEMO.csv'  # Replace with your actual file path
sleep_data = read_sleep_data(file_path)

import matplotlib.pyplot as plt
from collections import Counter

def create_sleep_graph(sleep_data):
    # Sort sleep_data by timestamp
    sleep_data.sort(key=lambda x: x['timestamp'])

    # Get the start and end times, rounded to the nearest hour
    start_time = sleep_data[0]['datetime'].replace(minute=0, second=0, microsecond=0)
    end_time = sleep_data[-1]['datetime'].replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    # Create hourly bins
    hourly_bins = [start_time + timedelta(hours=i) for i in range(int((end_time - start_time).total_seconds() / 3600) + 1)]

    # Extract timestamps and labels
    timestamps = [entry['datetime'] for entry in sleep_data]
    labels = [entry['label'] for entry in sleep_data]

    # Create a dictionary to count occurrences of each label per hour
    hourly_counts = {label: [0] * len(hourly_bins) for label in set(labels)}

    for timestamp, label in zip(timestamps, labels):
        hour_index = int((timestamp - start_time).total_seconds() / 3600)
        hourly_counts[label][hour_index] += 1

    # Find the first and last non-zero hours
    first_hour = min(next(i for i, count in enumerate(sum(hourly_counts.values(), [])) if count > 0), 0)
    last_hour = max(len(sum(hourly_counts.values(), [])) - next(i for i, count in enumerate(reversed(sum(hourly_counts.values(), []))) if count > 0) - 1, first_hour + 1)

    # Slice the data to focus only on hours with data
    hourly_bins = hourly_bins[first_hour:last_hour+1]
    for label in hourly_counts:
        hourly_counts[label] = hourly_counts[label][first_hour:last_hour+1]

    # Create the stacked histogram
    plt.figure(figsize=(15, 6))
    
    # Define custom colors
    color_map = {
        'NormalSleep': 'gray',
        'ObstructiveApnea': 'red',
        'Hypopnea': 'black',
        'Snore': 'blue',
        'MixedApnea': 'orange'
    }
    
    bottom = np.zeros(len(hourly_bins))
    for label, counts in hourly_counts.items():
        plt.bar(np.arange(len(hourly_bins)), counts, bottom=bottom, 
                label=label, color=color_map.get(label, 'gray'))
        bottom += counts

    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('Sleep Pattern Over Time')
    plt.legend()

    # Set x-axis ticks to show every hour
    tick_locations = range(0, len(hourly_bins))
    tick_labels = [hourly_bins[i].strftime('%Y-%m-%d %H:%M') for i in tick_locations]
    plt.xticks(tick_locations, tick_labels, rotation=45, ha='right')

    plt.tight_layout()

    # Save the graph
    plt.savefig('../data/sleep_pattern_graph.png')
    plt.close()

# Create and save the graph
create_sleep_graph(sleep_data)
print("Sleep pattern graph has been saved as 'sleep_pattern_graph.png' in the data directory.")
