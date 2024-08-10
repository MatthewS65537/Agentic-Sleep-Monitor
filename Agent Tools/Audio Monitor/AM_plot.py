import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

def visualize_csv_data(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert the 'audio_time' column to datetime
    df['audio_time'] = pd.to_datetime(df['audio_time'])

    # Create a time range covering the entire period
    time_range = pd.date_range(start=df['audio_time'].min(), end=df['audio_time'].max(), freq='T')

    # Create a binary matrix for the presence of each label over time
    labels = df['label'].unique()
    label_matrix = np.zeros((len(labels), len(df)))

    for i, label in enumerate(labels):
        label_times = df[df['label'] == label]['audio_time']
        label_indices = np.searchsorted(df['audio_time'], label_times)
        label_indices = label_indices[label_indices < len(df)]  # Ensure indices are within bounds
        label_matrix[i, label_indices] = 1

    # Plot the binary matrix as an image
    plt.figure(figsize=(14, 8))
    plt.imshow(label_matrix, aspect='auto', cmap='gray_r', interpolation='none')
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xticks(ticks=np.arange(0, len(df), max(1, len(df)//10)),
               labels=[f'Reading {i+1}' for i in range(0, len(df), max(1, len(df)//10))],
               rotation=45)
    plt.xlabel('Reading Index')
    plt.ylabel('Labels')
    plt.title('Labels Presence Over Time')
    plt.tight_layout()
    plt.savefig('./plots/labels_presence_over_time.png')
    # plt.show()

if __name__ == "__main__":
    csv_file = '../../data/audio_classification_log.csv'
    while True:
        time.sleep(10)
        visualize_csv_data(csv_file)