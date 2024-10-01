import numpy as np
from datetime import datetime
import csv
from typing import List, Dict, Tuple

def read_csv_data(file_path: str) -> Tuple[List[float], List[str]]:
    timestamps = []
    predictions = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            timestamps.append(float(row[0]))
            predictions.append(row[1])
    return timestamps, predictions

def create_sleep_state_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    sleep_states = {
        "NormalSleep": 0,
        "Hypopnea": 1,
        "Snore": 2,
        "ObstructiveApnea": 3,
        "MixedApnea": 4
    }
    id_name = {v: k for k, v in sleep_states.items()}
    return sleep_states, id_name

def create_prefix_array(sleep_state_numeric: List[int], num_states: int) -> np.ndarray:
    prefix_array = np.zeros((num_states, len(sleep_state_numeric)))
    for i, state in enumerate(sleep_state_numeric):
        if i > 0:
            prefix_array[:, i] = prefix_array[:, i-1]
        prefix_array[state, i] += 1
    return prefix_array

def query_prefix_array(prefix_array: np.ndarray, state: int, start_time: float, end_time: float, timestamps: List[float]) -> int:
    start_index = next((i for i, ts in enumerate(timestamps) if ts >= start_time), None)
    end_index = next((i for i, ts in enumerate(timestamps) if ts > end_time), None)

    if start_index is None or (end_index is not None and end_index <= start_index):
        return 0  # No valid range found

    if end_index is None:
        end_index = len(timestamps)  # Query until the end if no end_time is specified

    return prefix_array[state, end_index - 1] - (prefix_array[state, start_index - 1] if start_index > 0 else 0)

def compute_ahi(total_apneas: int, total_hypopneas: int, total_hours: float) -> float:
    if total_hours == 0:
        return 0  # Avoid division by zero
    return (total_apneas + total_hypopneas) / total_hours

def calculate_audio_stats() -> Dict[str, float]:
    file_path = './results/audio/audio_DEMO.csv'
    timestamps, predictions = read_csv_data(file_path)
    
    sleep_states, id_name = create_sleep_state_mappings()
    sleep_state_numeric = [sleep_states[pred] for pred in predictions]
    
    prefix_array = create_prefix_array(sleep_state_numeric, len(sleep_states))
    
    total_instances = {id_name[i]: prefix_array[i, -1] for i in range(len(sleep_states))}
    
    total_apneas = total_instances.get("ObstructiveApnea", 0)
    total_hypopneas = total_instances.get("Hypopnea", 0)
    total_hours = len(timestamps) / 360  # Time stamps are 10 seconds each
    
    ahi = compute_ahi(total_apneas, total_hypopneas, total_hours)
    
    total_apnea_events = total_instances.get("ObstructiveApnea", 0)
    total_snoring_episodes = total_instances.get("Snore", 0)
    total_snoring_time = total_snoring_episodes * 10
    
    result = {
        "event_distribution": total_instances,
        "AHI": ahi,
        "total_apnea_events": total_apnea_events,
        "total_snoring_episodes": total_snoring_episodes,
        "total_snoring_time": total_snoring_time
    }
    
    return result

if __name__ == "__main__":
    print(calculate_audio_stats())
    