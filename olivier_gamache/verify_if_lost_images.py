import pandas as pd
import numpy as np

def closest_exposure(exposure, sequence):
    closest = min(enumerate(sequence), key=lambda x: abs(x[1] - exposure))
    return closest[1], closest[0]

def verify_exposure_sequence(file_path, sequence):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Initialize the break counter
    break_nbr = []
    
    for i in range(1, len(df) - 1):
        # prev_exposure = df['ExposureTime'].iloc[i - 1]
        prev_exposure, prev_index = closest_exposure(df['ExposureTime'].iloc[i - 1], sequence)
        current_exposure, current_index = closest_exposure(df['ExposureTime'].iloc[i], sequence)
        next_exposure, next_index = closest_exposure(df['ExposureTime'].iloc[i + 1], sequence)
        timestamp = df['timestamp'].iloc[i]
        
        print(f"Previous: {prev_exposure}, Current: {current_exposure}, Next: {next_exposure}")
        
        # Check if the current exposure is within the buffer range of the previous and next exposures
        if current_index == 0:
            if prev_index != len(sequence) - 1  or abs(next_index - current_index) > 1:
                break_nbr.append(timestamp)
        elif current_index == len(sequence) - 1:
            if abs(current_index - prev_index) > 1  or next_index != 0:
                break_nbr.append(timestamp)
        elif abs(current_index - prev_index) > 1 or abs(next_index - current_index) > 1:
            break_nbr.append(timestamp)
    
    print("------------------------------------------")
    print(f"Number of break sequences: {len(break_nbr)}")
    print(f"Breaks at indexes: {break_nbr}")

def main():
    experiment_name = 'backpack_2024-09-24-17-17-09'
    camera_nbr = 1
    exposure_sequence = [0.2, 1, 5]

    file_path = f'/home/olivier_g/Desktop/tmp/{experiment_name}/camera{camera_nbr}/camera{camera_nbr}.csv'
    verify_exposure_sequence(file_path, exposure_sequence)

if __name__ == '__main__':
    main()