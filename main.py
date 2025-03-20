from collections import defaultdict
import os
import numpy as np
import pandas as pd
from data_processor import GPXDataProcessor
from gpx_data_parser import GPXParser
from rnn_model import RNNTracker

# ------------------------- Data Loading & Parsing -------------------------
def load_gpx_files(directory):
    """Returns a list of GPX file paths from the given directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.gpx')]

def parse_gpx_data(gpx_files):
    """Parses GPX files and returns a DataFrame."""
    parser = GPXParser(gpx_files)
    return parser.get_dataframe()

# ------------------------- Data Processing -------------------------
def pre_process_for_fitting(df):
    """
    Performs preprocessing steps that do not involve normalization,
    so that scalers can be fitted on the training data.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    # Compute time_seconds
    df["time_seconds"] = df.groupby("source_file")["time"].transform(
        lambda x: (x - x.min()).dt.total_seconds()
    )
    # Handle missing elevation values
    df['elevation'] = df.groupby('source_file')['elevation'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill')
    )
    df['elevation'] = df['elevation'].fillna(0)
    return df

def process_data(df, coords_scaler, elevation_scaler, time_scaler, pred_distance_scaler, cum_distance_scaler):
    """Processes raw GPX data using pre-fitted scalers and returns the normalized DataFrame."""
    processor = GPXDataProcessor(df, coords_scaler, elevation_scaler, time_scaler, pred_distance_scaler, cum_distance_scaler)
    return processor, processor.process_data()

def create_sequences(processor, df, sequence_length=15, is_train=False):
    """Creates sequences for model training/testing."""
    return processor.create_sequences(df, sequence_length, is_train)

def shuffle_data(X, y, seed=42):
    """Shuffles the sequences while keeping X and y aligned."""
    np.random.seed(seed)
    data = list(zip(X, y))
    np.random.shuffle(data)
    return map(np.array, zip(*data))

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in meters between two geographic points"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * 6371 * 1000 * np.arcsin(np.sqrt(a))  # Meters

from datetime import datetime

def time_difference(timestamp1, timestamp2):
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    dt1 = datetime.strptime(timestamp1, fmt)
    dt2 = datetime.strptime(timestamp2, fmt)
    return abs((dt2 - dt1).total_seconds())

def convert_to_float(obj):
    if isinstance(obj, list):
        return [convert_to_float(item) for item in obj]
    return float(obj)  # Convert individual elements

def find_point_index_to_predict(elapsed_time, pred_sec_ahead):
    for i in range(len(elapsed_time)):
        if elapsed_time[i] >= pred_sec_ahead:
            return i
    return -1

def compute_sequence_distances(seq):
    """
    Given a sequence (list) of points, where each point is expected to have the 
    structure [lat, lon, ...], compute the haversine distance between consecutive points.
    """
    distances = [0]
    for i in range(1, len(seq)):
        lat1, lon1 = seq[i-1][0], seq[i-1][1]
        lat2, lon2 = seq[i][0], seq[i][1]
        distances.append(haversine(lat1, lon1, lat2, lon2))
    return distances

# ------------------------- Main Execution -------------------------
def main():
    # Load and parse GPX files for train, test, and validation sets
    train_files = load_gpx_files('gpx_data/train')
    test_files = load_gpx_files('gpx_data/test')
    val_files = load_gpx_files('gpx_data/val')

    train_df_raw, test_df_raw, val_df_raw = map(parse_gpx_data, [train_files, test_files, val_files])

    train_data = np.array(train_df_raw[['latitude', 'longitude', 'elevation', 'elapsed_time', 'time', 'source_file']]).tolist()

    X_train = []
    Y_train = []

    seq_length = 50
    pred_sec_ahead = 142  # 2 hours ahead

    # Group training data by source_file
    from collections import defaultdict
    file_groups = defaultdict(list)
    for point in train_data:
        source_file = point[5]
        file_groups[source_file].append(point)

    for source_file, points in file_groups.items():
        temp_list = []
        if not points:
            continue
        init_ts = points[0][4]  # Initial timestamp for this file
        point_times = [p[3] for p in points]
        point_to_pred_pos = find_point_index_to_predict(point_times, pred_sec_ahead)
        if point_to_pred_pos is None:
            continue  # Skip if no valid prediction position

        for i, (lat, lon, elv, elapsed_time, timestamp, s_file) in enumerate(points):
            if i + seq_length + point_to_pred_pos >= len(points):
                break  # Not enough points ahead in this file

            if len(temp_list) == 0:
                # Calculate elapsed_time_next_point for the new sequence
                elapsed_time_next_point = time_difference(init_ts, points[i + seq_length + point_to_pred_pos][4])

            elapsed_time_seq = time_difference(init_ts, timestamp)
            elapsed_time_seq_scaled = elapsed_time_seq / elapsed_time_next_point
            temp_list.append([elv/8000, elapsed_time_seq_scaled])

            if len(temp_list) == seq_length:

                lat2, lon2 = points[i + point_to_pred_pos][0], points[i + point_to_pred_pos][1]

                # Generate sequence and calculate distances
                sequence = points[i - seq_length + 1 : i + 1]
                lat_lon_sequence = [(p[0], p[1]) for p in sequence]
                sequence_distances = compute_sequence_distances(lat_lon_sequence)
                max_distance = max(sequence_distances) if sequence_distances else 1
                normalized_distances = sequence_distances / max_distance

                augmented_temp_list = [point + [norm_d] for point, norm_d in zip(temp_list, normalized_distances)]

                if haversine(lat, lon, lat2, lon2) != 0:
                    X_train.append(augmented_temp_list)
                    Y_train.append(haversine(lat, lon, lat2, lon2))

                # Reset for next sequence
                temp_list = []
                init_ts = points[i + 1][4] if i + 1 < len(points) else timestamp


    # ------------------------- Model Loading -------------------------

    # tracker = RNNTracker.load("model.keras")
    # tracker.summary()
    # X_train = convert_to_float(X_train)
    # X_train = np.array(X_train, dtype=np.float64) 
    # print("Model expects input shape:", tracker.model.input_shape)
    # print("X_train shape:", X_train.shape)
    # scale_factor = np.max(Y_train)
    # Y_train = convert_to_float(Y_train)
    # Y_train = np.array(Y_train, dtype=np.float64)
    # Y_train = Y_train / scale_factor

    # tracker.plot_actual_vs_predicted_unscaled(X_train, Y_train, scale_factor)
    

    # ------------------------- Model Training -------------------------
    
    tracker = RNNTracker(input_shape=(50, 3))  # (sequence_length=50, features=5)
    tracker.compile(loss='mse', metrics=['accuracy'])
    tracker.summary()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_train, Y_train = shuffle_data(X_train, Y_train)
    scale_factor = np.max(Y_train)
    Y_train = Y_train / scale_factor


    history = tracker.train(X_train, Y_train, epochs=100)#, validation_data=(X_val, y_val))

    # # print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # # ------------------------- Model Evaluation & Visualization -------------------------
    tracker.history = history
    # tracker.plot_training_curves(metric='loss')
    # tracker.plot_training_curves(metric='accuracy')
    tracker.plot_actual_vs_predicted_unscaled(X_train, Y_train, scale_factor)

    # tracker.model.save("model.keras")
    x = 3

if __name__ == "__main__":
    main()
