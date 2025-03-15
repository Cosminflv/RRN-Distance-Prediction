import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

def find_point_index_to_predict(elapsed_time, pred_sec_ahead):
    for i in range(len(elapsed_time)):
        if elapsed_time[i] >= pred_sec_ahead:
            return i
    return -1

# ------------------------- Main Execution -------------------------
def main():
    # Load and parse GPX files for train, test, and validation sets
    train_files = load_gpx_files('gpx_data/train')
    test_files = load_gpx_files('gpx_data/test')
    val_files = load_gpx_files('gpx_data/val')

    train_df_raw, test_df_raw, val_df_raw = map(parse_gpx_data, [train_files, test_files, val_files])


    train_data = np.array(train_df_raw[['latitude', 'longitude', 'elevation', 'elapsed_time', 'time', 'source_file' ]]).tolist()

    # Precompute cumulative distances for training data
    cumulative_distances = []
    current_source = None
    cumulative_dist = 0.0
    prev_lat, prev_lon = None, None

    for i, (lat, lon, _, _, _, s_file) in enumerate(train_data):
        if i == 0 or s_file != current_source:
            # Reset for new track
            current_source = s_file
            cumulative_dist = 0.0
            prev_lat, prev_lon = lat, lon
        else:
            # Accumulate distance within the same track
            dist = haversine(prev_lat, prev_lon, lat, lon)
            cumulative_dist += dist
            prev_lat, prev_lon = lat, lon
        cumulative_distances.append(cumulative_dist)

    # Find max cumulative distance in training data for scaling
    max_cum_dist = max(cumulative_distances)

    X_train = []
    Y_train = []

    seq_lenght = 50
    pred_sec_ahead = 142
    temp_list = []
    init_ts = train_data[0][4] # Timestamp for first point
    point_times = [point[3] for point in train_data]
    point_to_pred_pos = find_point_index_to_predict(point_times, pred_sec_ahead)
    for i, (lat, lon, elv, elapsed_time, timestamp, s_file) in enumerate(train_data):
        scaled_cum_dist = cumulative_distances[i] / max_cum_dist  # [0, 1]
        elapsed_time_seq = time_difference(init_ts, timestamp)
        elapsed_time_next_point = time_difference(init_ts, train_data[i+point_to_pred_pos][4] if i+point_to_pred_pos < len(train_data) else train_data[i][4])
        elapsed_time_seq_scaled = elapsed_time_seq / elapsed_time_next_point # [0, 1]
        temp_list.append([(lat+90)/180, (lon+180)/360, elv/8000, scaled_cum_dist, elapsed_time_seq_scaled])
        if  i+point_to_pred_pos>=len(train_data):
            break

        if(len(temp_list) == seq_lenght):
            lat2, lon2 = train_data[i+point_to_pred_pos][0], train_data[i+point_to_pred_pos][1] if i+point_to_pred_pos < len(train_data) else (-1,-1)
            if lat2 == -1:
                break

            if haversine(lat, lon, lat2, lon2) != 0:        
                X_train.append(temp_list)             
                Y_train.append(haversine(lat, lon, lat2, lon2))
            temp_list = []
            init_ts = train_data[i+1][4]


        


                    # "latitude": lat,
                    # "longitude": lon,
                    # "elevation": float(ele.text) if ele is not None else None,
                    # "time": time.text if time is not None else None,
                    # "source_file": gpx_file

    # # Preprocess training data (without normalization) to compute necessary columns
    # train_df_pre = pre_process_for_fitting(train_df_raw)

    # # ----------------- Pre-Fit Scalers on Training Data -----------------
    # coordinates_scaler = StandardScaler().fit(train_df_pre[['latitude', 'longitude']])
    # elevation_scaler = StandardScaler().fit(train_df_pre[['elevation']])
    # time_scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_df_pre[['time_seconds']])
    # # For distance_scaler, you may want to compute training distances first or decide on a reasonable range.
    # # Here, we assume it is pre-fitted or you can leave it unfitted if you later fit it on computed training distances.
    # pred_distance_scaler = MinMaxScaler(feature_range=(0, 1))
    # cum_distance_scaler = MinMaxScaler(feature_range=(0, 1))


    # # Process training data with pre-fitted scalers.
    # processor_train, train_df_norm = process_data(train_df_raw, 
    #                                               coordinates_scaler, 
    #                                               elevation_scaler, 
    #                                               time_scaler, 
    #                                               pred_distance_scaler,
    #                                               cum_distance_scaler)
    # # For test and validation data, use the same pre-fitted scalers.
    # processor_test, test_df_norm = process_data(test_df_raw, 
    #                                               coordinates_scaler, 
    #                                               elevation_scaler, 
    #                                               time_scaler, 
    #                                               pred_distance_scaler,
    #                                               cum_distance_scaler)
    # processor_val, val_df_norm = process_data(val_df_raw, 
    #                                               coordinates_scaler, 
    #                                               elevation_scaler, 
    #                                               time_scaler, 
    #                                               pred_distance_scaler,
    #                                               cum_distance_scaler)

    # Create sequences for training, test, and validation sets.
    # X_train, y_train = create_sequences(processor_train, train_df_norm, sequence_length=50, is_train=True)
    # X_test, y_test = create_sequences(processor_test, test_df_norm, sequence_length=50, is_train=False)
    # X_val, y_val = create_sequences(processor_val, val_df_norm, sequence_length=50, is_train=False)

    # Shuffle the data
    # X_train, y_train = shuffle_data(X_train, Y_train)
    # X_test, y_test = shuffle_data(X_test, y_test)
    # X_val, y_val = shuffle_data(X_val, y_val)

    # print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ------------------------- Model Training -------------------------
    tracker = RNNTracker(input_shape=(50, 5))  # (sequence_length=50, features=5)
    tracker.compile(loss='mse', metrics=['accuracy'])
    tracker.summary()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    scale_factor = np.max(Y_train)
    Y_train = Y_train / scale_factor


    history = tracker.train(X_train, Y_train, epochs=100)#, validation_data=(X_val, y_val))

    # print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # ------------------------- Model Evaluation & Visualization -------------------------
    tracker.history = history
    tracker.plot_training_curves(metric='loss')
    tracker.plot_training_curves(metric='accuracy')
    tracker.plot_actual_vs_predicted(X_train, Y_train, scale_factor)
    x = 3

if __name__ == "__main__":
    main()
