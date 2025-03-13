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

# ------------------------- Main Execution -------------------------
def main():
    # Load and parse GPX files for train, test, and validation sets
    train_files = load_gpx_files('gpx_data/train')
    test_files = load_gpx_files('gpx_data/test')
    val_files = load_gpx_files('gpx_data/val')

    train_df_raw, test_df_raw, val_df_raw = map(parse_gpx_data, [train_files, test_files, val_files])

    # Preprocess training data (without normalization) to compute necessary columns
    train_df_pre = pre_process_for_fitting(train_df_raw)

    # ----------------- Pre-Fit Scalers on Training Data -----------------
    coordinates_scaler = StandardScaler().fit(train_df_pre[['latitude', 'longitude']])
    elevation_scaler = StandardScaler().fit(train_df_pre[['elevation']])
    time_scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_df_pre[['time_seconds']])
    # For distance_scaler, you may want to compute training distances first or decide on a reasonable range.
    # Here, we assume it is pre-fitted or you can leave it unfitted if you later fit it on computed training distances.
    pred_distance_scaler = MinMaxScaler(feature_range=(0, 1))
    cum_distance_scaler = MinMaxScaler(feature_range=(0, 1))


    # Process training data with pre-fitted scalers.
    processor_train, train_df_norm = process_data(train_df_raw, 
                                                  coordinates_scaler, 
                                                  elevation_scaler, 
                                                  time_scaler, 
                                                  pred_distance_scaler,
                                                  cum_distance_scaler)
    # For test and validation data, use the same pre-fitted scalers.
    processor_test, test_df_norm = process_data(test_df_raw, 
                                                  coordinates_scaler, 
                                                  elevation_scaler, 
                                                  time_scaler, 
                                                  pred_distance_scaler,
                                                  cum_distance_scaler)
    processor_val, val_df_norm = process_data(val_df_raw, 
                                                  coordinates_scaler, 
                                                  elevation_scaler, 
                                                  time_scaler, 
                                                  pred_distance_scaler,
                                                  cum_distance_scaler)

    # Create sequences for training, test, and validation sets.
    X_train, y_train = create_sequences(processor_train, train_df_norm, sequence_length=50, is_train=True)
    X_test, y_test = create_sequences(processor_test, test_df_norm, sequence_length=50, is_train=False)
    X_val, y_val = create_sequences(processor_val, val_df_norm, sequence_length=50, is_train=False)

    # Shuffle the data
    X_train, y_train = shuffle_data(X_train, y_train)
    X_test, y_test = shuffle_data(X_test, y_test)
    X_val, y_val = shuffle_data(X_val, y_val)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ------------------------- Model Training -------------------------
    tracker = RNNTracker(input_shape=(50, 5))  # (sequence_length=50, features=5)
    tracker.compile(loss='mse', metrics=['mae'])
    tracker.summary()

    history = tracker.train(X_train, y_train, epochs=30, validation_data=(X_val, y_val))
    loss, accuracy = tracker.evaluate(X_test, y_test, batch_size=64)

    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # ------------------------- Model Evaluation & Visualization -------------------------
    tracker.history = history
    tracker.plot_training_curves(metric='loss')
    tracker.plot_training_curves(metric='mae')
    tracker.plot_actual_vs_predicted(X_test, y_test)
    x = 3

if __name__ == "__main__":
    main()
