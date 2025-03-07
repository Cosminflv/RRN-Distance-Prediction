import os
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
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
def process_data(df, scaler=None):
    """Processes raw GPX data and returns the normalized DataFrame."""
    processor = GPXDataProcessor(df, scaler)
    return processor, processor.process_data()


def create_sequences(processor, df, sequence_length=15):
    """Creates sequences for model training/testing."""
    return processor.create_sequences(df, sequence_length)


def shuffle_data(X, y, seed=42):
    """Shuffles the sequences while keeping X and y aligned."""
    np.random.seed(seed)
    data = list(zip(X, y))
    np.random.shuffle(data)
    return map(np.array, zip(*data))


# ------------------------- Main Execution -------------------------
def main():
    # Load and parse GPX files
    train_files = load_gpx_files('gpx_data/train')
    test_files = load_gpx_files('gpx_data/test')
    val_files = load_gpx_files('gpx_data/val')

    train_df, test_df, val_df = map(parse_gpx_data, [train_files, test_files, val_files])

    # Initialize shared scaler
    shared_scaler = StandardScaler()

    # Process data with shared scaler
    processor_train, train_df_norm = process_data(train_df, shared_scaler)
    shared_scaler = processor_train.scaler  # Save scaler after first processing

    processor_test, test_df_norm = process_data(test_df, shared_scaler)
    processor_val, val_df_norm = process_data(val_df, shared_scaler)

    # Create sequences
    X_train, y_train = create_sequences(processor_train, train_df_norm)
    X_test, y_test = create_sequences(processor_test, test_df_norm)
    X_val, y_val = create_sequences(processor_val, val_df_norm)

    # Shuffle data
    X_train, y_train = shuffle_data(X_train, y_train)
    X_test, y_test = shuffle_data(X_test, y_test)
    X_val, y_val = shuffle_data(X_val, y_val)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ------------------------- Model Training -------------------------
    tracker = RNNTracker(input_shape=(15, 4))  # Assuming (sequence_length=15, features=4)
    tracker.compile(loss='mse', metrics=['mae'])
    tracker.summary()

    history = tracker.train(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    loss, accuracy = tracker.evaluate(X_test, y_test, batch_size=64)

    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # ------------------------- Model Evaluation & Visualization -------------------------
    tracker.history = history
    tracker.plot_training_curves(metric='loss')
    tracker.plot_training_curves(metric='mae')

    tracker.plot_actual_vs_predicted(X_test, y_test)

    x = 3

    # tracker.plot_actual_vs_predicted_coords(X_test, y_test, sample_points=200)
    # tracker.plot_time_scatter(X_test, y_test)
    # tracker.plot_error_distributions(X_test, y_test)
    # tracker.plot_residual_autocorrelation(X_test, y_test)
    # tracker.visualize_model_architecture()


if __name__ == "__main__":
    main()
