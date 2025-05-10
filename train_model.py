import os
import numpy as np

from gpx_data_parser import GPXParser
from rnn_model import RNNTracker
from utils import haversine


# ------------------------- Data Loading & Parsing -------------------------
def load_gpx_files(directory):
    """Returns a list of GPX file paths from the given directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.gpx')]

def parse_gpx_data(gpx_files):
    """Parses GPX files and returns a DataFrame."""
    parser = GPXParser(gpx_files)
    return parser.get_dataframe()

# ------------------------- Data Processing -------------------------

def shuffle_data(X, y, seed=42):
    """Shuffles the sequences while keeping X and y aligned."""
    np.random.seed(seed)
    data = list(zip(X, y))
    np.random.shuffle(data)
    return map(np.array, zip(*data))

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

def compute_sequence_time_diffs(seq):
    time_diffs = [0]
    for i in range(1, len(seq)):
        time_diffs.append(seq[i] - seq[i - 1])
    
    return time_diffs

# 14
# 10
MAX_DISTANCE_DIFF = 14 # in meters
MAX_TIME_DIFF_SEQ = 10  # in seconds
MAX_DIST = 4000 # gpt estimated 4000m in one hour
MAX_ELEV_DIFF = 2800 # Max elevation change (2800) for 4000m distance (steep mountain trail) 45 degreess slope

# ------------------------- Main Execution -------------------------
def main():
    seq_skips = 0

    # Load and parse GPX files for train, test, and validation sets
    train_files = load_gpx_files('gpx_data/train')
    test_files = load_gpx_files('gpx_data/test')
    val_files = load_gpx_files('gpx_data/val')

    train_df_raw, test_df_raw, val_df_raw = map(parse_gpx_data, [train_files, test_files, val_files])

    train_data = np.array(train_df_raw[['latitude', 'longitude', 'elevation', 'elapsed_time', 'time', 'source_file']]).tolist()

    X_train = []
    Y_train = []

    seq_length = 50
    pred_sec_ahead = 3600  # 1 hour ahead

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
        point_times = [p[3] for p in points]
        point_to_pred_pos = find_point_index_to_predict(point_times, pred_sec_ahead)
        if point_to_pred_pos is None:
            continue  # Skip if no valid prediction position

        for i, (lat, lon, elv, elapsed_time, timestamp, s_file) in enumerate(points):
            if i + seq_length + point_to_pred_pos >= len(points):
                break  # Not enough points ahead in this file

                # Calculate elevation difference
            if i == 0:
                diff_elev = 0.0  # First point has no previous elevation
            else:
                diff_elev = elv - points[i-1][2]  # Current - previous elevation

            # Scale to [-1, 1] using maximum guessed elevation difference

            scaled_diff = diff_elev / MAX_ELEV_DIFF


            temp_list.append([scaled_diff])

            if len(temp_list) == seq_length:
                lat2, lon2 = points[i + point_to_pred_pos][0], points[i + point_to_pred_pos][1]

                # Generate sequence and calculate distances
                sequence = points[i - seq_length + 1 : i + 1]
                lat_lon_sequence = [(p[0], p[1]) for p in sequence]
                elapsed_time_sequence = [(p[3]) for p in sequence]

                sequence_distances = compute_sequence_distances(lat_lon_sequence)
                sequence_time_diffs = compute_sequence_time_diffs(elapsed_time_sequence)

                # --- Validation Check 1 ---
                if any(distance > MAX_DISTANCE_DIFF for distance in sequence_distances) or any(
                        time_diff > MAX_TIME_DIFF_SEQ for time_diff in sequence_time_diffs):
                    temp_list = []
                    seq_skips += 1
                    print("Skipped", seq_skips, "sequences")
                    continue  # Skip this sequence

                # --- Validation Check 2 ---
                dist_to_y_label = haversine(lat, lon, lat2, lon2)
                if dist_to_y_label > MAX_DIST:
                    temp_list = []
                    seq_skips += 1
                    print("Skipped", seq_skips, "sequences")
                    continue

                # --- ONLY WRITE TO FILE IF VALIDATION PASSED ---
                # Convert raw points to TrackPoint objects
                # trackpoint_sequence = []
                # for p in sequence:
                #     tp = TrackPoint(
                #         latitude=p[0],
                #         longitude=p[1],
                #         elevation=None if p[2] == -9999 else p[2],  # Handle placeholder
                #         time=pd.to_datetime(p[4]))  # Convert string timestamp
                #     trackpoint_sequence.append(tp)
                #
                #     # Create filename with source file and sequence index
                #     source_file_stem = os.path.splitext(os.path.basename(p[5]))[0]  # p[5] is source_file
                #     filename = f"valid_sequences/{source_file_stem}_seq{i}.json"
                #
                #     # Ensure directory exists
                #     os.makedirs("valid_sequences", exist_ok=True)
                #
                #     # Write to file
                #     write_trackpoints_to_file(trackpoint_sequence, filename)

                # --- Continue with processing ---
                normalized_distances = [x / MAX_DISTANCE_DIFF for x in sequence_distances]
                normalized_time_diffs = [x / MAX_TIME_DIFF_SEQ for x in sequence_time_diffs]

                augmented_temp_list = [point + [norm_d] for point, norm_d in zip(temp_list, normalized_distances)]
                augmented_temp_list = [point + [time_diffs] for point, time_diffs in
                                       zip(augmented_temp_list, normalized_time_diffs)]

                if haversine(lat, lon, lat2, lon2) != 0:
                    X_train.append(augmented_temp_list)
                Y_train.append(dist_to_y_label)

                # Reset for next sequence
                temp_list = []




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
    
    tracker = RNNTracker(input_shape=(50, 3))  # (sequence_length=50, features=3)
    tracker.compile(loss='mse', metrics=['accuracy'])
    tracker.summary()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_train, Y_train = shuffle_data(X_train, Y_train)
    Y_train = Y_train / MAX_DIST


    history = tracker.train(X_train, Y_train, epochs=100)#, validation_data=(X_val, y_val))

    # # print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # # ------------------------- Model Evaluation & Visualization -------------------------
    tracker.history = history
    tracker.plot_training_curves(metric='loss')
    tracker.plot_training_curves(metric='accuracy')
    tracker.plot_actual_vs_predicted_unscaled(X_train, Y_train, MAX_DIST)

    # tracker.model.save("model.keras")
    x = 3


if __name__ == "__main__":
    main()
