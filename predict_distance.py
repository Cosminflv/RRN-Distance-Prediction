import numpy as np
import pandas as pd
from utils import haversine

from rnn_model import RNNTracker
from trackpoint import read_trackpoints_from_file

def predict_distance(trackpoints, tracker):
    """
    Processes 50 trackpoints through feature engineering and predicts the unscaled distance ahead.

    Args:
        trackpoints (list): List of 50 trackpoint dictionaries with keys:
                             'latitude', 'longitude', 'elevation', 'time'.
        tracker (RNNTracker): Trained RNNTracker model.

    Returns:
        list: Single-element list containing the predicted distance in meters.
    """
    # Constants from training configuration
    MAX_ELEV_DIFF = 14
    MAX_DISTANCE_DIFF = 14
    MAX_TIME_DIFF_SEQ = 10
    MAX_DIST = 220  # As defined during training

    # Validate input
    if len(trackpoints) != 50:
        raise ValueError("Exactly 50 trackpoints are required.")

    # Check for missing elevation or time data
    for tp in trackpoints:

        if tp.elevation is None:
            raise ValueError("All trackpoints must have elevation data.")
        if tp.time is None:
            raise ValueError("All trackpoints must have time data.")

    # Process elevation differences
    elev_features = []
    for i in range(len(trackpoints)):
        if i == 0:
            diff_elev = 0.0
        else:
            diff_elev = trackpoints[i].elevation - trackpoints[i - 1].elevation
        scaled_elev = diff_elev / MAX_ELEV_DIFF
        elev_features.append(scaled_elev)

    # Process haversine distances between consecutive points
    dist_features = []
    for i in range(len(trackpoints)):
        if i == 0:
            dist = 0.0
        else:
            prev_lat = trackpoints[i - 1].latitude
            prev_lon = trackpoints[i - 1].longitude
            curr_lat = trackpoints[i].latitude
            curr_lon = trackpoints[i].longitude
            dist = haversine(prev_lat, prev_lon, curr_lat, curr_lon)
        normalized_dist = dist / MAX_DISTANCE_DIFF
        dist_features.append(normalized_dist)

    # Process time differences
    times = [pd.to_datetime(tp.time) for tp in trackpoints]
    time_diffs = []
    for i in range(len(trackpoints)):
        if i == 0:
            diff_seconds = 0.0
        else:
            diff_seconds = (times[i] - times[i - 1]).total_seconds()
        normalized_time = diff_seconds / MAX_TIME_DIFF_SEQ
        time_diffs.append(normalized_time)

    # Combine features into sequence (shape: 50 timesteps, 3 features)
    input_sequence = np.array([
        [elev, dist, time]
        for elev, dist, time in zip(elev_features, dist_features, time_diffs)
    ]).reshape(1, 50, 3)  # Reshape for model input (batch_size=1)

    # Predict and unscale
    scaled_prediction = tracker.predict(input_sequence)[0][0]
    unscaled_prediction = scaled_prediction * MAX_DIST

    return [unscaled_prediction]

tracker = RNNTracker.load("model.keras")
tracker.model.load_weights("model.keras")
tracker.summary()
trackpoints = read_trackpoints_from_file("valid_sequences/Activity August 02, 2020_seq99.json")
prediction = predict_distance(trackpoints, tracker)
print(f"Predicted distance: {prediction[0]} meters")