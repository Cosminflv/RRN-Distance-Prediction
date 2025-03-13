import numpy as np
import pandas as pd

time_to_predict_position = 4  # Predict position 4 seconds ahead

class GPXDataProcessor:
    def __init__(self, dataframe, coords_scaler, elevation_scaler, time_scaler, distance_scaler_pred, distance_scaler_cum):
        self.df = dataframe.copy()
        self.coords_scaler = coords_scaler  # For data normalization
        self.elevation_scaler = elevation_scaler
        self.time_scaler = time_scaler  # For time normalization
        self.distance_scaler_pred = distance_scaler_pred  # For predicted distance normalization
        self.distance_scaler_cum = distance_scaler_cum

    def process_data(self):
        """Process data: handle missing elevation, normalize features, and calculate time features."""
        self._convert_time()
        self._calculate_time_diff()
        self._handle_missing_elevation()
        self._normalize_features() 
        return self.df
    
    def create_sequences(self, processed_df, sequence_length=5, is_train=True):
        """
        Create sequences with elevation and cumulative distance included in input features.

        Returns:
             X: Input sequences (n_samples, sequence_length, 5)  # Note: now 5 features per time step
             y: Target distances in meters (n_samples,)
        """
        X, y = [], []
        all_distances = []  # To collect distances for scaling

        for track_id, track in processed_df.groupby('source_file'):
            orig_lat = track['latitude'].values
            orig_lon = track['longitude'].values
            orig_time = track['time_seconds'].values
            # Existing normalized features
            features = track[['latitude_norm', 'longitude_norm', 'elevation_norm', 'time_seconds_norm']].values

            # Compute cumulative distance from the first point along the track.
            cumulative_distance = np.zeros(len(track))
            for idx in range(1, len(track)):
                cumulative_distance[idx] = cumulative_distance[idx-1] + self._haversine(
                    orig_lon[idx-1], orig_lat[idx-1],
                    orig_lon[idx], orig_lat[idx]
                )
            
            if is_train:
                # Fit the cumulative distance scaler on training data
                self.distance_scaler_cum.fit(cumulative_distance.reshape(-1, 1))
            
            # Normalize cumulative distance
            scaled_cumulative_distance = self.distance_scaler_cum.transform(cumulative_distance.reshape(-1, 1))

            # Add the cumulative distance as a new feature column.
            # new_features will have shape (n_points, 5)
            new_features = np.hstack((features, scaled_cumulative_distance))

            # Loop to create sequences for this track
            for i in range(sequence_length, len(new_features)):
                last_idx = i - 1
                target_time = orig_time[last_idx] + time_to_predict_position

                # Find the first point after target_time
                mask = orig_time[i:] >= target_time
                if not mask.any():
                    continue
                j = i + np.argmax(mask)

                # Calculate Haversine distance as target output
                distance = self._haversine(
                    orig_lon[last_idx], orig_lat[last_idx],
                    orig_lon[j], orig_lat[j]
                )

                # Append the sequence of features and target distance
                X.append(new_features[i-sequence_length:i])
                all_distances.append(distance)

        # Normalize all target distances at once
        all_distances = np.array(all_distances).reshape(-1, 1)
        if is_train:
            self.distance_scaler_pred.fit(all_distances)
        y_normalized = self.distance_scaler_pred.transform(all_distances)
        y = y_normalized.flatten()

        return np.array(X), y

    @staticmethod
    def _haversine(lon1, lat1, lon2, lat2):
        """Calculate distance in meters between two geographic points"""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * 6371 * 1000 * np.arcsin(np.sqrt(a))  # Meters

    def _handle_missing_elevation(self):
        """Fill missing elevation values within each track."""
        self.df['elevation'] = self.df.groupby('source_file')['elevation'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        self.df['elevation'] = self.df['elevation'].fillna(0)

    def _normalize_features(self):
        """Normalize latitude, longitude, and elevation."""

        coordinates_features = ["latitude", "longitude"]
        scaled_coords = self.coords_scaler.transform(self.df[coordinates_features])
        scaled_elevation = self.elevation_scaler.transform(self.df[["elevation"]])
        scaled_time = self.time_scaler.transform(self.df[["time_seconds"]])
        self.df["latitude_norm"] = scaled_coords[:, 0]
        self.df["longitude_norm"] = scaled_coords[:, 1]
        self.df["elevation_norm"] = scaled_elevation[:, 0]
        self.df["time_seconds_norm"] = scaled_time[:, 0]  # New normalized column

    def _convert_time(self):
        """Convert time to seconds since the start of each track."""
        self.df["time"] = pd.to_datetime(self.df["time"])
        self.df["time_seconds"] = (
            self.df.groupby("source_file")["time"]
            .transform(lambda x: (x - x.min()).dt.total_seconds())
        )

    def _calculate_time_diff(self):
        """Calculate time difference between consecutive points in each track."""
        self.df["time_diff"] = (
            self.df.groupby("source_file")["time_seconds"]
            .diff()
            .fillna(0)  # First point in each track has no previous point; set diff=0
        )

# Example usage:
# parser = GPXParser(["path_to_gpx_file1.gpx", "path_to_gpx_file2.gpx"])
# df = parser.get_dataframe()
# processor = GPXDataProcessor(df)
# processed_df = processor.process_data()
# print(processed_df.head())