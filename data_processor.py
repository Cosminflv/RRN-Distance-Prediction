import numpy as np
import pandas as pd

time_to_predict_position = 4  # Predict position 4 seconds ahead

class GPXDataProcessor:
    def __init__(self, dataframe, scaler):
        self.df = dataframe.copy()
        self.scaler = scaler  # For data normalization

    def process_data(self):
        """Process data: handle missing elevation, normalize features, and calculate time features."""
        self._convert_time()
        self._calculate_time_diff()
        self._handle_missing_elevation()
        self._normalize_coordinates() 
        return self.df
    
    def create_sequences(self, processed_df, sequence_length=5):
        """
        Create sequences with elevation included in input features.
        Returns:
            X: Input sequences (n_samples, sequence_length, 4)
            y: Target distances in meters (n_samples,)
        """
        X, y = [], []
        all_distances = []  # To collect distances for scaling
        
        for track_id, track in processed_df.groupby('source_file'):
            orig_lat = track['latitude'].values
            orig_lon = track['longitude'].values
            orig_time = track['time_seconds'].values
            features = track[['latitude_norm', 'longitude_norm', 'elevation_norm', 'time_seconds_norm']].values
            
            for i in range(sequence_length, len(features)):
                last_idx = i - 1
                target_time = orig_time[last_idx] + time_to_predict_position
                
                # Find first point after prediction_time
                mask = orig_time[i:] >= target_time
                if not mask.any():
                    continue
                j = i + np.argmax(mask)
                
                # Calculate Haversine distance
                distance = self._haversine(
                    orig_lon[last_idx], orig_lat[last_idx],
                    orig_lon[j], orig_lat[j]
                )
                
                X.append(features[i-sequence_length:i])
                all_distances.append(distance)
        

            # Normalize all distances at once
        all_distances = np.array(all_distances).reshape(-1, 1)
        y_normalized = self.scaler.fit_transform(all_distances)
        
        # Split normalized targets back into sequences
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

    def _normalize_coordinates(self):
        """Normalize latitude, longitude, and elevation."""
        features = ["latitude", "longitude", "elevation", "time_seconds"]
        scaled = self.scaler.fit_transform(self.df[features])
        self.df["latitude_norm"] = scaled[:, 0]
        self.df["longitude_norm"] = scaled[:, 1]
        self.df["elevation_norm"] = scaled[:, 2]
        self.df["time_seconds_norm"] = scaled[:, 3]  # New normalized column

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