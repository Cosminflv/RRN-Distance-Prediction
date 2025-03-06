import numpy as np
import pandas as pd

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
        
        Args:
            processed_df: DataFrame from process_data()
            sequence_length: Historical points per sequence
            
        Returns:
            X: Input sequences with shape (n_samples, sequence_length, 3)
            y: Target coordinates with shape (n_samples, 2)
        """
        X, y = [], []
        
        for track_id, track in processed_df.groupby('source_file'):
            # Include elevation in features, targets are still coordinates
            features = track[['latitude_norm', 'longitude_norm', 'elevation_norm', 'time_seconds_norm']].values
            targets = track[['latitude_norm', 'longitude_norm']].values
            
            if len(features) >= sequence_length:
                for i in range(sequence_length, len(features)):
                    if i + 30 < len(features):
                        X.append(features[i-sequence_length:i])
                        y.append(targets[i + 30])
        
        return np.array(X), np.array(y)

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