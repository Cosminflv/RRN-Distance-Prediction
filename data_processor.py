import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

class GPXDataProcessor:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.scaler = None  # For coordinate normalization

    def process_data(self):
        """
        Process data: convert time, calculate time diffs, normalize coordinates.

        Returns:
            df: DataFrame with processed columns
        """
        self._convert_time()
        self._calculate_time_diff()
        self._normalize_coordinates()
        return self.df
    
    def create_sequences(self, processed_df, sequence_length=5):
        """
        Create sequences of past coordinates and target future coordinates.
        
        Args:
            processed_df: DataFrame from GPXDataProcessor.process_data()
            sequence_length: Number of historical points to use for prediction
            
        Returns:
            X: Array of shape (n_samples, sequence_length, 2)
            y: Array of shape (n_samples, 2)
        """
        X, y = [], []
        
        # Process each track separately
        for track_id, track in processed_df.groupby('source_file'):
            # Get normalized coordinates as numpy array
            coords = track[['latitude_norm', 'longitude_norm']].values
            
            # Create sequences for this track
            if len(coords) >= sequence_length:
                for i in range(sequence_length, len(coords)):
                    X.append(coords[i-sequence_length:i])  # Historical sequence
                    y.append(coords[i])                   # Next point
                
        return np.array(X), np.array(y)

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

    def _normalize_coordinates(self):
        """Normalize latitude and longitude to zero mean and unit variance."""
        self.scaler = StandardScaler()
        scaled_coords = self.scaler.fit_transform(self.df[["latitude", "longitude"]])
        self.df["latitude_norm"] = scaled_coords[:, 0]
        self.df["longitude_norm"] = scaled_coords[:, 1]


# Example usage:
# parser = GPXParser(["path_to_gpx_file1.gpx", "path_to_gpx_file2.gpx"])
# df = parser.get_dataframe()
# processor = GPXDataProcessor(df)
# processed_df = processor.process_data()
# print(processed_df.head())