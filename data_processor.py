import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

class GPXDataProcessor:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.scaler = None  # For coordinate normalization

    def process_data(self):
        """Process data: convert time, calculate time diffs, normalize coordinates."""
        self._convert_time()
        self._calculate_time_diff()
        self._normalize_coordinates()
        return self.df

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