import pandas as pd
import xml.etree.ElementTree as ET

class GPXParser:
    def __init__(self, gpx_files):
        """Accepts a list of GPX files to parse."""
        self.gpx_files = gpx_files if isinstance(gpx_files, list) else [gpx_files]
        self.data = None

    def _retain_values(self, df, column, min_quartile, max_quartile):
        q_min, q_max = df[column].quantile([min_quartile, max_quartile])
        print("Keeping values between {} and {} of column {}".format(q_min, q_max, column))
        return df[(df[column] > q_min) & (df[column] < q_max)]

    def parse_gpx_csv(self):
        """Parses the GPX data stored in a CSV file and extracts track points into a DataFrame."""
        df = pd.read_csv('gpx_data/train/gpx-tracks-from-hikr.org.csv')

        df.dropna()
        df['avg_speed'] = df['length_3d']/df['moving_time']
        df = df[df['avg_speed'] < 2.5] # an avg of > 2.5m/s is probably not a hiking activity
        df = self._retain_values(df, 'min_elevation', 0.01, 1)

        trackpoints = []
        namespace = {'gpx': 'http://www.topografix.com/GPX/1/1'}

        for index, row in df.iterrows():
            gpx_str = row['gpx']
            try:
                root = ET.fromstring(gpx_str)
            except ET.ParseError:
                continue  # Skip invalid XML entries
            
            for trkpt in root.findall(".//gpx:trkpt", namespace):
                lat = float(trkpt.get("lat"))
                lon = float(trkpt.get("lon"))
                ele = trkpt.find("gpx:ele", namespace)
                time_elem = trkpt.find("gpx:time", namespace)

                elevation = float(ele.text) if ele is not None else None
                time = time_elem.text if time_elem is not None else None

                trackpoints.append({
                    "latitude": lat,
                    "longitude": lon,
                    "elevation": elevation,
                    "time": time,
                    "source_file": row['_id']  # Using '_id' as the track identifier
                })

        self.data = pd.DataFrame(trackpoints)

        if not self.data.empty:
         # Handle duplicates
         temp_df = self.data.copy()
         temp_df['elevation'] = temp_df['elevation'].fillna(-9999)
        
         # Convert 'time' to datetime, coercing errors to NaT
         temp_df['time_dt'] = pd.to_datetime(temp_df['time'], errors='coerce')
        
         # Drop rows where time is NaT (invalid/missing time)
         temp_df = temp_df.dropna(subset=['time_dt'])
        
         # Ensure 'time' column has valid values
         temp_df = temp_df[temp_df['time'].notna() & (temp_df['time'] != 'NaT')]
        
         # Group by source_file and compute elapsed time within each group
         temp_df['elapsed_time'] = temp_df.groupby('source_file')['time_dt'].transform(
             lambda x: (x - x.min()).dt.total_seconds()
         )
        
         # Restore original NaN values for elevation
         temp_df['elevation'] = temp_df['elevation'].replace(-9999, None)

         # Remove rows where elevation is None
         temp_df = temp_df.dropna(subset=['elevation'])
        
         self.data = temp_df
        else:
         self.data = pd.DataFrame()
        
        return self.data

    def parse_gpx(self):
        """Parses the GPX files and extracts latitude, longitude, elevation, and time."""
        namespace = {'gpx': 'http://www.topografix.com/GPX/1/1'}
        trackpoints = []
        
        for gpx_file in self.gpx_files:
            tree = ET.parse(gpx_file)
            root = tree.getroot()
            
            for trkpt in root.findall(".//gpx:trkpt", namespace):
                lat = float(trkpt.get("lat"))
                lon = float(trkpt.get("lon"))
                ele = trkpt.find("gpx:ele", namespace)
                time = trkpt.find("gpx:time", namespace)
                
                trackpoints.append({
                    "latitude": lat,
                    "longitude": lon,
                    "elevation": float(ele.text) if ele is not None else None,
                    "time": time.text if time is not None else None,
                    "source_file": gpx_file
                })
        
        self.data = pd.DataFrame(trackpoints)
        
        # REMOVE DUPLICATES
        # Handle NaN values for proper duplicate checking
        temp_df = self.data.copy()
        # Replace NaN with placeholders
        temp_df['elevation'] = temp_df['elevation'].fillna(-9999)
        temp_df['time'] = temp_df['time'].fillna('NaT')
        
        # Drop duplicates considering all attributes
        temp_df = temp_df.drop_duplicates(
            subset=['latitude', 'longitude', 'elevation', 'source_file']
        )

        removed_items_count =  self.data.size - temp_df.size 

        temp_df['time_dt'] = pd.to_datetime(temp_df['time'])
        start_time = temp_df['time_dt'].iloc[0]
        temp_df['elapsed_time'] = (temp_df['time_dt'] - start_time).dt.total_seconds()
        
        # Restore original NaN values
        temp_df['elevation'] = temp_df['elevation'].replace(-9999, None)
        temp_df['time'] = temp_df['time'].replace('NaT', None)
        
        self.data = temp_df
        return self.data

    def get_dataframe(self):
        """Returns the parsed GPX data as a DataFrame."""
        if self.data is None:
            return self.parse_gpx()
        return self.data
    
    def get_csv_dataframe(self):
        """Returns the parsed GPX data from CSV as a DataFrame."""
        if self.data is None:
            return self.parse_gpx_csv()
        return self.data

# Example usage:
# parser = GPXParser(["path_to_gpx_file1.gpx", "path_to_gpx_file2.gpx"])
# df = parser.get_dataframe()
# print(df.head())