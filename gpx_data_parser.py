import pandas as pd
import xml.etree.ElementTree as ET

class GPXParser:
    def __init__(self, gpx_files):
        """Accepts a list of GPX files to parse."""
        self.gpx_files = gpx_files if isinstance(gpx_files, list) else [gpx_files]
        self.data = None

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
            subset=['latitude', 'longitude', 'elevation']
        )

        removed_items_count =  self.data.size - temp_df.size 

        temp_df['time_dt'] = pd.to_datetime(temp_df['time'])

        # Calculate elapsed_time PER SOURCE FILE
        # Group by source_file and compute time relative to each file's start
        temp_df['start_time_per_file'] = temp_df.groupby('source_file')['time_dt'].transform('min')
        temp_df['elapsed_time'] = (temp_df['time_dt'] - temp_df['start_time_per_file']).dt.total_seconds()
        
        # Restore original NaN values
        temp_df['elevation'] = temp_df['elevation'].replace(-9999, None)
        temp_df['time'] = temp_df['time'].replace('NaT', None)
        temp_df = temp_df.drop(columns=['start_time_per_file'])  # Cleanup
        
        self.data = temp_df
        return self.data

    def get_dataframe(self):
        """Returns the parsed GPX data as a DataFrame."""
        if self.data is None:
            return self.parse_gpx()
        return self.data

# Example usage:
# parser = GPXParser(["path_to_gpx_file1.gpx", "path_to_gpx_file2.gpx"])
# df = parser.get_dataframe()
# print(df.head())