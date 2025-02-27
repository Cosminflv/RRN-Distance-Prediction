import os
from data_processor import GPXDataProcessor
from gpx_data_parser import GPXParser

gpx_files = [os.path.join('gpx_data', f) for f in os.listdir('gpx_data') if f.endswith('.gpx')]
parser = GPXParser(gpx_files)
df = parser.get_dataframe()
processor = GPXDataProcessor(df)
processed_df = processor.process_data()
print(processed_df.head())