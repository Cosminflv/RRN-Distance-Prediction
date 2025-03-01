import os
from data_processor import GPXDataProcessor
from gpx_data_parser import GPXParser
from rnn_model import RNNTracker
from utils import split_data_by_track

gpx_files = [os.path.join('gpx_data', f) for f in os.listdir('gpx_data') if f.endswith('.gpx')]
parser = GPXParser(gpx_files)
df = parser.get_dataframe()
processor = GPXDataProcessor(df)
processed_df = processor.process_data()

# Split the data first
train_df, val_df, test_df = split_data_by_track(processed_df)

# Then create sequences for each split
X_train, y_train = processor.create_sequences(train_df, sequence_length=5)
X_val, y_val = processor.create_sequences(val_df, sequence_length=5)
X_test, y_test = processor.create_sequences(test_df, sequence_length=5)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Initialize model
tracker = RNNTracker(input_shape=(5, 2))  # sequence_length=5, features=2

# Compile model with appropriate settings
tracker.compile(loss='mse', metrics=['mae'])

# View architecture
tracker.summary()

# Train model
history = tracker.train(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

loss, accuracy = tracker.evaluate(X_test, y_test, batch_size=32)

print(f"Test loss: {loss}, Test accuracy: {accuracy}")