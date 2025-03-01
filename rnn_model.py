from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

class RNNTracker:
    def __init__(self, input_shape, lstm_units=128, activation='relu'):
        """
        RNN-based position prediction model
        
        Args:
            input_shape: Tuple (sequence_length, features)
            lstm_units: Number of LSTM units
            activation: Activation function for LSTM
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.activation = activation
        self.model = self._build_model()
        
    def _build_model(self):
        """Construct the LSTM architecture"""
        model = Sequential()
        model.add(LSTM(
            self.lstm_units,
            activation=self.activation,
            input_shape=self.input_shape,
            return_sequences=False
        ))
        model.add(Dense(2))  # Predicts (lat_norm, lon_norm)
        return model
    
    def compile(self, optimizer='adam', loss='mse', metrics=None):
        """Configure the model training parameters"""
        _metrics = metrics or ['mae']  # Mean Absolute Error more meaningful than accuracy
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=_metrics
        )
        
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_data=None):
        """Train the model with optional validation data"""
        return self.model.fit(
            X_train, 
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        )
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """Evaluate model performance on test data"""
        return self.model.evaluate(X_test, y_test, batch_size=batch_size)
    
    def predict(self, input_data):
        """Make predictions from input sequence"""
        return self.model.predict(input_data)
    
    def summary(self):
        """Display model architecture summary"""
        return self.model.summary()
    
    def save(self, filepath):
        """Save entire model to filepath"""
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load model from filepath"""
        model = tf.keras.models.load_model(filepath)
        # Create dummy instance to return
        instance = cls(input_shape=model.layers[0].input_shape[1:])
        instance.model = model
        return instance

# Usage example:
if __name__ == "__main__":
    # After creating sequences from GPXDataProcessor
    # X, y = create_sequences(processed_data)
    
    # Initialize model
    tracker = RNNTracker(input_shape=(5, 2))  # sequence_length=5, features=2
    
    # Compile model with appropriate settings
    tracker.compile(loss='mse', metrics=['mae'])
    
    # View architecture
    tracker.summary()
    
    # Train model
    # history = tracker.train(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    
    # Evaluate model
    # loss, mae = tracker.evaluate(X_test, y_test)
    
    # Save model
    # tracker.save('path/to/model.h5')
    
    # Load model
    # loaded_tracker = RNNTracker.load('path/to/model.h5')