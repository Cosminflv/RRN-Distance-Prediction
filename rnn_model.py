from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf  # Requires statsmodels

class RNNTracker:
    def __init__(self, input_shape, lstm_units=128, activation='relu', scaler=None):
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
        self.history = None  # To store training history
        
    def _build_model(self):
        """Construct the LSTM architecture"""
        inputs = Input(shape=self.input_shape)  # Define input layer
        x = LSTM(self.lstm_units, activation=self.activation, return_sequences=False)(inputs)
        outputs = Dense(2)(x)  # Predicts (lat_norm, lon_norm)

        model = Model(inputs=inputs, outputs=outputs)  # Functional API model
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
    
    def plot_training_curves(self, metric='loss', figsize=(10, 6)):
        """Plot training vs validation loss/MAE curves"""
        if self.history is None:
            raise ValueError("Train the model first using .train()")
        
        plt.figure(figsize=figsize)
        plt.plot(self.history.history[metric], label=f'Training {metric}')
        plt.plot(self.history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Training vs Validation {metric.upper()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.show()

    def plot_actual_vs_predicted(self, X, y_true, figsize=(10, 6), sample_points=200):
        """Scatter plot of true vs predicted coordinates (in original WGS84 scale)"""

        # Predict normalized values
        y_pred = self.predict(X)

        # Plot results
        plt.figure(figsize=figsize) 
        plt.scatter(y_true[:sample_points, 0], y_true[:sample_points, 1], 
                    label='True Positions', alpha=0.5, marker='o', color='blue')
        plt.scatter(y_pred[:sample_points, 0], y_pred[:sample_points, 1], 
                    label='Predicted Positions', alpha=0.5, marker='x', color='red')

        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title('Actual vs Predicted Positions (Denormalized)')
        plt.legend()
        plt.show()

    def plot_error_distributions(self, X, y_true, figsize=(12, 5)):
        """Histogram of latitude/longitude errors"""
        y_pred = self.predict(X)
        errors = y_true - y_pred
        
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.hist(errors[:, 0], bins=50, color='blue', alpha=0.7)
        plt.title('Latitude Error Distribution')
        plt.xlabel('Error')

        plt.subplot(1, 2, 2)
        plt.hist(errors[:, 1], bins=50, color='red', alpha=0.7)
        plt.title('Longitude Error Distribution')
        plt.xlabel('Error')
        
        plt.tight_layout()
        plt.show()

    def plot_trajectory_comparison(self, X_sequence, y_true_trajectory, figsize=(10, 6)):
        """Compare true vs predicted trajectory for a sample sequence"""
        y_pred = self.predict(X_sequence[np.newaxis, ...])[0]
        
        plt.figure(figsize=figsize)
        plt.plot(y_true_trajectory[:, 0], y_true_trajectory[:, 1], 
                'g-', label='True Trajectory', linewidth=2)
        plt.plot(y_pred[0], y_pred[1], 'ro--', 
                label='Predicted Position', markersize=8)
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title('Trajectory Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_residual_autocorrelation(self, X, y_true, figsize=(12, 5)):
        """ACF plots for temporal correlation in errors"""
        y_pred = self.predict(X)
        errors = y_true - y_pred
        
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plot_acf(errors[:, 0], title='Latitude Errors ACF')
        
        plt.subplot(1, 2, 2)
        plot_acf(errors[:, 1], title='Longitude Errors ACF')
        
        plt.tight_layout()
        plt.show()

    def plot_geospatial_errors(self, X, y_true, figsize=(10, 6)):
        """2D kernel density estimate of errors (requires seaborn)"""
        import seaborn as sns
        y_pred = self.predict(X)
        errors = y_true - y_pred
        
        plt.figure(figsize=figsize)
        sns.kdeplot(x=errors[:, 1], y=errors[:, 0], 
                   cmap='viridis', fill=True, levels=20)
        plt.xlabel('Longitude Error')
        plt.ylabel('Latitude Error')
        plt.title('Geospatial Error Distribution')
        plt.colorbar(label='Density')
        plt.show()

    def visualize_model_architecture(self, show_shapes=True, show_layer_names=True):
        """Generate model architecture diagram (requires pydot)"""
        tf.keras.utils.plot_model(self.model, show_shapes=show_shapes,
                                 show_layer_names=show_layer_names, 
                                 rankdir='LR', dpi=65, to_file='model.png')

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