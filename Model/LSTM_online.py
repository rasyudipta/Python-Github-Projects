import tensorflow as tf
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda

# Callback to record the time taken per epoch
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)

# Callback to evaluate and record test loss at the end of each epoch
class TestLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        super(TestLossCallback, self).__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.test_loss = []
    def on_epoch_end(self, epoch, logs=None):
        loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.test_loss.append(loss)

class LSTMonline:
    def __init__(self, input_shape, lstm_units=32, dense_units=1, learning_rate=0.001):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.batch_size = None
        self._build_model()

    def _build_model(self):
        # Create an input layer.
        inputs = Input(shape=self.input_shape)
        # Use the LSTM layer to return states.
        lstm_out, state_h, state_c = LSTM(self.lstm_units, activation='tanh', return_state=True)(inputs)
        # For predictions, use only the LSTM output.
        predictions = Dense(self.dense_units)(lstm_out)
        # Build the main model.
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mae')
        # Also create a state model to retrieve the hidden and cell states.
        self.state_model = Model(inputs=inputs, outputs=[state_h, state_c])

    def initial_train(self, X_train, Y_train, epochs=10, batch_size=32, verbose=1, test_data=None):
        self.batch_size = batch_size
        time_callback = TimeHistory()
        callbacks = [time_callback]
        if test_data is not None:
            test_callback = TestLossCallback(test_data[0], test_data[1])
            callbacks.append(test_callback)
        history = self.model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=verbose,
            callbacks=callbacks
        )
        history.history['epoch_times'] = time_callback.times
        if test_data is not None:
            history.history['test_loss'] = test_callback.test_loss
        return history.history

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def online_update(self, X, Y):
        loss = self.model.train_on_batch(X, Y)
        return loss

    def get_states(self, X):
        """
        Returns the hidden state (h_state) and cell state (c_state) for the given input X.
        Note: This method bypasses the Sequential model and directly calls the stored LSTM layer.
        """
        # Call the LSTM layer directly (it returns a tuple: (output, h_state, c_state))
        output, h_state, c_state = self.lstm_layer(X)
        return h_state, c_state
        
