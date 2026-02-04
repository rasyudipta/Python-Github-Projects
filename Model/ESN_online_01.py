import numpy as np
import pandas as pd
from numpy.linalg import pinv
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import joblib
import pickle  # For saving the entire model

class EchoStateNetwork:
    def __init__(self,
                 input_dim,
                 output_dim,
                 reservoir_size=500,
                 spectral_radius=0.95,
                 connectivity=0.1,  # the input scaling was kick out
                 noise=1e-6,
                 washout=100,
                 ridge_alpha=1e-6,
                 lamda=0.98,
                 random_state=None,
                 include_bias=False,
                 output_feedback=False):
        """
        Initialize the Echo State Network.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.noise = noise
        self.washout = washout
        self.ridge_alpha = ridge_alpha
        self.lamda = lamda
        self.random_state = random_state
        self.include_bias = include_bias  # Flag to include scalar bias
        self.output_feedback = output_feedback  # Flag for output feedback
        self._initialize_weights()
        self.states_train = None  # To store training states
        self.states_test = None  # To store testing states
        self.states_val = None  # To store validation states
        self.predictions_train = None
        self.predictions_test = None
        self.predictions_val = None
        self.history = {'train_mae': [], 'val_mae': [], 'test_mae': []}
        # For online update (RLS): inverse correlation matrix
        self.P = None
        self.last_state = np.zeros(self.reservoir_size)  # Initialize reservoir state
        self.last_output = np.zeros(self.output_dim)  # For output feedback
        # For recording online states during online updates
        self.state_history = []

    def _initialize_weights(self):
        rng = np.random.default_rng(self.random_state)
        # Initialize input weights (W_in)
        self.W_in = rng.uniform(-1, 1, (self.reservoir_size, self.input_dim))
        # Initialize reservoir weights (W_res) with given connectivity
        num_elements = int(self.reservoir_size * self.reservoir_size * self.connectivity)
        row_indices = rng.integers(0, self.reservoir_size, size=num_elements)
        col_indices = rng.integers(0, self.reservoir_size, size=num_elements)
        data = rng.uniform(-1, 1, size=num_elements)
        W_sparse = sparse.coo_matrix((data, (row_indices, col_indices)),
                                     shape=(self.reservoir_size, self.reservoir_size))
        W_sparse = W_sparse.tocsr()
        W_dense = W_sparse.toarray()
        largest_eigenvalue = self._power_iteration(W_dense)
        if largest_eigenvalue == 0:
            raise ValueError("Largest eigenvalue is zero; cannot scale the reservoir weights.")
        W_dense *= self.spectral_radius / largest_eigenvalue
        self.W_res = W_dense
        # Initialize bias if required
        if self.include_bias:
            self.bias = rng.uniform(-1, 1)
        else:
            self.bias = 0.0
        # Initialize feedback weights if output_feedback enabled
        if self.output_feedback:
            self.W_fb = rng.uniform(-1, 1, (self.reservoir_size, self.output_dim))
        else:
            self.W_fb = None
        # Output weights to be trained later
        self.W_out = None

    def _activation(self, x):
        """Activation function (tanh)."""
        return np.tanh(x)

    def _power_iteration(self, A, num_simulations: int = 100):
        """Estimate the largest eigenvalue of matrix A using Power Iteration."""
        b_k = np.random.rand(A.shape[1])
        for _ in range(num_simulations):
            b_k1 = np.dot(A, b_k)
            b_k1_norm = np.linalg.norm(b_k1)
            if b_k1_norm == 0:
                return 0
            b_k = b_k1 / b_k1_norm
        eigenvalue = np.dot(b_k, np.dot(A, b_k)) / np.dot(b_k, b_k)
        return eigenvalue

    def fit(self, inputs, outputs, washout, ridge_alpha):
        """
        Train the ESN using the provided input and output data.
        """
        n_samples = inputs.shape[0]
        if inputs.shape[1] != self.input_dim:
            raise ValueError(f"Expected inputs with {self.input_dim} features, but got {inputs.shape[1]}.")
        if outputs.shape[1] != self.output_dim:
            raise ValueError(f"Expected outputs with {self.output_dim} features, but got {outputs.shape[1]}.")
        state_dim = self.reservoir_size + self.input_dim
        if self.include_bias:
            state_dim += 1
        states = np.zeros((n_samples - self.washout, state_dim))
        x = np.zeros(self.reservoir_size)
        rng = np.random.default_rng(self.random_state)
        for t in range(n_samples):
            u = inputs[t]
            if self.output_feedback and t > 0:
                fb_term = np.dot(self.W_fb, outputs[t - 1])
            else:
                fb_term = 0.0
            pre_activation = (
                    np.dot(self.W_in, u)
                    + np.dot(self.W_res, x)
                    + fb_term
                    + self.bias
            )
            x = self._activation(pre_activation) + self.noise * rng.standard_normal(self.reservoir_size)
            if t >= self.washout:
                state = np.hstack((x, u))
                if self.include_bias:
                    state = np.hstack((state, self.bias))
                states[t - self.washout, :] = state
        Y = outputs[self.washout:]
        S = states
        Y_target = Y
        S_T_S = np.dot(S.T, S)
        reg = self.ridge_alpha * np.eye(S.shape[1])
        self.P = pinv(S_T_S + reg)
        self.W_out = np.dot(Y_target.T, np.dot(S, self.P))
        self.states_train = S.copy()
        print("ESN training completed with output feedback =", self.output_feedback)

    def predict(self, inputs, continuation=True, init_state=None):
        """
        Generate predictions using the trained ESN.
        """
        if self.W_out is None:
            raise ValueError("The ESN has not been trained. Call fit() first.")
        n_samples = inputs.shape[0]
        if inputs.shape[1] != self.input_dim:
            raise ValueError(f"Expected inputs with {self.input_dim} features, but got {inputs.shape[1]}.")
        if continuation and hasattr(self, 'last_state'):
            x = self.last_state
        elif init_state is not None:
            if len(init_state) != self.reservoir_size:
                raise ValueError(f"init_state should have length {self.reservoir_size}, but got {len(init_state)}.")
            x = init_state
        else:
            x = np.zeros(self.reservoir_size)
        state_dim = self.reservoir_size + self.input_dim
        if self.include_bias:
            state_dim += 1
        predictions = []
        states = []
        rng = np.random.default_rng(self.random_state)
        prev_output = self.last_output
        for t in range(n_samples):
            u = inputs[t]
            if self.output_feedback:
                fb_term = np.dot(self.W_fb, prev_output)
            else:
                fb_term = 0.0
            pre_activation = (
                    np.dot(self.W_in, u)
                    + np.dot(self.W_res, x)
                    + fb_term
                    + self.bias
            )
            x = self._activation(pre_activation) + self.noise * rng.standard_normal(self.reservoir_size)
            state = np.hstack((x, u))
            if self.include_bias:
                state = np.hstack((state, self.bias))
            y = np.dot(self.W_out, state)
            predictions.append(y)
            states.append(state)
            prev_output = y
        self.last_state = x
        self.last_output = prev_output
        self.states_test = np.array(states)
        return np.array(predictions)

    def compute_error(self, predictions, targets, metric='MSE'):
        """
        Compute the error between predictions and targets using the specified metric.
        """
        available_metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
        metric = metric.upper()
        if metric not in available_metrics:
            raise ValueError(f"Unsupported metric '{metric}'. Supported metrics are: {available_metrics}")
        if metric == 'MSE':
            error = np.mean((predictions - targets) ** 2, axis=0)
        elif metric == 'RMSE':
            error = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
        elif metric == 'MAE':
            error = np.mean(np.abs(predictions - targets), axis=0)
        elif metric == 'MAPE':
            epsilon = 1e-10
            error = np.mean(np.abs((targets - predictions) / (targets + epsilon)), axis=0) * 100
        elif metric == 'R2':
            ss_res = np.sum((targets - predictions) ** 2, axis=0)
            ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2, axis=0)
            error = 1 - ss_res / (ss_tot + 1e-10)
        if self.output_dim > 1:
            return np.mean(error)
        else:
            return float(error)

    def evaluate(self, predictions, targets, metrics=['MSE', 'RMSE', 'MAE']):
        """
        Compute multiple error metrics between predictions and targets.
        """
        errors = {}
        for metric in metrics:
            try:
                errors[metric] = self.compute_error(predictions, targets, metric=metric)
            except ValueError as e:
                print(f"Warning: {e}")
        return errors

    def get_weights(self):
        """
        Retrieve all weight matrices.
        """
        return {
            'W_in': self.W_in,
            'W_res': self.W_res,
            'W_out': self.W_out,
            'bias': self.bias if self.include_bias else None,
            'W_fb': self.W_fb if self.output_feedback else None
        }

    def get_states(self, dataset='train'):
        """
        Retrieve reservoir states from batch training.
        """
        if dataset == 'train':
            return self.states_train
        elif dataset == 'test':
            return self.states_test
        elif dataset == 'val':
            return self.states_val
        else:
            raise ValueError("dataset parameter must be 'train', 'test', or 'val'.")

    def get_state_online(self):
        """
        Retrieve reservoir states recorded during online updates.
        """
        if hasattr(self, 'state_history') and self.state_history:
            return np.array(self.state_history)
        else:
            return None

    def save_model(self, filename):
        """
        Save the entire ESN model to a file.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"ESN model saved to '{filename}'.")

    @staticmethod
    def load_model(filename):
        """
        Load an ESN model from a file.
        """
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"ESN model loaded from '{filename}'.")
        return model

    def plot_weight_matrix(self, weight_matrix='W_res', title='Weight Matrix'):
        """
        Plot the specified weight matrix.
        """
        weights = self.get_weights()
        if weight_matrix not in weights:
            raise ValueError("weight_matrix must be 'W_in', 'W_res', or 'W_out'.")
        plt.figure(figsize=(10, 8))
        plt.imshow(weights[weight_matrix], aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Neurons' if weight_matrix == 'W_res' else 'Inputs/Outputs')
        plt.ylabel('Neurons' if weight_matrix == 'W_res' else 'Outputs')
        plt.show()

    def plot_states(self, dataset='train', num_neurons=10, title='Reservoir States'):
        """
        Plot the reservoir states over time.
        """
        states = self.get_states(dataset)
        if states is None:
            raise ValueError(f"No states stored for dataset '{dataset}'.")
        plt.figure(figsize=(12, 6))
        for i in range(min(num_neurons, self.reservoir_size)):
            plt.plot(states[:, i], label=f'Neuron {i + 1}')
        plt.legend()
        plt.title(title)
        plt.xlabel('Time Step')
        plt.ylabel('State Value')
        plt.grid(True)
        plt.show()

    def online_update(self, new_input, new_target, ridge_alpha=1e-6):
        """
        Perform an online update of the ESN output weights using a single new sample.
        Also records the full state vector for this online update.
        """
        new_input = np.asarray(new_input)
        if new_input.ndim != 1 or new_input.shape[0] != self.input_dim:
            raise ValueError(f"new_input must be a 1D array with {self.input_dim} elements.")
        if self.output_feedback:
            fb_term = np.dot(self.W_fb, self.last_output)
        else:
            fb_term = 0.0
        rng = np.random.default_rng(self.random_state)
        pre_activation = (
                np.dot(self.W_in, new_input)
                + np.dot(self.W_res, self.last_state)
                + fb_term
                + self.bias
        )
        x_new = self._activation(pre_activation) + self.noise * rng.standard_normal(self.reservoir_size)
        s = np.hstack((x_new, new_input))
        if self.include_bias:
            s = np.hstack((s, self.bias))
        # Record the full state vector for online update
        self.state_history.append(s.copy())
        # Update last_state
        self.last_state = x_new
        y_pred = np.dot(self.W_out, s)
        error = new_target - y_pred
        s_col = s.reshape(-1, 1)
        denom = self.lamda + np.dot(s, np.dot(self.P, s))
        k = np.dot(self.P, s_col) / denom
        error_col = error.reshape(-1, 1)
        self.W_out = self.W_out + np.dot(error_col, k.T)
        self.P = (1 / self.lamda) * (self.P - np.dot(k, np.dot(s.reshape(1, -1), self.P)))
        self.last_output = new_target
        mae = np.mean(np.abs(error))
        return mae
