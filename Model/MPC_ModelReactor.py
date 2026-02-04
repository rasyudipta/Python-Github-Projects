import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import copy
from joblib import load
from scipy.optimize import minimize, differential_evolution # Menggunakan SLSQP dan differential_evolution
from ModelReactor_class import *

# ===================== MPC UNTUK REAKTOR =====================
class ReactorMPC:
    def __init__(self, initial_control_sequence, setpoint,
                 delta_u_max, u_min, u_max, R_weight, Q_weight,
                 delta_t=1, optimizer='slsqp', M=10, N=3):
        self.control_sequence = list(initial_control_sequence)
        self.setpoint = setpoint
        self.delta_u_max = delta_u_max
        self.u_min = u_min
        self.u_max = u_max
        self.R_weight = R_weight
        self.Q_weight = Q_weight
        self.optimizer = optimizer
        self.u_prev = initial_control_sequence[-1]
        self.M = M  # Prediction horizon (number of simulation steps)
        self.N = N  # Control horizon (number of decision moves)

        # Instantiate the reactor simulation model
        self.reactor_model = Sim_Reactor_Model_2(delta_t=delta_t)

    def simulate_horizon(self, u_seq):
        backup = copy.deepcopy(self.reactor_model)
        errors = []
        for m in range(self.M):
            # Use u_seq[m] if available; otherwise, hold last value
            u_m = u_seq[m] if m < len(u_seq) else u_seq[-1]
            out = self.reactor_model.step(u_m)
            pred_temp = out['Temperature (K)']
            errors.append(pred_temp - self.setpoint)
        self.reactor_model = backup  # Restore model state
        return errors

    def cost_function(self, u):
        u_seq = u  # Candidate future control moves (length N)
        errors = self.simulate_horizon(u_seq)
        # Tracking cost: sum of squared errors weighted by Q_weight
        tracking_cost = self.Q_weight * np.sum(np.array(errors) ** 2)

        # Control move cost: first move is relative to previous control; subsequent moves are differences.
        control_moves = np.array([u_seq[0] - self.u_prev] +
                                 [u_seq[i] - u_seq[i - 1] for i in range(1, self.N)])
        # Define R as a diagonal matrix (size N) with R_weight on the diagonal
        R = self.R_weight * np.eye(self.N)
        control_cost = control_moves.T @ R @ control_moves

        return tracking_cost + control_cost

    def constraint1(self, u):
        # Enforce: u[0] - u_prev <= delta_u_max
        return self.delta_u_max - (u[0] - self.u_prev)

    def constraint2(self, u):
        # Enforce: u_prev - u[0] <= delta_u_max
        return self.delta_u_max - (self.u_prev - u[0])

    def solve_de(self):
        bounds = [(self.u_min, self.u_max)] * self.N
        result = differential_evolution(lambda x: self.cost_function(x), bounds)
        return result.x

    def solve(self):
        bounds = [(self.u_min, self.u_max)] * self.N
        cons = [{'type': 'ineq', 'fun': self.constraint1},
                {'type': 'ineq', 'fun': self.constraint2}]
        x0 = np.array([self.u_prev] * self.N)
        if self.optimizer == 'de':
            return self.solve_de()
        else:
            res = minimize(self.cost_function, x0, method='SLSQP', bounds=bounds, constraints=cons)
            if res.success:
                return res.x
            else:
                return x0

    def update(self):
        u_seq_opt = self.solve()
        # Append the first control move to the control sequence
        self.control_sequence.append(u_seq_opt[0])
        self.u_prev = u_seq_opt[0]
        # Apply the control to the reactor model for one step update
        out = self.reactor_model.step(u_seq_opt[0])
        return out