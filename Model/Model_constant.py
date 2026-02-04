import numpy as np
import pandas as pd
import math
import openpyxl
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.integrate import trapezoid
from trimesh.util import tolist

class ReactionConstants:
    def __init__(self):
        # Constants for Steam Methane Reforming reactions
        # r1: CH4 + H2O <--> 3H2 + CO
        # r2: CO + H2O <--> H2 + CO2
        # r3: CH4 + 2H2O <--> 4H2 + CO2

        # temperature
        self.Tkelvin = 273.15  # kelvin
        self.Tref = self.Tkelvin + 25  # K 25
        self.Tin = self.Tkelvin + 50  # K 300
        self.Tout = self.Tkelvin + 850  # K 850

        # pressure
        self.P = 1  # bar

        # constant
        self.R = 0.08314  # L.bar/mol.K
        self.R1 = 8.314  # J/mol.K
        self.Vstp = 22.7  # L/mol

        # Activation energy
        self.EA1 = 240.1  # kJ/mol
        self.EA2 = 249.9  # kJ/mol
        self.EA3 = 67.13  # kJ/mol

        # Pre-exponential factors (kmol, bar, kgcat, h)
        self.A1 = 4.23e15
        self.A2 = 1.96e6
        self.A3 = 1.02e15

        # Henry's law constants for different species
        self.B_H2O = 1.77e5
        self.B_CH4 = 6.65e-5
        self.B_CO = 8.23e-5
        self.B_H2 = 6.12e-5
reaction_const = ReactionConstants()


class EnthalpyCoefficients:
    def __init__(self):
        # Polynomial coefficients for enthalpy calculation Joule/(mol.K) (Yaws' coefficients)
        self.c_CH4 = [34.942, -4.00E-02, 1.92E-04, -1.53E-07, 3.93E-11]
        self.c_H2O = [33.933, -8.42E-03, 2.99E-05, -1.78E-08, 3.69E-12]
        self.c_H2 = [25.399, 2.02E-02, -3.85E-05, 3.19E-08, -8.76E-12]
        self.c_CO = [29.556, -6.58E-03, 2.01E-05, -1.22E-08, 2.26E-12]
        self.c_CO2 = [27.437, 4.23E-02, -1.96E-05, 4.00E-09, -2.99E-13]
enthalpy_coeff = EnthalpyCoefficients()


class MoleculeWeight:
    def __init__(self):
        # Mass balance molecular weights
        self.WM_CH4 = 16.042  # g/mol
        self.WM_H2O = 18.016  # g/mol
        self.WM_H2 = 2.016  # g/mol
        self.WM_CO = 28.01  # g/mol
        self.WM_CO2 = 44.01  # g/mol
molecule_weight = MoleculeWeight()


class Coil:
    def __init__(self):
        # Coil
        self.N = 9  # turn winding
        self.h_coil = 0.135  # m
        self.ms = 0.011  # kg
        self.mol_Co = 58.93  # g/mol
        self.mol_Ni = 58.69  # g/mol
        self.mol_CoNi = (self.mol_Ni + self.mol_Co) / 2
        self.mol = (self.ms * 1000) / self.mol_CoNi
        self.L_coil = 1.18  # m
        self.r0 = 0.0032512  # m
        self.A_coil = 0.000490873852123405  # m^2
        self.rho_copper = 1.72E-08  # Ω m
        self.miu_copper = 1.26E-06  # H/m

        self.mu_0 = 4 * np.pi * 10 ** -7  # T·m/A
        self.gJ = 1.3333333  # Landé g-factor
        self.mu_B = 9.274e-24  # Bohr magneton (A*m^2)
        self.kB = 1.38e-23  # Boltzmann constant (J/K)
        self.J = 4.5  # Total angular momentum quantum number
        self.rho_CoNi = 870  # kg/m3
        self.Ms = 110 * self.rho_CoNi * 1000  # A/m
        self.Cp = 44.31  # J/(K.mol)
coil_params = Coil()


class FlowRate:
    def __init__(self):
        # Flow rate
        self.FCH4 = 30 / 3600  # L/s
        self.SCratio = 2
flow_params = FlowRate()


class InitialConditions:
    def __init__(self, reaction_const, flow_params):
        # Initial condition
        self.Init_nCH4 = reaction_const.P * (flow_params.FCH4 / reaction_const.Vstp) / (
                    reaction_const.R * reaction_const.Tin)
        self.Init_nH2O = flow_params.SCratio * self.Init_nCH4

        # Conversion intervals
        self.t = 1
        self.x_Init = 0.56
        self.x_Fin = 0.96

        # Initial moles
        self.nCH4 = self.Init_nCH4
        self.nH2O = self.Init_nH2O
        self.nH2 = 0.107 * self.Init_nCH4
        self.nCO = 0
        self.nCO2 = 0
initial_cond = InitialConditions(reaction_const, flow_params)


class ReactorDimension:
    def __init__(self):
        # Input parameters
        self.r_quartz_inner = 0.0065  # Inner radius of quartz tube (m)
        self.r_quartz_outer = 0.0070  # Outer radius of quartz tube (m)
        self.r_promasil_outer = 0.0125  # Outer radius of Promasil insulation (m)
        self.L = 0.36  # Length of the tube (m)

        self.k_quartz = 1.4  # Thermal conductivity of quartz (W/m·K)
        self.k_promasil = 0.163  # Thermal conductivity of Promasil (W/m·K)

        self.Water_D = 2 * self.r_promasil_outer
        self.Water_v = 1.0  # Coolant flow velocity (m/s)
        self.Water_k = 0.6  # Thermal conductivity of water (W/m·K)
        self.Water_Cp = 4186  # Specific heat capacity of water (J/kg·K)
        self.Water_mu = 1e-3  # Dynamic viscosity of water (Pa·s)
        self.Water_rho = 1000  # Density of water (kg/m³)
reactor_params = ReactorDimension()

