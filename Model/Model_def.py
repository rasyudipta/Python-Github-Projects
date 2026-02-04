from Model_constant import *


class KineticsParameters :
    # Enthalpy of spcecies
    def Enthalpy_i(self, c_CH4, c_H2O, c_H2, c_CO, c_CO2, T):
        dH_CH4 = np.polyval(c_CH4[::-1], T)
        dH_H2O = np.polyval(c_H2O[::-1], T)
        dH_H2 = np.polyval(c_H2[::-1], T)
        dH_CO = np.polyval(c_CO[::-1], T)
        dH_CO2 = np.polyval(c_CO2[::-1], T)
        return dH_CH4, dH_H2O, dH_H2, dH_CO, dH_CO2

    # reaction rate
    def Reaction_rate(self, A1, A2, A3, EA1, EA2, EA3, R1, T):
        k1 = A1 * np.exp(-(EA1 * 1000) / (R1 * T))
        k2 = A2 * np.exp(-(EA2 * 1000) / (R1 * T))
        k3 = A3 * np.exp(-(EA3 * 1000) / (R1 * T))
        return k1, k2, k3

    # Equilibrium constant
    def Equilibrium_c(self, T):
        K1 = 1.198e17 * np.exp(-26830 / T)
        K2 = 1.767e-2 * np.exp(4400 / T)
        K3 = 2.117e15 * np.exp(-22430 / T)
        return K1, K2, K3

    # Henry's constants adjusted for temperature
    def Henry_constant_i(self, B_CH4, B_H2O, B_CO, B_H2, dH_CH4, dH_H2O, dH_CO, dH_H2, R1, T):
        KCH4 = B_CH4 * np.exp(-dH_CH4 / 1000 / (R1 * T))
        KH2O = B_H2O * np.exp(-dH_H2O / 1000 / (R1 * T))
        KCO = B_CO * np.exp(-dH_CO / 1000 / (R1 * T))
        KH2 = B_H2 * np.exp(-dH_H2 / 1000 / (R1 * T))
        return KCH4, KH2O, KCO, KH2
kinetic_param = KineticsParameters()

class ReactionKinetics :
    def conversion(self, t, x_Init, x_Fin):
        # Fungsi eksponensial untuk perubahan yang semakin lambat
        x1 = x_Init + (x_Fin - x_Init) * (1 - np.exp(-t * 2))
        x2 = x_Init + (x_Fin - x_Init) * (1 - np.exp(-t * 2))  # Variasi konstanta skala
        x3 = x_Init + (x_Fin - x_Init) * (1 - np.exp(-t * 2))
        return x1, x2, x3


    def reaction_1(self, x1, nCH4, nH2O, nH2, nCO):
        # Reaction 1: CH4 + H2O <--> 3H2 + CO
        nCH4_r1 = (nCH4 * 0.5) - ((nCH4 * 0.5) * x1)
        nH2O_r1 = (nH2O * 0.5) - ((nCH4 * 0.5) * x1)
        nH2_r1 = (nH2 * 0.5) + 3 * ((nCH4 * 0.5) * x1)
        nCO_r1 = (nCO * 0.5) + ((nCH4 * 0.5) * x1)
        return nCH4_r1, nH2O_r1, nH2_r1, nCO_r1


    def reaction_2(self, x2, nCO_r1, nH2O_r1, nH2_r1, nCO2):
        # Reaction 2: CO + H2O <--> H2 + CO2
        nCO_r2 = nCO_r1 - (nCO_r1 * x2)
        nH2O_r2 = nH2O_r1 - (nCO_r1 * x2)
        nH2_r2 = nH2_r1 + (nCO_r1 * x2)
        nCO2_r2 = nCO2 + (nCO_r1 * x2)
        return nCO_r2, nH2O_r2, nH2_r2, nCO2_r2


    def reaction_3(self, x3, nCH4, nH2O, nH2, nCO2):
        # Reaction 3: CH4 + 2H2O <--> 4H2 + CO2
        nCH4_r3 = (nCH4 * 0.5) - ((nCH4 * 0.5) * x3)
        nH2O_r3 = (nH2O * 0.5) - 2 * ((nCH4 * 0.5) * x3)
        nH2_r3 = (nH2 * 0.5) + 4 * ((nCH4 * 0.5) * x3)
        nCO2_r3 = (nCO2 * 0.5) + ((nCH4 * 0.5) * x3)
        return nCH4_r3, nH2O_r3, nH2_r3, nCO2_r3
reaction_kinetic = ReactionKinetics()

class StreamIn :
    def In_stream(self, nCH4, nH2O, nH2, nCO, nCO2):
        # In Stream amounts
        nCH4_in = nCH4
        nH2O_in = nH2O
        nH2_in = nH2
        nCO_in = nCO
        nCO2_in = nCO2
        return nCH4_in, nH2O_in, nH2_in, nCO2_in, nCO_in
mol_stream_in = StreamIn()

class StreamOut :
    def Out_stream(self, nCH4_r1, nCH4_r3, nH2O_r2, nH2O_r3, nH2_r2, nH2_r3, nCO_r2, nCO2_r2, nCO2_r3):
        # Out Stream amounts
        nCH4_out = nCH4_r1 + nCH4_r3
        nH2O_out = nH2O_r2 + nH2O_r3
        nH2_out = nH2_r2 + nH2_r3
        nCO_out = nCO_r2
        nCO2_out = nCO2_r2 + nCO2_r3
        return nCH4_out, nH2O_out, nH2_out, nCO_out, nCO2_out
mol_stream_out = StreamOut()

class Conversion:
    def total_conversion(self, nCH4_out, Init_nCH4):
        # Total number of moles before and after reactions
        n_initial = Init_nCH4  # + Init_nH2O
        n_final = nCH4_out  # + nH2O_final + nH2_final + nCO_final + nCO2_final

        # Total conversion
        x_total = (n_initial - n_final) / n_initial
        return x_total
X_conversion = Conversion()

class MassBalance :
    def __init__(self):
        self.WM_CH4 = molecule_weight.WM_CH4
        self.WM_H2O = molecule_weight.WM_H2O
        self.WM_H2  = molecule_weight.WM_H2
        self.WM_CO  = molecule_weight.WM_CO
        self.WM_CO2 = molecule_weight.WM_CO2

    def Mass_In(self, nCH4_in, nH2O_in, nH2_in, nCO2_in, nCO_in):
        m_CH4_in = nCH4_in * self.WM_CH4  # g/h
        m_H2O_in = nH2O_in * self.WM_H2O  # g/h
        m_H2_in = nH2_in * self.WM_H2  # g/h
        m_CO_in = nCO_in * self.WM_CO  # g/h
        m_CO2_in = nCO2_in * self.WM_CO2  # g/h
        m_in_total = m_CH4_in + m_H2O_in + m_H2_in + m_CO_in + m_CO2_in
        return m_in_total

    def Mass_Out(self, nCH4_out, nH2O_out, nH2_out, nCO_out, nCO2_out):
        m_CH4_out = nCH4_out * self.WM_CH4  # g/h
        m_H2O_out = nH2O_out * self.WM_H2O  # g/h
        m_H2_out = nH2_out * self.WM_H2  # g/h
        m_CO_out = nCO_out * self.WM_CO  # g/h
        m_CO2_out = nCO2_out * self.WM_CO2  # g/h
        m_out_total = m_CH4_out + m_H2O_out + m_H2_out + m_CO_out + m_CO2_out
        return m_out_total

    def Mass_Balance_result(self, m_in_total, m_out_total):
        Mass_Balance_result = m_in_total - m_out_total
        return Mass_Balance_result
mass_balance = MassBalance()

class PartialPressure :
    def partial_pressure_r1(self, nCH4_r1, nH2O_r1, nH2_r1, nCO_r1, P):
        # Partial pressures of components in reactions
        p_CH4_r1 = P * nCH4_r1
        p_H2O_r1 = P * nH2O_r1
        p_H2_r1 = P * nH2_r1
        p_CO_r1 = P * nCO_r1
        return p_CH4_r1, p_H2O_r1, p_H2_r1, p_CO_r1

    def partial_pressure_r2(self, nCO_r2, nH2O_r2, nH2_r2, nCO2_r2, P) :
        p_CO_r2 = P * nCO_r2
        p_H2O_r2 = P * nH2O_r2
        p_H2_r2 = P * nH2_r2
        p_CO2_r2 = P * nCO2_r2
        return p_CO_r2, p_H2O_r2, p_H2_r2, p_CO2_r2

    def partial_pressure_r3(self, nCH4_r3, nH2O_r3, nH2_r3, nCO2_r3, P):
        p_CH4_r3 = P * nCH4_r3
        p_H2O_r3 = P * nH2O_r3
        p_H2_r3 = P * nH2_r3
        p_CO2_r3 = P * nCO2_r3
        return p_CH4_r3, p_H2O_r3, p_H2_r3, p_CO2_r3

    def partial_pressure_out(self, nCH4_out, nH2O_out, nH2_out, nCO_out, nCO2_out, P) :
        p_CH4_out = P * nCH4_out
        p_H2O_out = P * nH2O_out
        p_H2_out = P * nH2_out
        p_CO_out = P * nCO_out
        p_CO2_out = P * nCO2_out
        return p_CH4_out, p_H2O_out, p_H2_out, p_CO_out, p_CO2_out
partial_pressure = PartialPressure()

class ReactionRate :
    def reaction_denominator(self, KCH4, KH2O, KCO, KH2, p_CH4, p_H2O, p_H2, p_CO):
        # Reaction rates
        D = 1 + KCO * p_CO + KH2 * p_H2 + KCH4 * p_CH4 + KH2O * (p_H2O / p_H2)
        return D

    def reaction_rate_1(self, p_CH4_r1, p_H2O_r1, p_H2_r1, p_CO_r1, K1, k1, D):
        r1 = ((k1 / p_H2_r1 ** 2.5) * (p_CH4_r1 * p_H2O_r1 - (p_H2_r1 ** 3 * p_CO_r1) / K1)) / D ** 2
        return r1

    def reaction_rate_2(self, p_CO_r2, p_H2O_r2, p_H2_r2, p_CO2_r2, K2, k2, D) :
        r2 = ((k2 / p_H2_r2) * (p_CO_r2 * p_H2O_r2 - (p_H2_r2 * p_CO2_r2) / K2)) / D ** 2
        return r2

    def reaction_rate_3(self, p_CH4_r3, p_H2O_r3, p_H2_r3, p_CO2_r3, K3, k3, D):
        r3 = ((k3 / p_H2_r3 ** 3.5) * (p_CH4_r3 * p_H2O_r3 ** 2 - (p_H2_r3 ** 4 * p_CO2_r3) / K3)) / D ** 2
        return r3
reaction_rate = ReactionRate()

class HeatReactions :
    def delta_T_reaction(self, T, Tref):
        dT_reaction = T - Tref
        return dT_reaction

    def Heat_reaction_1(self, nCH4_r1, nH2O_r1, nH2_r1, nCO_r1, dH_CH4, dH_H2O, dH_H2, dH_CO, dT_reaction):
        # Reaction 1: CH4 + H2O <--> 3H2 + CO
        Q_r1 = (-nCH4_r1 * dH_CH4 + -nH2O_r1 * dH_H2O + nH2_r1 * dH_H2 + nCO_r1 * dH_CO) * dT_reaction
        return Q_r1

    def Heat_reaction_2(self, nCO_r1, nH2O_r2, nH2_r2, nCO2_r2, nH2_r1, dH_H2O, dH_H2, dH_CO, dH_CO2, dT_reaction):
        # Reaction 2: CO + H2O <--> H2 + CO2
        Q_r2 = (-nCO_r1 * dH_CO + -nH2O_r2 * dH_H2O + (nH2_r2 - nH2_r1) * dH_H2 + nCO2_r2 * dH_CO2) * dT_reaction
        return Q_r2

    def Heat_reaction_3(self, nCH4_r3, nH2O_r3, nH2_r3, nCO2_r3, dH_CH4, dH_H2O, dH_H2, dH_CO2, dT_reaction):
        # Reaction 3: CH4 + 2H2O <--> 4H2 + CO2
        Q_r3 = (-nCH4_r3 * dH_CH4 + -nH2O_r3 * dH_H2O + nH2_r3 * dH_H2 + nCO2_r3 * dH_CO2) * dT_reaction
        return Q_r3

    def HeatReactionTotal(self, Q_r1, Q_r2, Q_r3):
        Qreaction = Q_r1 + Q_r2 + Q_r3
        return Qreaction
heat_reaction = HeatReactions()

class HeatStreamOut :
    def __init__(self):
        self.WM_CH4 = molecule_weight.WM_CH4
        self.WM_H2O = molecule_weight.WM_H2O
        self.WM_H2  = molecule_weight.WM_H2
        self.WM_CO  = molecule_weight.WM_CO
        self.WM_CO2 = molecule_weight.WM_CO2

    def enthalpy_change_out(self, c_CH4, c_H2O, c_H2, c_CO, c_CO2, Tout, Tref):
        dH_CH4_out = np.polyval(c_CH4[::-1], Tout - Tref)
        dH_H2O_out = np.polyval(c_H2O[::-1], Tout - Tref)
        dH_H2_out = np.polyval(c_H2[::-1], Tout - Tref)
        dH_CO_out = np.polyval(c_CO[::-1], Tout - Tref)
        dH_CO2_out = np.polyval(c_CO2[::-1], Tout - Tref)
        return dH_CH4_out, dH_H2O_out, dH_H2_out, dH_CO_out, dH_CO2_out

    def Heat_flow_out(self, nCH4_out, nH2O_out, nH2_out, nCO_out, nCO2_out, c_CH4, c_H2O, c_H2, c_CO, c_CO2, Tout, Tref):
        dH_CH4_out, dH_H2O_out, dH_H2_out, dH_CO_out, dH_CO2_out = self.enthalpy_change_out(
            c_CH4, c_H2O, c_H2, c_CO, c_CO2, Tout, Tref
        )

        # Calculate heat input for each component
        Q_CH4_out = nCH4_out * dH_CH4_out
        Q_H2O_out = nH2O_out * dH_H2O_out
        Q_H2_out = nH2_out * dH_H2_out
        Q_CO_out = nCO_out * dH_CO_out
        Q_CO2_out = nCO2_out * dH_CO2_out

        Q_flow_out = Q_CH4_out + Q_H2O_out + Q_H2_out + Q_CO_out + Q_CO2_out
        return Q_flow_out
heat_stream_out = HeatStreamOut()

class HeatWallOut :
    # Calculate h_outer using empirical correlations for Nusselt number
    def calculate_h_outer(self, D, v, k_fluid, Cp, mu, rho, flow_type="laminar"):
        Re = (rho * v * D) / mu
        Pr = (mu * Cp) / k_fluid

        if flow_type == "laminar" and Re < 2300:
            Nu = 0.664 * (Re ** 0.5) * (Pr ** (1 / 3))
        else:
            Nu = 0.023 * (Re ** 0.8) * (Pr ** (1 / 3))

        h_outer = (Nu * k_fluid) / D
        return h_outer


    # Function to calculate convective resistance for outer layer
    def convective_resistance(self, h_outer, r_outer, L):
        A_outer = 2 * np.pi * r_outer * L
        return 1 / (h_outer * A_outer)


    # Function to calculate thermal resistance for a cylindrical layer
    def thermal_resistance_cylindrical(self, r_inner, r_outer, k, L):
        return math.log(r_outer / r_inner) / (2 * math.pi * k * L)


    # Heat transfer through reactor
    def Heat_out(self, T, Tref, r_quartz_inner, r_quartz_outer, k_quartz, r_promasil_outer, k_promasil, L, h_outer):
        R_quartz = self.thermal_resistance_cylindrical(r_quartz_inner, r_quartz_outer, k_quartz, L)
        R_promasil = self.thermal_resistance_cylindrical(r_quartz_outer, r_promasil_outer, k_promasil, L)
        R_cooling = self.convective_resistance(h_outer, r_promasil_outer, L)
        R_total = R_quartz + R_promasil + R_cooling
        Q_wall_out = (T - Tref) / R_total
        return Q_wall_out
heat_wall_out = HeatWallOut()

class HeatStreamIn :
    def __init__(self):
        self.WM_CH4 = molecule_weight.WM_CH4
        self.WM_H2O = molecule_weight.WM_H2O
        self.WM_H2 = molecule_weight.WM_H2
        self.WM_CO = molecule_weight.WM_CO
        self.WM_CO2 = molecule_weight.WM_CO2

    def enthalpy_change_in(self, c_CH4, c_H2O, c_H2, c_CO, c_CO2, Tin, Tref):
        dH_CH4_in = np.polyval(c_CH4[::-1], Tin - Tref)
        dH_H2O_in = np.polyval(c_H2O[::-1], Tin - Tref)
        dH_H2_in = np.polyval(c_H2[::-1], Tin - Tref)
        dH_CO_in = np.polyval(c_CO[::-1], Tin - Tref)
        dH_CO2_in = np.polyval(c_CO2[::-1], Tin - Tref)
        return dH_CH4_in, dH_H2O_in, dH_H2_in, dH_CO_in, dH_CO2_in

    def Heat_in(self, nCH4, nH2O, nH2, nCO, nCO2, c_CH4, c_H2O, c_H2, c_CO, c_CO2, Tin, Tref):
        dH_CH4_in, dH_H2O_in, dH_H2_in, dH_CO_in, dH_CO2_in = self.enthalpy_change_in(
            c_CH4, c_H2O, c_H2, c_CO, c_CO2, Tin, Tref
        )

        # Calculate heat input for each component
        Q_CH4_in = nCH4 * dH_CH4_in
        Q_H2O_in = nH2O * dH_H2O_in
        Q_H2_in = nH2 * dH_H2_in
        Q_CO_in = nCO * dH_CO_in
        Q_CO2_in = nCO2 * dH_CO2_in

        # Total heat input
        Q_flow_in = Q_CH4_in + Q_H2O_in + Q_H2_in + Q_CO_in + Q_CO2_in

        return Q_flow_in
heat_stream_in = HeatStreamIn()