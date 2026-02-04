from Model_def import *
from Model_constant import *


# ===================== SIM_REACTOR_MODEL_2 (Incremental Update + Sequence Simulation) =====================
class Sim_Reactor_Model_2:
    def __init__(self, delta_t=1, f_constant=50000):
        self.delta_t = delta_t
        self.f_constant = f_constant
        self.reset_state()

    def reset_state(self):
        self.current_index = 0
        self.current_time = 0.0
        self.T_history = [reaction_const.Tin]  # suhu awal

        # History untuk logging
        self.times = []
        self.I0_values = []
        self.f_values = []
        self.f_list = []
        self.I_t_values = []
        self.M_t_values = []
        self.H_t_values = []
        self.B_t_values = []
        self.Hys_area_values = []
        self.T_values = []
        self.P_rest_values = []
        self.P_hys_values = []
        self.P_coil_values = []
        self.P_in_values = []
        self.Qin_values = []
        self.Q_flow_in_values = []
        self.Q_flow_out_values = []
        self.Q_wall_out_values = []
        self.Qout_values = []
        self.Qreaction_values = []
        self.sum_mol_cp_values = []
        self.x1_values = []
        self.x2_values = []
        self.x3_values = []

    def step(self, new_I0):
        current_time = self.current_time
        self.times.append(current_time)
        current_I0 = new_I0
        self.I0_values.append(current_I0)
        f = self.f_constant
        self.f_values.append(f)
        self.f_list.append(f)

        Tin_Tref = reaction_const.Tin - reaction_const.Tref
        Tout_Tref = reaction_const.Tout - reaction_const.Tref
        T_Tref = self.T_history[-1] - reaction_const.Tref

        # Perhitungan enthalpi
        dH_CH4, dH_H2O, dH_H2, dH_CO, dH_CO2 = kinetic_param.Enthalpy_i(
            enthalpy_coeff.c_CH4, enthalpy_coeff.c_H2O, enthalpy_coeff.c_H2,
            enthalpy_coeff.c_CO, enthalpy_coeff.c_CO2, T_Tref)
        dH_CH4_in, dH_H2O_in, dH_H2_in, dH_CO_in, dH_CO2_in = heat_stream_in.enthalpy_change_in(
            enthalpy_coeff.c_CH4, enthalpy_coeff.c_H2O, enthalpy_coeff.c_H2,
            enthalpy_coeff.c_CO2, enthalpy_coeff.c_CO2, reaction_const.Tin, reaction_const.Tref)

        # Laju reaksi dan kesetimbangan
        k1, k2, k3 = kinetic_param.Reaction_rate(
            reaction_const.A1, reaction_const.A2, reaction_const.A3,
            reaction_const.EA1, reaction_const.EA2, reaction_const.EA3,
            reaction_const.R1, self.T_history[-1])
        K1, K2, K3 = kinetic_param.Equilibrium_c(self.T_history[-1])
        KCH4, KH2O, KCO, KH2 = kinetic_param.Henry_constant_i(
            reaction_const.B_CH4, reaction_const.B_H2O, reaction_const.B_CO,
            reaction_const.B_H2, reaction_const.R1, dH_CH4, dH_H2O, dH_CO, dH_H2,
            self.T_history[-1])

        # Aliran panas dan reaksi
        h_outer = heat_wall_out.calculate_h_outer(
            reactor_params.Water_D, reactor_params.Water_v, reactor_params.Water_k,
            reactor_params.Water_Cp, reactor_params.Water_mu, reactor_params.Water_rho, flow_type="laminar")
        dH_CH4_out, dH_H2O_out, dH_H2_out, dH_CO_out, dH_CO2_out = heat_stream_out.enthalpy_change_out(
            enthalpy_coeff.c_CH4, enthalpy_coeff.c_H2O, enthalpy_coeff.c_H2,
            enthalpy_coeff.c_CO, enthalpy_coeff.c_CO2, reaction_const.Tout, reaction_const.Tref)
        nCH4_in, nH2O_in, nH2_in, nCO2_in, nCO_in = mol_stream_in.In_stream(
            initial_cond.nCH4, initial_cond.nH2O, initial_cond.nH2, initial_cond.nCO, initial_cond.nCO2)
        x1, x2, x3 = reaction_kinetic.conversion(current_time, initial_cond.x_Init, initial_cond.x_Fin)
        self.x1_values.append(x1)
        self.x2_values.append(x2)
        self.x3_values.append(x3)

        nCH4_r1, nH2O_r1, nH2_r1, nCO_r1 = reaction_kinetic.reaction_1(
            x1, initial_cond.nCH4, initial_cond.nH2O, initial_cond.nH2, initial_cond.nCO)
        nCO_r2, nH2O_r2, nH2_r2, nCO2_r2 = reaction_kinetic.reaction_2(
            x2, nCO_r1, nH2O_r1, nH2_r1, initial_cond.nCO2)
        nCH4_r3, nH2O_r3, nH2_r3, nCO2_r3 = reaction_kinetic.reaction_3(
            x3, initial_cond.nCH4, initial_cond.nH2O, initial_cond.nH2, initial_cond.nCO2)
        nCH4_out, nH2O_out, nH2_out, nCO_out, nCO2_out = mol_stream_out.Out_stream(
            nCH4_r1, nCH4_r3, nH2O_r2, nH2O_r3, nH2_r2, nH2_r3, nCO_r2, nCO2_r2, nCO2_r3)
        sum_mol_cp = (nCH4_in * molecule_weight.WM_CH4 * dH_CH4 +
                      nH2O_in * molecule_weight.WM_H2O * dH_H2O +
                      nH2_in * molecule_weight.WM_H2 * dH_H2 +
                      nCO_in * molecule_weight.WM_CO * dH_CO +
                      nCO2_in * molecule_weight.WM_CO2 * dH_CO2)
        self.sum_mol_cp_values.append(sum_mol_cp)

        Q_flow_out = heat_stream_out.Heat_flow_out(
            nCH4_out, nH2O_out, nH2_out, nCO_out, nCO2_out,
            enthalpy_coeff.c_CH4, enthalpy_coeff.c_H2O, enthalpy_coeff.c_H2,
            enthalpy_coeff.c_CO2, enthalpy_coeff.c_CO2, reaction_const.Tout, reaction_const.Tref)
        Qreaction = heat_reaction.HeatReactionTotal(
            heat_reaction.Heat_reaction_1(nCH4_r1, nH2O_r1, nH2_r1, nCO_r1,
                                          dH_CH4, dH_H2O, dH_H2, dH_CO, T_Tref),
            heat_reaction.Heat_reaction_2(nCO_r1, nH2O_r2, nH2_r2, nCO2_r2,
                                          nH2_r1, dH_H2O, dH_H2, dH_CO, dH_CO2, T_Tref),
            heat_reaction.Heat_reaction_3(nCH4_r3, nH2O_r3, nH2_r3, nCO2_r3,
                                          dH_CH4, dH_H2O, dH_H2, dH_CO2, T_Tref))
        Q_wall_out = heat_wall_out.Heat_out(
            self.T_history[-1], reaction_const.Tref, reactor_params.r_quartz_inner,
            reactor_params.r_quartz_outer, reactor_params.k_quartz, reactor_params.k_promasil,
            reactor_params.r_promasil_outer, reactor_params.L, h_outer)
        Q_flow_in = heat_stream_in.Heat_in(
            nCH4_in, nH2O_in, nH2_in, nCO2_in, nCO_in,
            enthalpy_coeff.c_CH4, enthalpy_coeff.c_H2O, enthalpy_coeff.c_H2,
            enthalpy_coeff.c_CO2, enthalpy_coeff.c_CO2, reaction_const.Tin, reaction_const.Tref)
        Qout = Q_flow_out + Q_wall_out
        self.Q_flow_out_values.append(Q_flow_out)
        self.Q_wall_out_values.append(Q_wall_out)
        self.Q_flow_in_values.append(Q_flow_in)
        self.Qout_values.append(Qout)
        self.Qreaction_values.append(Qreaction)

        omega = 2 * np.pi * f
        n_substeps = 5
        delta_t_sub = self.delta_t / n_substeps
        hysteresis_area = 0
        I_t_vals = []
        H_t_vals = []
        B_t_vals = []
        for sub_index in range(n_substeps):
            sub_time = current_time + sub_index * delta_t_sub
            I_t = current_I0 * np.sin(omega * sub_time)
            H_t = coil_params.N * I_t / coil_params.h_coil
            B_t = coil_params.mu_0 * coil_params.N * I_t
            I_t_vals.append(I_t)
            H_t_vals.append(H_t)
            B_t_vals.append(B_t)
            hysteresis_area = simpson(H_t_vals, B_t_vals)
        self.I_t_values.append(I_t_vals[-1])
        self.H_t_values.append(H_t_vals[-1])
        self.B_t_values.append(B_t_vals[-1])
        self.M_t_values.append(coil_params.Ms * (
            (((2 * coil_params.J + 1) / (2 * coil_params.J)) *
             (np.cosh(((2 * coil_params.J + 1) / (2 * coil_params.J)) * (I_t_vals[-1])) /
              (np.sinh(((2 * coil_params.J + 1) / (2 * coil_params.J)) * (I_t_vals[-1]) + 1e-10))) -
             (1 / (2 * coil_params.J)) * (np.cosh(I_t_vals[-1] / (2 * coil_params.J)) /
                                          (np.sinh(I_t_vals[-1] / (2 * coil_params.J)) + 1e-10)))
        ))
        self.Hys_area_values.append(hysteresis_area)

        P_hys = hysteresis_area * f * coil_params.ms
        self.P_hys_values.append(P_hys)
        I_rms = current_I0 / np.sqrt(2)
        self.P_coil_values.append(I_rms ** 2 * coil_params.L_coil * (np.pi * coil_params.r0 / coil_params.A_coil) *
                                  np.sqrt(coil_params.N * np.sqrt(np.pi) * coil_params.r0 *
                                          coil_params.rho_copper * coil_params.miu_copper / coil_params.h_coil) *
                                  np.sqrt(f))
        P_coil = self.P_coil_values[-1]
        P_in_temp = P_hys + P_coil
        P_rest = 0.05 * P_in_temp
        self.P_rest_values.append(P_rest)
        P_in = P_hys + P_coil + P_rest
        self.P_in_values.append(P_in)
        Qin = P_in * current_time
        self.Qin_values.append(Qin)

        dT_dt = (Q_flow_in + Qin - Qreaction - Q_flow_out - Q_wall_out) / sum_mol_cp
        new_T = self.T_history[-1] + dT_dt * self.delta_t
        self.T_history.append(new_T)
        self.T_values.append(new_T)

        self.current_index += 1
        self.current_time += self.delta_t

        return {'Time (s)': current_time,
                'I0_control (A)': current_I0,
                'dT/dt (K/s)': dT_dt,
                'Temperature (K)': new_T}

    def simulate_sequence(self, control_sequence):
        self.reset_state()
        outputs = []
        for u in control_sequence:
            out = self.step(u)
            outputs.append(out)
        return outputs