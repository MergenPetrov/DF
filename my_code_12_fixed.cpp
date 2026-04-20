if (element_status)
        {
          const double q_tot_buf = wsncs->wsn_mixture_molar_rate;
          const unsigned int nseg_vars = 1U + mp.nc + 1U;
          const double fd_eps_p = 1.e-3;
          const unsigned int df_fd_eval_mode_flag = 0x80000000U;

          struct df_fd_wsncs_backup_t
            {
              double wsn_mixture_molar_rate = 0.0;
              double wsn_C_0 = 0.0;
              double wsn_drift_velocity = 0.0;
              double wsn_C_0_OW = 0.0;
              double wsn_drift_velocity_OW = 0.0;
              double wsn_mixture_mass_rate = 0.0;
              double wsn_mmw = 0.0;
              double rho_avg_DF = 0.0;

              std::vector<double> wsn_component_rate;
              std::vector<double> phase_S;
              std::vector<double> phase_xi;
              std::vector<double> component_phase_x;
              std::vector<double> avg_D_xi;
              std::vector<double> avg_D_rho;
              std::vector<double> phase_D_S;
              std::vector<double> phase_D_xi;
              std::vector<double> component_phase_D_x;

              df_fd_wsncs_backup_t (const well_segment_node_computation_status *src,
                                    const model_parameters &mp_in)
              {
                wsn_mixture_molar_rate = src->wsn_mixture_molar_rate;
                wsn_C_0 = src->wsn_C_0;
                wsn_drift_velocity = src->wsn_drift_velocity;
                wsn_C_0_OW = src->wsn_C_0_OW;
                wsn_drift_velocity_OW = src->wsn_drift_velocity_OW;
                wsn_mixture_mass_rate = src->wsn_mixture_mass_rate;
                wsn_mmw = src->wsn_mmw;
                rho_avg_DF = src->rho_avg_DF;

                wsn_component_rate.resize (mp_in.nc, 0.0);
                for (unsigned int ic = 0; ic < mp_in.nc; ++ic)
                  wsn_component_rate[ic] = src->wsn_component_rate[ic];

                phase_S.resize (mp_in.np, 0.0);
                phase_xi.resize (mp_in.np, 0.0);
                for (unsigned int ip = 0; ip < mp_in.np; ++ip)
                  {
                    phase_S[ip] = src->phase_S[ip];
                    phase_xi[ip] = src->phase_xi[ip];
                  }

                component_phase_x.resize (mp_in.nc * mp_in.np, 0.0);
                for (unsigned int k = 0; k < mp_in.nc * mp_in.np; ++k)
                  component_phase_x[k] = src->component_phase_x[k];

                avg_D_xi.resize (1U + mp_in.nc, 0.0);
                avg_D_rho.resize (1U + mp_in.nc, 0.0);
                for (unsigned int id = 0; id < 1U + mp_in.nc; ++id)
                  {
                    avg_D_xi[id] = src->avg_D_xi[id];
                    avg_D_rho[id] = src->avg_D_rho[id];
                  }

                phase_D_S.resize ((1U + mp_in.nc) * mp_in.np, 0.0);
                phase_D_xi.resize ((1U + mp_in.nc) * mp_in.np, 0.0);
                for (unsigned int k = 0; k < (1U + mp_in.nc) * mp_in.np; ++k)
                  {
                    phase_D_S[k] = src->phase_D_S[k];
                    phase_D_xi[k] = src->phase_D_xi[k];
                  }

                component_phase_D_x.resize ((1U + mp_in.nc) * mp_in.nc * mp_in.np, 0.0);
                for (unsigned int k = 0; k < (1U + mp_in.nc) * mp_in.nc * mp_in.np; ++k)
                  component_phase_D_x[k] = src->component_phase_D_x[k];
              }

              void restore (well_segment_node_computation_status *dst,
                            const model_parameters &mp_in) const
              {
                dst->wsn_mixture_molar_rate = wsn_mixture_molar_rate;
                dst->wsn_C_0 = wsn_C_0;
                dst->wsn_drift_velocity = wsn_drift_velocity;
                dst->wsn_C_0_OW = wsn_C_0_OW;
                dst->wsn_drift_velocity_OW = wsn_drift_velocity_OW;
                dst->wsn_mixture_mass_rate = wsn_mixture_mass_rate;
                dst->wsn_mmw = wsn_mmw;
                dst->rho_avg_DF = rho_avg_DF;

                for (unsigned int ic = 0; ic < mp_in.nc; ++ic)
                  dst->wsn_component_rate[ic] = wsn_component_rate[ic];

                for (unsigned int ip = 0; ip < mp_in.np; ++ip)
                  {
                    dst->phase_S[ip] = phase_S[ip];
                    dst->phase_xi[ip] = phase_xi[ip];
                  }

                for (unsigned int k = 0; k < mp_in.nc * mp_in.np; ++k)
                  dst->component_phase_x[k] = component_phase_x[k];

                for (unsigned int id = 0; id < 1U + mp_in.nc; ++id)
                  {
                    dst->avg_D_xi[id] = avg_D_xi[id];
                    dst->avg_D_rho[id] = avg_D_rho[id];
                  }

                for (unsigned int k = 0; k < (1U + mp_in.nc) * mp_in.np; ++k)
                  {
                    dst->phase_D_S[k] = phase_D_S[k];
                    dst->phase_D_xi[k] = phase_D_xi[k];
                  }

                for (unsigned int k = 0; k < (1U + mp_in.nc) * mp_in.nc * mp_in.np; ++k)
                  dst->component_phase_D_x[k] = component_phase_D_x[k];
              }
            };

          const df_fd_wsncs_backup_t wsncs_backup (wsncs, mp);

          struct df_persist_seed_state_t
            {
              const void *seg_key = nullptr;
              bool valid = false;
              double alpha_g = 1.0 / 3.0;
              double alpha_o = 1.0 / 3.0;
              double alpha_w = 1.0 / 3.0;
              double C0 = 1.0;
              double Vd = 0.0;
              double C0OW = 1.0;
              double VdOW = 0.0;
            };

          static std::vector<df_persist_seed_state_t> df_seed_state_store;
          const void *df_seed_seg_key = static_cast<const void *> (seg.wsn);
          auto find_df_seed_state =
            [&] (const void *seg_key) -> df_persist_seed_state_t *
            {
              for (auto &seed_state : df_seed_state_store)
                {
                  if (seed_state.seg_key == seg_key)
                    return &seed_state;
                }
              return nullptr;
            };

          df_persist_seed_state_t *df_seed_state_ptr = find_df_seed_state (df_seed_seg_key);
          bool df_seed_initialized = (df_seed_state_ptr != nullptr && df_seed_state_ptr->valid);

          double prev_df_alpha_g_seed = 1.0 / 3.0;
          double prev_df_alpha_o_seed = 1.0 / 3.0;
          double prev_df_alpha_w_seed = 1.0 / 3.0;
          double prev_df_C0_seed = 1.0;
          double prev_df_Vd_seed = 0.0;
          double prev_df_C0OW_seed = 1.0;
          double prev_df_VdOW_seed = 0.0;

          if (df_seed_initialized)
            {
              prev_df_alpha_g_seed = df_seed_state_ptr->alpha_g;
              prev_df_alpha_o_seed = df_seed_state_ptr->alpha_o;
              prev_df_alpha_w_seed = df_seed_state_ptr->alpha_w;
              prev_df_C0_seed = df_seed_state_ptr->C0;
              prev_df_Vd_seed = df_seed_state_ptr->Vd;
              prev_df_C0OW_seed = df_seed_state_ptr->C0OW;
              prev_df_VdOW_seed = df_seed_state_ptr->VdOW;
            }

          auto apply_fd_perturbation =
            [&] (fully_implicit_element_status *es, double &qtot_work, unsigned int var_id, double delta)
            {
              if (var_id == 0)
                {
                  es->p += delta;
                }
              else if (var_id < 1U + mp.nc)
                {
                  es->component_N[var_id - 1] += delta;
                  es->component_N_tot += delta;
                }
              else
                {
                  qtot_work += delta;
                }
            };

          auto fd_step_value =
            [&] (unsigned int var_id) -> double
            {
              if (var_id == 0)
                return fd_eps_p;
              if (var_id < 1U + mp.nc)
                return 1.e-4 * std::max (1.0, fabs (element_status->component_N[var_id - 1]));
              return 1.e-6 * std::max (1.0, fabs (q_tot_buf));
            };

          auto zero_df_derivatives =
            [&] ()
            {
              for (unsigned int id = 0; id < nseg_vars; ++id)
                {
                  wsncs->D_average_volumetric_mixture_velocity_D_seg_vars[id] = 0.0;
                  wsncs->D_C0_D_seg_vars[id] = 0.0;
                  wsncs->D_drift_velocity_D_seg_vars[id] = 0.0;
                  wsncs->D_C0_OW_D_seg_vars[id] = 0.0;
                  wsncs->D_drift_velocity_OW_D_seg_vars[id] = 0.0;
                  wsncs->D_rho_avg_D_seg_vars[id] = 0.0;
                  wsncs->D_mixture_mass_rate_D_seg_vars[id] = 0.0;
                }

              for (unsigned int ic = 0; ic < mp.nc; ++ic)
                {
                  for (unsigned int id = 0; id < nseg_vars; ++id)
                    wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id] = 0.0;
                }
            };

          auto compute_mass_rate_and_mmw =
            [&] ()
            {
              wsncs->wsn_mixture_mass_rate = 0.0;
              wsncs->wsn_mmw = 0.0;
              for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                {
                  wsncs->wsn_mixture_mass_rate +=
                      wsncs->wsn_component_rate[ic] * component_molar_weights[ic];
                  wsncs->wsn_mmw += component_z_for_flow[ic] * component_molar_weights[ic];
                }
            };

          // ------------------------ base DF value path
          double base_average_velocity = 0.0;
          double base_gas_interfacial_tension = 0.0;
          double base_bubble_rise_velocity = 0.0;
          double base_liq_density = 0.0;
          double base_diametr_dimless = 0.0;
          double base_Kut_number = 0.0;
          double base_flooding_velocity = 0.0;
          double base_ksi = 0.0;
          double base_eta = 0.0;
          double base_gas_phase_velocity = 0.0;
          double base_liquid_phase_velocity = 0.0;
          double base_K_g = 0.0;
          double base_avg_xi = 0.0;
          double base_xi = 0.0;
          double base_sigma_o = 0.0;
          double base_sigma_w = 0.0;
          double base_oil_water_interfacial_tension = 0.0;
          double base_bubble_rise_velocity_OW = 0.0;
          double base_C_0 = 0.0;
          double base_drift_velocity = 0.0;
          double base_C0_OW = 0.0;
          double base_Vd_OW = 0.0;
          double base_oil_velocity = 0.0;
          double base_water_velocity = 0.0;
          double base_alpha_g = 0.0;
          double base_beta_o = 0.0;
          double base_alpha_o = 0.0;
          double base_alpha_w = 0.0;
          double base_R_g = 0.0;
          double base_R_o = 0.0;
          unsigned int base_active_flags = 0U;
          int base_inner_it = 0;

          base_active_flags = 0U;
          test_function (rep,
                         seg,
                         wsncs,
                         element_status,
                         mp,
                         new_status,
                         i_meshblock,
                         current_therm_comp_input_props,
                         itd,
                         base_average_velocity,
                         base_gas_interfacial_tension,
                         base_bubble_rise_velocity,
                         base_liq_density,
                         base_diametr_dimless,
                         base_Kut_number,
                         base_flooding_velocity,
                         base_ksi,
                         base_eta,
                         base_gas_phase_velocity,
                         base_liquid_phase_velocity,
                         base_K_g,
                         base_avg_xi,
                         base_xi,
                         base_sigma_o,
                         base_sigma_w,
                         base_oil_water_interfacial_tension,
                         base_bubble_rise_velocity_OW,
                         base_C0_OW,
                         base_Vd_OW,
                         base_oil_velocity,
                         base_water_velocity,
                         base_alpha_g,
                         base_beta_o,
                         base_alpha_o,
                         base_alpha_w,
                         base_R_g,
                         base_R_o,
                         base_active_flags,
                         base_inner_it,
                         df_seed_initialized,
                         prev_df_alpha_g_seed,
                         prev_df_alpha_o_seed,
                         prev_df_alpha_w_seed,
                         prev_df_C0_seed,
                         prev_df_Vd_seed,
                         prev_df_C0OW_seed,
                         prev_df_VdOW_seed);

          base_C_0 = wsncs->wsn_C_0;
          base_drift_velocity = wsncs->wsn_drift_velocity;

          if (seg.wsn->wsn_index != TOP_SEG_INDEX)
            {
              compute_mass_rate_and_mmw ();

              const double base_alpha_sum_seed = base_alpha_g + base_alpha_o + base_alpha_w;
              if (std::isfinite (base_alpha_sum_seed) && base_alpha_sum_seed > 1.0e-12
                  && base_alpha_g >= 0.0 && base_alpha_o >= 0.0 && base_alpha_w >= 0.0)
                {
                  if (df_seed_state_ptr == nullptr)
                    {
                      df_seed_state_store.push_back (df_persist_seed_state_t ());
                      df_seed_state_ptr = &df_seed_state_store.back ();
                      df_seed_state_ptr->seg_key = df_seed_seg_key;
                    }

                  df_seed_state_ptr->valid = true;
                  df_seed_state_ptr->alpha_g = base_alpha_g / base_alpha_sum_seed;
                  df_seed_state_ptr->alpha_o = base_alpha_o / base_alpha_sum_seed;
                  df_seed_state_ptr->alpha_w = base_alpha_w / base_alpha_sum_seed;
                  df_seed_state_ptr->C0 = (std::isfinite (base_C_0) && fabs (base_C_0) > tnm::min_compare) ? base_C_0 : 1.0;
                  df_seed_state_ptr->Vd = std::isfinite (base_drift_velocity) ? base_drift_velocity : 0.0;
                  df_seed_state_ptr->C0OW = (std::isfinite (base_C0_OW) && fabs (base_C0_OW) > tnm::min_compare) ? base_C0_OW : 1.0;
                  df_seed_state_ptr->VdOW = std::isfinite (base_Vd_OW) ? base_Vd_OW : 0.0;

                  df_seed_initialized = true;
                  prev_df_alpha_g_seed = df_seed_state_ptr->alpha_g;
                  prev_df_alpha_o_seed = df_seed_state_ptr->alpha_o;
                  prev_df_alpha_w_seed = df_seed_state_ptr->alpha_w;
                  prev_df_C0_seed = df_seed_state_ptr->C0;
                  prev_df_Vd_seed = df_seed_state_ptr->Vd;
                  prev_df_C0OW_seed = df_seed_state_ptr->C0OW;
                  prev_df_VdOW_seed = df_seed_state_ptr->VdOW;
                }
            }
          zero_df_derivatives ();

          if (seg.wsn->wsn_index != TOP_SEG_INDEX)
            {
              for (unsigned int var_id = 0; var_id < nseg_vars; ++var_id)
                {
                  const double fd_eps = fd_step_value (var_id);
                  if (!(fd_eps > 0.0))
                    continue;

                  fully_implicit_element_status element_status_prev (*element_status);
                  fully_implicit_element_status *element_status_prev_ptr = &element_status_prev;
                  copy_segment_params_to_element_status (seg, element_status_prev_ptr);
                  wsncs_backup.restore (wsncs, mp);
                  double qtot_prev = q_tot_buf;
                  apply_fd_perturbation (element_status_prev_ptr, qtot_prev, var_id, -fd_eps);
                  wsncs->wsn_mixture_molar_rate = qtot_prev;

                  double prev_average_velocity = 0.0;
                  double prev_gas_interfacial_tension = 0.0;
                  double prev_bubble_rise_velocity = 0.0;
                  double prev_liq_density = 0.0;
                  double prev_diametr_dimless = 0.0;
                  double prev_Kut_number = 0.0;
                  double prev_flooding_velocity = 0.0;
                  double prev_ksi = 0.0;
                  double prev_eta = 0.0;
                  double prev_gas_phase_velocity = 0.0;
                  double prev_liquid_phase_velocity = 0.0;
                  double prev_K_g = 0.0;
                  double prev_avg_xi = 0.0;
                  double prev_xi = 0.0;
                  double prev_sigma_o = 0.0;
                  double prev_sigma_w = 0.0;
                  double prev_oil_water_interfacial_tension = 0.0;
                  double prev_bubble_rise_velocity_OW = 0.0;
                  double prev_C0_OW = 0.0;
                  double prev_Vd_OW = 0.0;
                  double prev_oil_velocity = 0.0;
                  double prev_water_velocity = 0.0;
                  double prev_alpha_g = 0.0;
                  double prev_beta_o = 0.0;
                  double prev_alpha_o = 0.0;
                  double prev_alpha_w = 0.0;
                  double prev_R_g = 0.0;
                  double prev_R_o = 0.0;
                  unsigned int prev_active_flags = 0U;
                  int prev_inner_it = 0;

                  prev_active_flags = df_fd_eval_mode_flag;
                  test_function (rep,
                                 seg,
                                 wsncs,
                                 element_status_prev_ptr,
                                 mp,
                                 new_status,
                                 i_meshblock,
                                 current_therm_comp_input_props,
                                 itd,
                                 prev_average_velocity,
                                 prev_gas_interfacial_tension,
                                 prev_bubble_rise_velocity,
                                 prev_liq_density,
                                 prev_diametr_dimless,
                                 prev_Kut_number,
                                 prev_flooding_velocity,
                                 prev_ksi,
                                 prev_eta,
                                 prev_gas_phase_velocity,
                                 prev_liquid_phase_velocity,
                                 prev_K_g,
                                 prev_avg_xi,
                                 prev_xi,
                                 prev_sigma_o,
                                 prev_sigma_w,
                                 prev_oil_water_interfacial_tension,
                                 prev_bubble_rise_velocity_OW,
                                 prev_C0_OW,
                                 prev_Vd_OW,
                                 prev_oil_velocity,
                                 prev_water_velocity,
                                 prev_alpha_g,
                                 prev_beta_o,
                                 prev_alpha_o,
                                 prev_alpha_w,
                                 prev_R_g,
                                 prev_R_o,
                                 prev_active_flags,
                                 prev_inner_it,
                                 df_seed_initialized,
                                 prev_df_alpha_g_seed,
                                 prev_df_alpha_o_seed,
                                 prev_df_alpha_w_seed,
                                 prev_df_C0_seed,
                                 prev_df_Vd_seed,
                                 prev_df_C0OW_seed,
                                 prev_df_VdOW_seed);

                  const double prev_C0 = wsncs->wsn_C_0;
                  const double prev_drift_velocity = wsncs->wsn_drift_velocity;
                  const double prev_C0OW_live = wsncs->wsn_C_0_OW;
                  const double prev_VdOW_live = wsncs->wsn_drift_velocity_OW;
                  const double prev_rho_avg = wsncs->rho_avg_DF;
                  std::vector<double> prev_component_rate (mp.nc, 0.0);
                  for (unsigned int ic = 0; ic < mp.nc; ++ic)
                    prev_component_rate[ic] = wsncs->wsn_component_rate[ic];
                  double prev_mix_mass_rate = 0.0;
                  for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                    prev_mix_mass_rate += prev_component_rate[ic] * component_molar_weights[ic];

                  fully_implicit_element_status element_status_next (*element_status);
                  fully_implicit_element_status *element_status_next_ptr = &element_status_next;
                  copy_segment_params_to_element_status (seg, element_status_next_ptr);
                  wsncs_backup.restore (wsncs, mp);
                  double qtot_next = q_tot_buf;
                  apply_fd_perturbation (element_status_next_ptr, qtot_next, var_id, +fd_eps);
                  wsncs->wsn_mixture_molar_rate = qtot_next;

                  double next_average_velocity = 0.0;
                  double next_gas_interfacial_tension = 0.0;
                  double next_bubble_rise_velocity = 0.0;
                  double next_liq_density = 0.0;
                  double next_diametr_dimless = 0.0;
                  double next_Kut_number = 0.0;
                  double next_flooding_velocity = 0.0;
                  double next_ksi = 0.0;
                  double next_eta = 0.0;
                  double next_gas_phase_velocity = 0.0;
                  double next_liquid_phase_velocity = 0.0;
                  double next_K_g = 0.0;
                  double next_avg_xi = 0.0;
                  double next_xi = 0.0;
                  double next_sigma_o = 0.0;
                  double next_sigma_w = 0.0;
                  double next_oil_water_interfacial_tension = 0.0;
                  double next_bubble_rise_velocity_OW = 0.0;
                  double next_C0_OW = 0.0;
                  double next_Vd_OW = 0.0;
                  double next_oil_velocity = 0.0;
                  double next_water_velocity = 0.0;
                  double next_alpha_g = 0.0;
                  double next_beta_o = 0.0;
                  double next_alpha_o = 0.0;
                  double next_alpha_w = 0.0;
                  double next_R_g = 0.0;
                  double next_R_o = 0.0;
                  unsigned int next_active_flags = 0U;
                  int next_inner_it = 0;

                  next_active_flags = df_fd_eval_mode_flag;
                  test_function (rep,
                                 seg,
                                 wsncs,
                                 element_status_next_ptr,
                                 mp,
                                 new_status,
                                 i_meshblock,
                                 current_therm_comp_input_props,
                                 itd,
                                 next_average_velocity,
                                 next_gas_interfacial_tension,
                                 next_bubble_rise_velocity,
                                 next_liq_density,
                                 next_diametr_dimless,
                                 next_Kut_number,
                                 next_flooding_velocity,
                                 next_ksi,
                                 next_eta,
                                 next_gas_phase_velocity,
                                 next_liquid_phase_velocity,
                                 next_K_g,
                                 next_avg_xi,
                                 next_xi,
                                 next_sigma_o,
                                 next_sigma_w,
                                 next_oil_water_interfacial_tension,
                                 next_bubble_rise_velocity_OW,
                                 next_C0_OW,
                                 next_Vd_OW,
                                 next_oil_velocity,
                                 next_water_velocity,
                                 next_alpha_g,
                                 next_beta_o,
                                 next_alpha_o,
                                 next_alpha_w,
                                 next_R_g,
                                 next_R_o,
                                 next_active_flags,
                                 next_inner_it,
                                 df_seed_initialized,
                                 prev_df_alpha_g_seed,
                                 prev_df_alpha_o_seed,
                                 prev_df_alpha_w_seed,
                                 prev_df_C0_seed,
                                 prev_df_Vd_seed,
                                 prev_df_C0OW_seed,
                                 prev_df_VdOW_seed);

                  const double next_C0 = wsncs->wsn_C_0;
                  const double next_drift_velocity = wsncs->wsn_drift_velocity;
                  const double next_C0OW_live = wsncs->wsn_C_0_OW;
                  const double next_VdOW_live = wsncs->wsn_drift_velocity_OW;
                  const double next_rho_avg = wsncs->rho_avg_DF;
                  std::vector<double> next_component_rate (mp.nc, 0.0);
                  for (unsigned int ic = 0; ic < mp.nc; ++ic)
                    next_component_rate[ic] = wsncs->wsn_component_rate[ic];
                  double next_mix_mass_rate = 0.0;
                  for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                    next_mix_mass_rate += next_component_rate[ic] * component_molar_weights[ic];

                  wsncs->D_average_volumetric_mixture_velocity_D_seg_vars[var_id] =
                      (next_average_velocity - prev_average_velocity) / (2.0 * fd_eps);
                  wsncs->D_C0_D_seg_vars[var_id] =
                      (next_C0 - prev_C0) / (2.0 * fd_eps);
                  wsncs->D_drift_velocity_D_seg_vars[var_id] =
                      (next_drift_velocity - prev_drift_velocity) / (2.0 * fd_eps);
                  wsncs->D_C0_OW_D_seg_vars[var_id] =
                      (next_C0OW_live - prev_C0OW_live) / (2.0 * fd_eps);
                  wsncs->D_drift_velocity_OW_D_seg_vars[var_id] =
                      (next_VdOW_live - prev_VdOW_live) / (2.0 * fd_eps);
                  wsncs->D_rho_avg_D_seg_vars[var_id] =
                      (next_rho_avg - prev_rho_avg) / (2.0 * fd_eps);
                  wsncs->D_mixture_mass_rate_D_seg_vars[var_id] =
                      (next_mix_mass_rate - prev_mix_mass_rate) / (2.0 * fd_eps);

                  for (unsigned int ic = 0; ic < mp.nc; ++ic)
                    {
                      wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + var_id] =
                          (next_component_rate[ic] - prev_component_rate[ic]) / (2.0 * fd_eps);
                    }
                }
            }

          wsncs_backup.restore (wsncs, mp);
          wsncs->wsn_mixture_molar_rate = q_tot_buf;
          base_active_flags = 0U;
          test_function (rep,
                         seg,
                         wsncs,
                         element_status,
                         mp,
                         new_status,
                         i_meshblock,
                         current_therm_comp_input_props,
                         itd,
                         base_average_velocity,
                         base_gas_interfacial_tension,
                         base_bubble_rise_velocity,
                         base_liq_density,
                         base_diametr_dimless,
                         base_Kut_number,
                         base_flooding_velocity,
                         base_ksi,
                         base_eta,
                         base_gas_phase_velocity,
                         base_liquid_phase_velocity,
                         base_K_g,
                         base_avg_xi,
                         base_xi,
                         base_sigma_o,
                         base_sigma_w,
                         base_oil_water_interfacial_tension,
                         base_bubble_rise_velocity_OW,
                         base_C0_OW,
                         base_Vd_OW,
                         base_oil_velocity,
                         base_water_velocity,
                         base_alpha_g,
                         base_beta_o,
                         base_alpha_o,
                         base_alpha_w,
                         base_R_g,
                         base_R_o,
                         base_active_flags,
                         base_inner_it,
                         df_seed_initialized,
                         prev_df_alpha_g_seed,
                         prev_df_alpha_o_seed,
                         prev_df_alpha_w_seed,
                         prev_df_C0_seed,
                         prev_df_Vd_seed,
                         prev_df_C0OW_seed,
                         prev_df_VdOW_seed);
          base_C_0 = wsncs->wsn_C_0;
          base_drift_velocity = wsncs->wsn_drift_velocity;
          if (seg.wsn->wsn_index != TOP_SEG_INDEX)
            {
              compute_mass_rate_and_mmw ();

              const double base_alpha_sum_seed = base_alpha_g + base_alpha_o + base_alpha_w;
              if (std::isfinite (base_alpha_sum_seed) && base_alpha_sum_seed > 1.0e-12
                  && base_alpha_g >= 0.0 && base_alpha_o >= 0.0 && base_alpha_w >= 0.0)
                {
                  if (df_seed_state_ptr == nullptr)
                    {
                      df_seed_state_store.push_back (df_persist_seed_state_t ());
                      df_seed_state_ptr = &df_seed_state_store.back ();
                      df_seed_state_ptr->seg_key = df_seed_seg_key;
                    }

                  df_seed_state_ptr->valid = true;
                  df_seed_state_ptr->alpha_g = base_alpha_g / base_alpha_sum_seed;
                  df_seed_state_ptr->alpha_o = base_alpha_o / base_alpha_sum_seed;
                  df_seed_state_ptr->alpha_w = base_alpha_w / base_alpha_sum_seed;
                  df_seed_state_ptr->C0 = (std::isfinite (base_C_0) && fabs (base_C_0) > tnm::min_compare) ? base_C_0 : 1.0;
                  df_seed_state_ptr->Vd = std::isfinite (base_drift_velocity) ? base_drift_velocity : 0.0;
                  df_seed_state_ptr->C0OW = (std::isfinite (base_C0_OW) && fabs (base_C0_OW) > tnm::min_compare) ? base_C0_OW : 1.0;
                  df_seed_state_ptr->VdOW = std::isfinite (base_Vd_OW) ? base_Vd_OW : 0.0;
                }
            }
        }

      if (seg.wsn->wsn_index == TOP_SEG_INDEX)
        {
          wsncs->wsn_mixture_mass_rate = 0.0;
          wsncs->wsn_mmw = 0.0;
          for (unsigned int ic : range (mp.nc))
            {
              wsncs->wsn_component_rate[ic] = seg.wsncs->wsn_mixture_molar_rate * component_z_for_flow[ic];
              wsncs->wsn_mixture_mass_rate += wsncs->wsn_component_rate[ic] * component_molar_weights[ic];
              wsncs->wsn_mmw += component_z_for_flow[ic] * component_molar_weights[ic];
            }
        }

void wells_compute_base::test_function (
                    report_system *rep,
                    wcb_segment_status &seg,
                    well_segment_node_computation_status *wsncs,
                    fully_implicit_element_status *element_status,
                    const model_parameters &mp,
                    const int new_status,
                    const meshblock_index &i_meshblock,
                    const thermal_composition_input_properties *current_therm_comp_input_props,
                    mc_msw_iteration_debug_data_t &itd,
                    double &average_volumetric_velocity,
                    double &gas_inter_tension,
                    double &bubble_rise_velocity_buf,
                    double &liq_density_buf,
                    double &diametr_dimless_buf,
                    double &Kut_number_buf,
                    double &flooding_velocity_buf,
                    double &ksi_buf,
                    double &eta_buf,
                    double &phase_gas_velocity_buf,
                    double &phase_liquid_velocity_buf,
                    double &K_g_buf,
                    double &avg_xi,
                    double &xi,
                    double &sigma_o,
                    double &sigma_w,
                    double &oil_inter_tension_buf,
                    double &bubble_rise_velocity_OW_buf,
                    double &C_OW_buf,
                    double &V_d_OW_buf,
                    double &oil_velocity_buf,
                    double &water_velocity_buf,
                    double &dbg_alpha_g,
                    double &dbg_beta_o,
                    double &dbg_alpha_o,
                    double &dbg_alpha_w,
                    double &dbg_R_g,
                    double &dbg_R_o,
                    unsigned int &dbg_active_flags,
                    int &dbg_inner_it,
                    const bool use_previous_df_seed,
                    const double prev_df_alpha_g_seed,
                    const double prev_df_alpha_o_seed,
                    const double prev_df_alpha_w_seed,
                    const double prev_df_C0_seed,
                    const double prev_df_Vd_seed,
                    const double prev_df_C0OW_seed,
                    const double prev_df_VdOW_seed)
{
  const bool is_fd_eval = (dbg_active_flags == 0x80000000U);

  average_volumetric_velocity = 0.0;
  gas_inter_tension = 0.0;
  bubble_rise_velocity_buf = 0.0;
  liq_density_buf = 0.0;
  diametr_dimless_buf = 0.0;
  Kut_number_buf = 0.0;
  flooding_velocity_buf = 0.0;
  ksi_buf = 0.0;
  eta_buf = 0.0;
  phase_gas_velocity_buf = 0.0;
  phase_liquid_velocity_buf = 0.0;
  K_g_buf = 0.0;
  avg_xi = 0.0;
  xi = 0.0;
  sigma_o = 0.0;
  sigma_w = 0.0;
  oil_inter_tension_buf = 0.0;
  bubble_rise_velocity_OW_buf = 0.0;
  C_OW_buf = 1.0;
  V_d_OW_buf = 0.0;
  oil_velocity_buf = 0.0;
  water_velocity_buf = 0.0;
  dbg_alpha_g = 0.0;
  dbg_beta_o = 0.0;
  dbg_alpha_o = 0.0;
  dbg_alpha_w = 0.0;
  dbg_R_g = 0.0;
  dbg_R_o = 0.0;
  dbg_active_flags = 0U;
  dbg_inner_it = 0;

  if (!wsncs || !element_status)
    return;

  if (wsncs->wsn_flow_dir == segment_flow_direction_t::from_parent_to_child)
    set_flow_direction_dependent_segment_params_to_element_status (seg, element_status, new_status);

  if (element_status->component_N_tot > tnm::min_compare)
    {
      if (auto err = run_flash <true> (rep,
                                       i_meshblock,
                                       element_status,
                                       current_therm_comp_input_props,
                                       itd);
          err != segments_solver_err_t::none)
        return;
    }

  const double q_tot_local = wsncs->wsn_mixture_molar_rate;
  fill_wsncs_from_element_status (wsncs, element_status, mp);
  wsncs->wsn_mixture_molar_rate = q_tot_local;

  const double surf_mult = internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT ();
  const double area = seg.wsn->pipe_props.area;
  const double diameter = seg.wsn->pipe_props.diameter;
  const double df_den_eps = tnm::min_compare;

  auto clamp_unit_interval_df =
    [&] (double v) -> double
    {
      if (!std::isfinite (v))
        return 0.5;
      if (v < 0.0)
        return 0.0;
      if (v > 1.0)
        return 1.0;
      return v;
    };

  auto solve_small_dense_df =
    [&] (std::vector<double> A_local,
         std::vector<double> b_local,
         std::vector<double> &x_local) -> bool
    {
      const unsigned int n_local = static_cast<unsigned int> (x_local.size ());
      if (n_local == 0)
        return false;

      for (unsigned int i = 0; i < n_local; ++i)
        {
          unsigned int pivot_row = i;
          double pivot_abs = fabs (A_local[i * n_local + i]);
          for (unsigned int r = i + 1; r < n_local; ++r)
            {
              const double cand_abs = fabs (A_local[r * n_local + i]);
              if (cand_abs > pivot_abs)
                {
                  pivot_abs = cand_abs;
                  pivot_row = r;
                }
            }

          if (pivot_abs <= 1.0e-20)
            return false;

          if (pivot_row != i)
            {
              for (unsigned int c = 0; c < n_local; ++c)
                std::swap (A_local[i * n_local + c], A_local[pivot_row * n_local + c]);
              std::swap (b_local[i], b_local[pivot_row]);
            }

          const double pivot = A_local[i * n_local + i];
          for (unsigned int c = i; c < n_local; ++c)
            A_local[i * n_local + c] /= pivot;
          b_local[i] /= pivot;

          for (unsigned int r = 0; r < n_local; ++r)
            {
              if (r == i)
                continue;
              const double factor = A_local[r * n_local + i];
              if (fabs (factor) <= 0.0)
                continue;
              for (unsigned int c = i; c < n_local; ++c)
                A_local[r * n_local + c] -= factor * A_local[i * n_local + c];
              b_local[r] -= factor * b_local[i];
            }
        }

      x_local = b_local;
      return true;
    };

  auto reconstruct_phase_rates_df =
    [&] (std::vector<double> &phase_rate_out) -> bool
    {
      if (element_status->component_N_tot <= tnm::min_compare)
        return false;

      std::vector<double> component_z_df (mp.nc, 0.0);
      double component_z_sum_df = 0.0;
      for (unsigned int ic = 0; ic < mp.nc; ++ic)
        {
          component_z_df[ic] = std::max (0.0, element_status->component_N[ic]);
          component_z_sum_df += component_z_df[ic];
        }
      if (component_z_sum_df <= 1.0e-12)
        return false;
      for (unsigned int ic = 0; ic < mp.nc; ++ic)
        component_z_df[ic] /= component_z_sum_df;

      const double q_tot_local_df = wsncs->wsn_mixture_molar_rate;
      const double flow_sign_df = (q_tot_local_df < 0.0) ? -1.0 : 1.0;
      std::vector<double> component_rate_total_df (mp.nc, 0.0);
      for (unsigned int ic = 0; ic < mp.nc; ++ic)
        component_rate_total_df[ic] = q_tot_local_df * component_z_df[ic];

      double best_residual = 1.0e300;
      std::vector<double> best_phase_rates (mp.np, 0.0);
      unsigned int best_mask = 0U;

      const unsigned int max_mask = (1U << mp.np);
      for (unsigned int mask = 1U; mask < max_mask; ++mask)
        {
          std::vector<unsigned int> active_phases_local;
          for (unsigned int ip = 0; ip < mp.np; ++ip)
            {
              if (((mask >> ip) & 1U) == 0U)
                continue;
              if (!(element_status->phase_xi[ip] > tnm::min_compare))
                {
                  active_phases_local.clear ();
                  break;
                }
              active_phases_local.push_back (ip);
            }
          if (active_phases_local.empty ())
            continue;

          const unsigned int k_local = static_cast<unsigned int> (active_phases_local.size ());
          std::vector<double> normal_matrix_local (k_local * k_local, 0.0);
          std::vector<double> rhs_local (k_local, 0.0);

          for (unsigned int ia = 0; ia < k_local; ++ia)
            {
              const unsigned int ip_a = active_phases_local[ia];
              for (unsigned int ja = 0; ja < k_local; ++ja)
                {
                  const unsigned int ip_b = active_phases_local[ja];
                  double accum = 0.0;
                  for (unsigned int ic = 0; ic < mp.nc; ++ic)
                    accum += element_status->component_phase_x[ic * mp.np + ip_a]
                             * element_status->component_phase_x[ic * mp.np + ip_b];
                  normal_matrix_local[ia * k_local + ja] = accum;
                }

              double rhs_accum = 0.0;
              for (unsigned int ic = 0; ic < mp.nc; ++ic)
                rhs_accum += element_status->component_phase_x[ic * mp.np + ip_a]
                             * component_rate_total_df[ic];
              rhs_local[ia] = rhs_accum;
            }

          std::vector<double> phase_rate_active_local (k_local, 0.0);
          if (!solve_small_dense_df (normal_matrix_local, rhs_local, phase_rate_active_local))
            continue;

          bool sign_consistent_local = true;
          for (unsigned int ia = 0; ia < k_local; ++ia)
            {
              if (flow_sign_df * phase_rate_active_local[ia] < -1.0e-12)
                {
                  sign_consistent_local = false;
                  break;
                }
            }
          if (!sign_consistent_local)
            continue;

          std::vector<double> phase_rate_full_local (mp.np, 0.0);
          for (unsigned int ia = 0; ia < k_local; ++ia)
            {
              const unsigned int ip = active_phases_local[ia];
              phase_rate_full_local[ip] =
                  flow_sign_df * std::max (0.0, flow_sign_df * phase_rate_active_local[ia]);
            }

          double residual_local = 0.0;
          for (unsigned int ic = 0; ic < mp.nc; ++ic)
            {
              double reconstructed_rate = 0.0;
              for (unsigned int ip = 0; ip < mp.np; ++ip)
                reconstructed_rate += element_status->component_phase_x[ic * mp.np + ip]
                                      * phase_rate_full_local[ip];
              const double defect = reconstructed_rate - component_rate_total_df[ic];
              residual_local += defect * defect;
            }

          if (residual_local < best_residual)
            {
              best_residual = residual_local;
              best_phase_rates = phase_rate_full_local;
              best_mask = mask;
            }
        }

      if (!(best_mask > 0U))
        return false;

      phase_rate_out = best_phase_rates;
      return true;
    };

  double beta_o_first_seed = 0.5;
  double oil_superficial_velocity_input_seed_dbg = 0.0;
  double water_superficial_velocity_input_seed_dbg = 0.0;
  if (mp.model_type_3_phase ())
    {
      std::vector<double> phase_rate_seed_df (mp.np, 0.0);
      if (reconstruct_phase_rates_df (phase_rate_seed_df))
        {
          const double q_o_abs = fabs (phase_rate_seed_df[PHASE_OIL]);
          const double q_w_abs = fabs (phase_rate_seed_df[PHASE_WATER]);
          if (q_o_abs + q_w_abs > 1.0e-12)
            beta_o_first_seed = q_o_abs / (q_o_abs + q_w_abs);

          if (element_status->phase_xi[PHASE_OIL] > tnm::min_compare)
            oil_superficial_velocity_input_seed_dbg = tnav_div (phase_rate_seed_df[PHASE_OIL], area * element_status->phase_xi[PHASE_OIL]);
          if (element_status->phase_xi[PHASE_WATER] > tnm::min_compare)
            water_superficial_velocity_input_seed_dbg = tnav_div (phase_rate_seed_df[PHASE_WATER], area * element_status->phase_xi[PHASE_WATER]);
        }
    }

  if (use_previous_df_seed)
    {
      const double alpha_l_prev_seed = prev_df_alpha_o_seed + prev_df_alpha_w_seed;
      if (alpha_l_prev_seed > df_den_eps)
        beta_o_first_seed = prev_df_alpha_o_seed / alpha_l_prev_seed;
    }
  beta_o_first_seed = clamp_unit_interval_df (beta_o_first_seed);

  average_volumetric_velocity = tnav_div (q_tot_local, area);
  avg_xi = element_status->avg_xi;
  xi = element_status->phase_xi[PHASE_GAS];

  if (seg.wsn->wsn_index == TOP_SEG_INDEX)
    {
      wsncs->wsn_C_0 = 1.0;
      wsncs->wsn_drift_velocity = 0.0;
      wsncs->wsn_C_0_OW = 1.0;
      wsncs->wsn_drift_velocity_OW = 0.0;
      return;
    }

  auto compute_inclination_multiplier =
    [&] () -> double
    {
      if (fabs (seg.wsn->pipe_props.depth_change) < tnm::min_compare)
        return 0.0;

      double length = fabs (seg.wsn->pipe_props.length);
      double depth = fabs (seg.wsn->pipe_props.depth_change);
      if (length < tnm::min_compare)
        return 0.0;

      const double cos_theta = (depth > length - tnm::min_compare) ? 1.0 : depth / length;
      if (depth > length - tnm::min_compare)
        return 1.0;

      return tnav_pow (cos_theta, 0.5) * tnav_pow (1.0 + sqrt (std::max (0.0, 1.0 - cos_theta * cos_theta)), 2.0);
    };

  const double drift_incl_mult = compute_inclination_multiplier ();

  double alpha_g_seed = 1.0 / 3.0;
  double alpha_o_seed = 1.0 / 3.0;
  double alpha_w_seed = 1.0 / 3.0;
  if (use_previous_df_seed)
    {
      alpha_g_seed = prev_df_alpha_g_seed;
      alpha_o_seed = prev_df_alpha_o_seed;
      alpha_w_seed = prev_df_alpha_w_seed;
    }

  const double alpha_l_seed = alpha_o_seed + alpha_w_seed;

  double gas_phase_velocity_seed = average_volumetric_velocity;
  double liquid_phase_velocity_seed = average_volumetric_velocity;
  double oil_phase_velocity_seed = average_volumetric_velocity;
  double water_phase_velocity_seed = average_volumetric_velocity;

  if (use_previous_df_seed)
    {
      gas_phase_velocity_seed = prev_df_C0_seed * average_volumetric_velocity + prev_df_Vd_seed;

      if (fabs (1.0 - alpha_g_seed) > df_den_eps)
        {
          liquid_phase_velocity_seed =
              ((1.0 - alpha_g_seed * prev_df_C0_seed) / (1.0 - alpha_g_seed)) * average_volumetric_velocity
              - (alpha_g_seed / (1.0 - alpha_g_seed)) * prev_df_Vd_seed;
        }

      const double beta_o_seed_local = (fabs (alpha_l_seed) > df_den_eps)
                                       ? (alpha_o_seed / alpha_l_seed)
                                       : 0.0;

      oil_phase_velocity_seed = prev_df_C0OW_seed * liquid_phase_velocity_seed + prev_df_VdOW_seed;

      if (fabs (1.0 - beta_o_seed_local) > df_den_eps)
        {
          water_phase_velocity_seed =
              ((1.0 - beta_o_seed_local * prev_df_C0OW_seed) / (1.0 - beta_o_seed_local)) * liquid_phase_velocity_seed
              - (beta_o_seed_local / (1.0 - beta_o_seed_local)) * prev_df_VdOW_seed;
        }
      else
        {
          water_phase_velocity_seed = liquid_phase_velocity_seed;
        }
    }

  phase_holdups_DF prev_holdups (alpha_g_seed,
                                 alpha_l_seed,
                                 alpha_o_seed,
                                 alpha_w_seed);
  phase_holdups_DF temp_holdups (0.0, 0.0, 0.0, 0.0);
  phase_holdups_DF new_holdups (0.0, 0.0, 0.0, 0.0);

  phase_vel_DF prev_vels (gas_phase_velocity_seed,
                          liquid_phase_velocity_seed,
                          oil_phase_velocity_seed,
                          water_phase_velocity_seed);
  phase_vel_DF new_vels (0.0, 0.0, 0.0, 0.0);

  const double holdup_tolerance = 1.e-4;
  const int max_df_it = 10;
  double error = 100.0;
  int df_it = 0;

  // --- superficial velocities from input molar rates (FIX #4) ---------------
  // Константы в итерации: vsg, vso, vsw вычисляются из wsn_mixture_molar_rate
  // и фазовых мольных долей через reconstruct_phase_rates_df. Это даёт
  // фикспоинт по одной переменной αg (и βo), а не по четырём.
  double vsg_input = 0.0;
  double vso_input = 0.0;
  double vsw_input = 0.0;
  {
    std::vector<double> phase_rate_in (mp.np, 0.0);
    if (reconstruct_phase_rates_df (phase_rate_in))
      {
        if (element_status->phase_xi[PHASE_GAS] > tnm::min_compare)
          vsg_input = tnav_div (phase_rate_in[PHASE_GAS],
                                area * element_status->phase_xi[PHASE_GAS]);

        if (mp.model_type_3_phase () && !mp.gaswat_or_co2store_or_co2storage ())
          {
            if (element_status->phase_xi[PHASE_OIL] > tnm::min_compare)
              vso_input = tnav_div (phase_rate_in[PHASE_OIL],
                                    area * element_status->phase_xi[PHASE_OIL]);
            if (element_status->phase_xi[PHASE_WATER] > tnm::min_compare)
              vsw_input = tnav_div (phase_rate_in[PHASE_WATER],
                                    area * element_status->phase_xi[PHASE_WATER]);
          }
        else
          {
            if (element_status->phase_xi[PHASE_WATER] > tnm::min_compare)
              vsw_input = tnav_div (phase_rate_in[PHASE_WATER],
                                    area * element_status->phase_xi[PHASE_WATER]);
          }
      }
  }
  const double vsl_input = vso_input + vsw_input;
  const double mixture_velocity_input = vsg_input + vsl_input;
  // ---------------------------------------------------------------------------

  while (error > holdup_tolerance && df_it < max_df_it)
    {
      temp_holdups = prev_holdups;
      new_holdups = prev_holdups;
      new_vels = prev_vels;

      // -------------------- gas / liquid stage
      if (temp_holdups.liquid > tnm::min_compare && element_status->p >= 0.0)
        {
          const double gas_density = element_status->phase_rho[PHASE_GAS];
          // FIX #4: константные superficial-velocities из входных расходов.
          const double gas_superficial_velocity = vsg_input;
          const double liq_superficial_velocity = vsl_input;
          const double mixture_velocity = mixture_velocity_input;

          double oil_api = 0.0;
          double liq_density = 0.0;
          if (mp.model_type_3_phase () && !(mp.gaswat_or_co2store_or_co2storage ()))
            {
              liq_density = (temp_holdups.water * element_status->phase_rho[PHASE_WATER]
                             + temp_holdups.oil * element_status->phase_rho[PHASE_OIL])
                            / temp_holdups.liquid;
              if (element_status->phase_rho[PHASE_OIL] > tnm::min_compare)
                {
                  oil_api = internal_const::API_to_input_density_mult (units_system_t::metric)
                            / element_status->phase_rho[PHASE_OIL]
                            - internal_const::API_to_input_density_add ();
                }
            }
          else
            {
              liq_density = element_status->phase_rho[PHASE_WATER];
            }

          std::vector<double> fake_vector;
          double sigma_o_local = 0.0;
          double sigma_w_local = 0.0;
          double D_sigma_o_D_p_dummy = 0.0;
          double D_sigma_w_D_p_dummy = 0.0;
          double gas_liq_interfacial_tension =
              surf_mult * pipe_gas_liq_interfacial_tension_holdup_weightening (
                  element_status->p * converter_metric_to_field.pressure_mult (),
                  160.0,
                  oil_api,
                  temp_holdups.oil,
                  temp_holdups.water,
                  nullptr,
                  fake_vector,
                  sigma_o_local,
                  sigma_w_local,
                  D_sigma_o_D_p_dummy,
                  D_sigma_w_D_p_dummy);

          gas_liq_interfacial_tension = std::max (gas_liq_interfacial_tension, tnm::min_compare);

          double bubble_rise_velocity = tnav_pow (
              gas_liq_interfacial_tension * internal_const::grav_metric () * fabs (liq_density - gas_density)
                  / (liq_density * liq_density),
              0.25);

          double diametr_dimless =
              sqrt (internal_const::grav_metric () * fabs (liq_density - gas_density) / gas_liq_interfacial_tension)
              * diameter;

          double linear_interpolation_derivative_dummy = 0.0;
          double Kut_number = compute_critical_Kutateladze_number_by_diametr (diametr_dimless,
                                                                              linear_interpolation_derivative_dummy);

          double flooding_velocity = 0.0;
          if (gas_density > tnm::min_compare_well_limit)
            flooding_velocity = Kut_number * sqrt (liq_density / gas_density) * bubble_rise_velocity;

          double profile_param_C0 = 1.0;
          const double A = 1.2;
          const double B = 0.3;
          const double Fv = 1.0;
          if (fabs (A - 1.0) > tnm::min_compare)
            {
              double beta = temp_holdups.gas;
              if (flooding_velocity > tnm::min_compare)
                beta *= std::max (1.0, Fv * fabs (mixture_velocity) / flooding_velocity);

              // FIX #1: β* должна обрезаться до 0 при β < B, а не браться по модулю.
              double beta_norm = (beta - B) / (1.0 - B);
              if (beta_norm < 0.0)
                beta_norm = 0.0;
              else if (beta_norm > 1.0)
                beta_norm = 1.0;

              profile_param_C0 = A / (1.0 + (A - 1.0) * beta_norm * beta_norm);
              if (fabs (profile_param_C0) < tnm::min_division)
                profile_param_C0 = tnm::min_division;
            }

          double K_g_low = 1.53 / profile_param_C0;
          double K_g_high = Kut_number;
          double K_g = K_g_low;
          if (temp_holdups.gas < 0.2 + tnm::min_compare)
            K_g = K_g_low;
          else if (temp_holdups.gas > 0.4 - tnm::min_compare)
            K_g = K_g_high;
          else
            K_g = interpolate_y_against_x (temp_holdups.gas, 0.2, 0.4, K_g_low, K_g_high);

          // FIX #2: sqrt охватывает весь знаменатель drift velocity (Eclipse 8.78).
          const double vd_denom_inner =
              temp_holdups.gas * profile_param_C0 * (gas_density / liq_density)
            + 1.0 - temp_holdups.gas * profile_param_C0;
          double drift_velocity =
              (1.0 - temp_holdups.gas * profile_param_C0) * profile_param_C0 * K_g * bubble_rise_velocity
              / sqrt (std::max (vd_denom_inner, tnm::min_compare));

          if (drift_incl_mult == 0.0)
            drift_velocity = 0.0;
          else
            drift_velocity *= drift_incl_mult;

          new_vels.gas = profile_param_C0 * mixture_velocity + drift_velocity;

          if (fabs (new_vels.gas) > tnm::min_compare)
            new_holdups.gas = gas_superficial_velocity / new_vels.gas;
          else
            new_holdups.gas = 0.0;

          if (new_holdups.gas > 1.0 - tnm::min_compare)
            {
              new_holdups.gas = 1.0;
              new_holdups.liquid = 0.0;
              new_vels.liquid = 0.0;
            }
          else
            {
              new_holdups.liquid = 1.0 - new_holdups.gas;
              if (fabs (new_holdups.liquid) > tnm::min_compare)
                new_vels.liquid = liq_superficial_velocity / new_holdups.liquid;
              else
                new_vels.liquid = 0.0;
            }

          gas_inter_tension = gas_liq_interfacial_tension;
          bubble_rise_velocity_buf = bubble_rise_velocity;
          liq_density_buf = liq_density;
          diametr_dimless_buf = diametr_dimless;
          Kut_number_buf = Kut_number;
          flooding_velocity_buf = flooding_velocity;
          ksi_buf = 0.0;
          eta_buf = 0.0;
          phase_gas_velocity_buf = new_vels.gas;
          phase_liquid_velocity_buf = new_vels.liquid;
          K_g_buf = K_g;
          sigma_o = sigma_o_local;
          sigma_w = sigma_w_local;
          wsncs->wsn_C_0 = profile_param_C0;
          wsncs->wsn_drift_velocity = drift_velocity;

          if (!is_fd_eval)
            {
              PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS, "MERGEN: Well {}: it = {}, Seg_idx = {}, error = {}, average_volumetric_mixture_velocity = {}, element_status->avg_xi = {}, "
                                                                               "seg.wsn->pipe_props.area = {}, wsncs->wsn_mixture_molar_rate = {}, wsncs->wsn_C_0 = {}, wsncs->wsn_drift_velocity = {}, "
                                                                               "gas_liq_interfacial_tension = {}, gas_density = {}, liquid_density = {}, diametr_dimless = {}, bubble_rise_velocity = {}, "
                                                                               "Kut_number = {}, flooding_velocity = {}, K_g = {}, element_status->phase_S[PHASE_GAS] = {}, "
                                                                               "seg.wsn->pipe_props.diameter = {}, phase_S_WAT = {}, phase_S_OIL = {}, phase_S_GAS = {}, "
                                                                               "wsncs->wsn_C_0_OW = {}, wsncs->wsn_drift_velocity_OW = {}, "
                                                                               "new_holdups.gas = {}, new_holdups.liquid = {}, new_holdups.oil = {}, new_holdups.water = {} "
                                                                               "new_vels.gas = {}, new_vels.liquid = {}, new_vels.oil = {}, new_vels.water = {} \n",
                                                     wcb_wis->get_well_name (), df_it, seg.wsn->wsn_index, error, average_volumetric_velocity, element_status->avg_xi,
                                                     seg.wsn->pipe_props.area, wsncs->wsn_mixture_molar_rate, wsncs->wsn_C_0, wsncs->wsn_drift_velocity,
                                                     gas_liq_interfacial_tension, gas_density, liq_density, diametr_dimless, bubble_rise_velocity,
                                                     Kut_number, flooding_velocity, K_g, element_status->phase_S[PHASE_GAS],
                                                     seg.wsn->pipe_props.diameter, element_status->phase_S[PHASE_WATER], element_status->phase_S[PHASE_OIL], element_status->phase_S[PHASE_GAS],
                                                     wsncs->wsn_C_0_OW, wsncs->wsn_drift_velocity_OW, new_holdups.gas, new_holdups.liquid, new_holdups.oil, new_holdups.water,
                                                     new_vels.gas, new_vels.liquid, new_vels.oil, new_vels.water);
            }
        }

      // -------------------- oil / water stage
      if (mp.model_type_3_phase ())
        {
          phase_holdups_DF ow_prev (0.0, new_holdups.liquid, 0.0, 0.0);
          if (new_holdups.liquid > tnm::min_division)
            {
              if (df_it == 0 && !use_previous_df_seed)
                {
                  ow_prev.oil = beta_o_first_seed;
                  ow_prev.water = 1.0 - beta_o_first_seed;
                }
              else
                {
                  const double prev_liquid_holdup = std::max (temp_holdups.oil + temp_holdups.water, tnm::min_division);
                  ow_prev.oil = temp_holdups.oil / prev_liquid_holdup;
                  ow_prev.water = temp_holdups.water / prev_liquid_holdup;
                }
            }

          if (!is_fd_eval)
            {
              PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                   "MERGEN-OW-SEED: Well {}: Seg_idx = {}, df_it = {}, beta_o_first_seed = {}, qso_seed = {}, qsw_seed = {}, ow_prev.oil = {}, ow_prev.water = {}\n",
                   wcb_wis->get_well_name (), seg.wsn->wsn_index, df_it, beta_o_first_seed,
                   oil_superficial_velocity_input_seed_dbg, water_superficial_velocity_input_seed_dbg,
                   ow_prev.oil, ow_prev.water);
            }

          if (ow_prev.oil > tnm::min_compare && ow_prev.water > tnm::min_compare && element_status->p >= 0.0)
            {
              // FIX #4: oil/water superficial velocities из входных расходов,
              // делим на текущий αl (новый после gas/liquid стадии).
              const double alpha_l_current =
                  std::max (new_holdups.liquid, tnm::min_division);
              const double oil_superficial_velocity   = vso_input / alpha_l_current;
              const double water_superficial_velocity = vsw_input / alpha_l_current;
              const double liquid_velocity = oil_superficial_velocity + water_superficial_velocity;

              if (fabs (liquid_velocity) < tnm::min_compare)
                {
                  new_vels.oil = 0.0;
                  new_vels.water = 0.0;
                  new_holdups.oil = ow_prev.oil * new_holdups.liquid;
                  new_holdups.water = ow_prev.water * new_holdups.liquid;
                }
              else
                {
                  double profile_param_C0_OW = 1.0;
                  if (ow_prev.oil < 0.4)
                    profile_param_C0_OW = 1.2;
                  else if (ow_prev.oil > 0.7)
                    profile_param_C0_OW = 1.0;
                  else
                    profile_param_C0_OW = interpolate_y_against_x (ow_prev.oil, 0.4, 0.7, 1.2, 1.0);

                  double oil_api = internal_const::API_to_input_density_mult (units_system_t::metric)
                                   / element_status->phase_rho[PHASE_OIL]
                                   - internal_const::API_to_input_density_add ();
                  double gas_oil_interfacial_tension =
                      surf_mult * pipe_gas_oil_interfacial_tension (
                          oil_api,
                          element_status->p * converter_metric_to_field.pressure_mult (),
                          160.0);
                  double gas_wat_interfacial_tension =
                      surf_mult * pipe_gas_wat_interfacial_tension (
                          element_status->p * converter_metric_to_field.pressure_mult (),
                          160.0);
                  // FIX #10.1: σow = |σgo - σwg|, без весов по holdup'ам (Eclipse стр.697).
                  double wat_oil_interfacial_tension =
                      fabs (gas_oil_interfacial_tension - gas_wat_interfacial_tension);

                  double bubble_rise_velocity_OW = tnav_pow (
                      wat_oil_interfacial_tension * internal_const::grav_metric ()
                          * fabs (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL])
                          / (element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER]),
                      0.25);

                  double drift_velocity_OW = 1.53 * bubble_rise_velocity_OW * tnav_pow (1.0 - ow_prev.oil, 2.0);
                  if (drift_incl_mult == 0.0)
                    drift_velocity_OW = 0.0;
                  else
                    drift_velocity_OW *= drift_incl_mult;

                  new_vels.oil = profile_param_C0_OW * liquid_velocity + drift_velocity_OW;
                  if (fabs (new_vels.oil) > tnm::min_compare)
                    new_holdups.oil = oil_superficial_velocity / new_vels.oil;
                  else
                    new_holdups.oil = 0.0;

                  if (new_holdups.oil > 1.0 - tnm::min_compare)
                    {
                      new_holdups.oil = 1.0;
                      new_holdups.water = 0.0;
                      new_vels.water = 0.0;
                    }
                  else
                    {
                      new_holdups.water = 1.0 - new_holdups.oil;
                      if (fabs (new_holdups.water) > tnm::min_compare)
                        new_vels.water = water_superficial_velocity / new_holdups.water;
                      else
                        new_vels.water = 0.0;
                    }

                  new_holdups.oil *= new_holdups.liquid;
                  new_holdups.water *= new_holdups.liquid;

                  oil_inter_tension_buf = wat_oil_interfacial_tension;
                  bubble_rise_velocity_OW_buf = bubble_rise_velocity_OW;
                  C_OW_buf = profile_param_C0_OW;
                  V_d_OW_buf = drift_velocity_OW;
                  oil_velocity_buf = new_vels.oil;
                  water_velocity_buf = new_vels.water;
                  wsncs->wsn_C_0_OW = profile_param_C0_OW;
                  wsncs->wsn_drift_velocity_OW = drift_velocity_OW;
                }
            }
          else
            {
              new_holdups.oil = 0.0;
              new_holdups.water = new_holdups.liquid;
              new_vels.oil = new_vels.liquid;
              new_vels.water = new_vels.liquid;
              C_OW_buf = 1.0;
              V_d_OW_buf = 0.0;
              oil_velocity_buf = new_vels.oil;
              water_velocity_buf = new_vels.water;
              wsncs->wsn_C_0_OW = 1.0;
              wsncs->wsn_drift_velocity_OW = 0.0;
            }
        }
      else if (mp.is_water_gas ())
        {
          new_holdups.oil = 0.0;
          new_holdups.water = new_holdups.liquid;
          new_vels.oil = 0.0;
          new_vels.water = new_vels.liquid;
          oil_velocity_buf = 0.0;
          water_velocity_buf = new_vels.water;
          C_OW_buf = 1.0;
          V_d_OW_buf = 0.0;
          wsncs->wsn_C_0_OW = 1.0;
          wsncs->wsn_drift_velocity_OW = 0.0;
        }

      error = tnav_sqr (prev_holdups.oil - new_holdups.oil)
              + tnav_sqr (prev_holdups.water - new_holdups.water)
              + tnav_sqr (prev_holdups.gas - new_holdups.gas);
      error = sqrt (error);

      prev_holdups = new_holdups;
      prev_vels = new_vels;
      ++df_it;
    }

  dbg_inner_it = df_it;
  dbg_alpha_g = prev_holdups.gas;
  dbg_alpha_o = prev_holdups.oil;
  dbg_alpha_w = prev_holdups.water;
  dbg_beta_o = (fabs (prev_holdups.liquid) > tnm::min_division)
               ? prev_holdups.oil / prev_holdups.liquid
               : 0.0;
  dbg_R_g = prev_holdups.gas * prev_vels.gas - prev_vels.gas * prev_holdups.gas;
  dbg_R_o = prev_holdups.oil * prev_vels.oil - prev_vels.oil * prev_holdups.oil;
  dbg_active_flags = 0U;

  phase_gas_velocity_buf = prev_vels.gas;
  phase_liquid_velocity_buf = prev_vels.liquid;
  oil_velocity_buf = prev_vels.oil;
  water_velocity_buf = prev_vels.water;

  if (!is_fd_eval)
    {
      wsncs->phase_S[PHASE_GAS] = prev_holdups.gas;
      wsncs->phase_S[PHASE_OIL] = prev_holdups.oil;
      wsncs->phase_S[PHASE_WATER] = prev_holdups.water;
    }

  wsncs->rho_avg_DF = element_status->phase_rho[PHASE_GAS] * prev_holdups.gas
                      + element_status->phase_rho[PHASE_OIL] * prev_holdups.oil
                      + element_status->phase_rho[PHASE_WATER] * prev_holdups.water;

  for (auto ic = mp.nc0; ic < mp.nc; ++ic)
    wsncs->wsn_component_rate[ic] = 0.0;

  for (unsigned int ip = 0; ip < mp.np; ++ip)
    {
      double phase_velocity = 0.0;
      double phase_holdup = 0.0;
      if (ip == PHASE_GAS)
        {
          phase_velocity = prev_vels.gas;
          phase_holdup = prev_holdups.gas;
        }
      else if (ip == PHASE_OIL)
        {
          phase_velocity = prev_vels.oil;
          phase_holdup = prev_holdups.oil;
        }
      else
        {
          phase_velocity = prev_vels.water;
          phase_holdup = prev_holdups.water;
        }

      const double phase_superficial_flux = phase_velocity * phase_holdup * area;
      for (auto ic = mp.nc0; ic < mp.nc; ++ic)
        {
          wsncs->wsn_component_rate[ic] +=
              phase_superficial_flux
              * element_status->component_phase_x[ic * mp.np + ip]
              * element_status->phase_xi[ip];
        }
    }
}