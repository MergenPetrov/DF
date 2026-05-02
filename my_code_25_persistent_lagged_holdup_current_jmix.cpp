if (element_status)
        {
          const double q_tot_buf = wsncs->wsn_mixture_molar_rate;
          const double surf_mult = internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT ();
          const unsigned int nseg_vars_fd = 1U + mp.nc + 1U;
          // 0              -> pressure
          // 1 .. mp.nc     -> component_N[id - 1]
          // nseg_vars_fd-1 -> q_tot
          const unsigned int fd_var_id = nseg_vars_fd - nseg_vars_fd + 0;

          // ----------------------------------------------------------------------------
          // Persistent lagged DF input state.
          //
          // The previous fix tried to recover previous-Newton holdups from wsncs->phase_S.
          // The new log showed that this scratch object is rebuilt from element_status/Flash
          // before the next outer Newton evaluation, so phase_inputs_from_lagged_map stayed 0.
          // Keep a small per-segment state in this code path instead.
          //
          // First production evaluation for a segment: no entry -> Flash inputs.
          // Later production evaluations: use previous DF holdups as lagged phase
          // split/seed, but keep jmix tied to the current Newton q_tot.
          // Therefore alpha_input is frozen in the lagged branch, while
          // d(jsg/jso/jsw)/d(seg_var) retains the current d(jmix)/d(seg_var) coupling.
          // ----------------------------------------------------------------------------
          struct df_lagged_phase_input_state_t
            {
              const void *seg_key = nullptr;
              unsigned int seg_idx = 0U;
              bool valid = false;
              unsigned long generation = 0UL;

              double alpha_g = 0.0;
              double alpha_o = 0.0;
              double alpha_w = 0.0;

              // Physical signed superficial velocities [m/s].
              double jsg_raw = 0.0;
              double jso_raw = 0.0;
              double jsw_raw = 0.0;
              double jmix_raw = 0.0;

              double rho_avg = 0.0;
              double qtot_raw = 0.0;
            };

          static std::vector<df_lagged_phase_input_state_t> df_lagged_phase_input_states;

          const void *df_lagged_segment_key = static_cast<const void *> (seg.wsn);
          const unsigned int df_lagged_segment_idx = seg.wsn ? seg.wsn->wsn_index : 0U;

          auto df_normalize_lagged_state = [&] (df_lagged_phase_input_state_t &st) -> bool
            {
              if (!st.valid || !st.seg_key)
                return false;

              if (!std::isfinite (st.alpha_g) || !std::isfinite (st.alpha_o) || !std::isfinite (st.alpha_w))
                return false;
              if (st.alpha_g < -1.0e-10 || st.alpha_o < -1.0e-10 || st.alpha_w < -1.0e-10)
                return false;

              st.alpha_g = std::max (0.0, st.alpha_g);
              st.alpha_o = std::max (0.0, st.alpha_o);
              st.alpha_w = std::max (0.0, st.alpha_w);

              const double sum_alpha = st.alpha_g + st.alpha_o + st.alpha_w;
              if (!std::isfinite (sum_alpha) || sum_alpha <= tnm::min_compare)
                return false;

              st.alpha_g /= sum_alpha;
              st.alpha_o /= sum_alpha;
              st.alpha_w /= sum_alpha;

              // Stored superficial velocities are diagnostic in the current
              // lagged-holdup mode.  Do not invalidate an otherwise good holdup
              // state because a diagnostic velocity is missing.
              if (!std::isfinite (st.jsg_raw)) st.jsg_raw = 0.0;
              if (!std::isfinite (st.jso_raw)) st.jso_raw = 0.0;
              if (!std::isfinite (st.jsw_raw)) st.jsw_raw = 0.0;

              const double jmix_from_phases = st.jsg_raw + st.jso_raw + st.jsw_raw;
              if (!std::isfinite (st.jmix_raw)
                  || fabs (st.jmix_raw - jmix_from_phases) > 1.0e-9 * std::max (1.0, fabs (jmix_from_phases)))
                st.jmix_raw = std::isfinite (jmix_from_phases) ? jmix_from_phases : 0.0;

              if (!std::isfinite (st.rho_avg) || st.rho_avg <= tnm::min_compare)
                st.rho_avg = 1.0;

              return true;
            };

          auto df_load_lagged_state = [&] (df_lagged_phase_input_state_t &out) -> bool
            {
              for (const auto &st : df_lagged_phase_input_states)
                {
                  if (st.seg_key == df_lagged_segment_key
                      && st.seg_idx == df_lagged_segment_idx
                      && st.valid)
                    {
                      out = st;
                      out.valid = df_normalize_lagged_state (out);
                      return out.valid;
                    }
                }

              out = df_lagged_phase_input_state_t ();
              out.seg_key = df_lagged_segment_key;
              out.seg_idx = df_lagged_segment_idx;
              out.valid = false;
              return false;
            };

          auto df_store_lagged_state = [&] (df_lagged_phase_input_state_t st)
            {
              st.seg_key = df_lagged_segment_key;
              st.seg_idx = df_lagged_segment_idx;
              if (!df_normalize_lagged_state (st))
                return;

              for (auto &old_st : df_lagged_phase_input_states)
                {
                  if (old_st.seg_key == st.seg_key && old_st.seg_idx == st.seg_idx)
                    {
                      st.generation = old_st.generation + 1UL;
                      old_st = st;
                      return;
                    }
                }

              st.generation = 1UL;
              df_lagged_phase_input_states.push_back (st);
            };

          df_lagged_phase_input_state_t df_lagged_input_state;
          const bool df_lagged_input_state_valid =
              df_load_lagged_state (df_lagged_input_state);

          auto df_seed_wsncs_for_test_function = [&] (well_segment_node_computation_status *dst)
            {
              if (!dst)
                return;

              if (df_lagged_input_state_valid)
                {
                  dst->phase_S[PHASE_GAS] = df_lagged_input_state.alpha_g;
                  dst->phase_S[PHASE_OIL] = df_lagged_input_state.alpha_o;
                  dst->phase_S[PHASE_WATER] = df_lagged_input_state.alpha_w;
                  dst->rho_avg_DF = std::max (tnm::min_compare, df_lagged_input_state.rho_avg);

                  // Do not pass frozen jmix/jsp to test_function(): finite-difference
                  // probes must use the current q_tot-derived jmix, same as production.
                  if (nseg_vars_fd >= 1U)
                    dst->D_average_volumetric_mixture_velocity_D_seg_vars[0] = 0.0;
                }
              else
                {
                  // Force test_function() and the old wsncs-based validity test to fall back
                  // to Flash on the first evaluation for this segment.
                  dst->rho_avg_DF = 0.0;
                  if (nseg_vars_fd >= 1U)
                    dst->D_average_volumetric_mixture_velocity_D_seg_vars[0] = 0.0;
                }
            };

          df_seed_wsncs_for_test_function (wsncs);

          auto apply_fd_perturbation =
            [&] (fully_implicit_element_status *es, double &qtot_work, double delta)
            {
              if (fd_var_id == 0)
                {
                  es->p += delta;
                }
              else if (fd_var_id < 1U + mp.nc)
                {
                  es->component_N[fd_var_id - 1] += delta;
                  es->component_N_tot += delta;
                }
              else
                {
                  qtot_work += delta;
                }
            };

          double fd_base_value = 1.0;
          if (fd_var_id == 0)
            fd_base_value = std::max (1.0, fabs (element_status->p));
          else if (fd_var_id < 1U + mp.nc)
            fd_base_value = std::max (1.0, fabs (element_status->component_N[fd_var_id - 1]));
          else
            fd_base_value = std::max (1.0, fabs (q_tot_buf));

          (void) fd_base_value;

          const double fd_eps = 1.e-3;

          // Backup live segment computation status so finite-difference calls do not
          // leak temporary state into the production Jacobian assembly.
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
              std::vector<double> D_rho_avg_D_seg_vars;
              std::vector<double> D_phase_holdup_D_seg_vars;
              std::vector<double> D_q_c_D_seg_vars;
              std::vector<double> D_mixture_mass_rate_D_seg_vars;
              std::vector<double> D_average_volumetric_mixture_velocity_D_seg_vars;

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

                D_rho_avg_D_seg_vars.resize (1U + mp_in.nc + 1U, 0.0);
                D_mixture_mass_rate_D_seg_vars.resize (1U + mp_in.nc + 1U, 0.0);
                D_average_volumetric_mixture_velocity_D_seg_vars.resize (1U + mp_in.nc + 1U, 0.0);
                D_phase_holdup_D_seg_vars.resize ((1U + mp_in.nc + 1U) * mp_in.np, 0.0);
                D_q_c_D_seg_vars.resize ((1U + mp_in.nc + 1U) * mp_in.nc, 0.0);
                for (unsigned int id = 0; id < 1U + mp_in.nc + 1U; ++id)
                  {
                    D_rho_avg_D_seg_vars[id] = src->D_rho_avg_D_seg_vars[id];
                    D_mixture_mass_rate_D_seg_vars[id] = src->D_mixture_mass_rate_D_seg_vars[id];
                    D_average_volumetric_mixture_velocity_D_seg_vars[id] = src->D_average_volumetric_mixture_velocity_D_seg_vars[id];
                  }
                for (unsigned int k = 0; k < (1U + mp_in.nc + 1U) * mp_in.np; ++k)
                  D_phase_holdup_D_seg_vars[k] = src->D_phase_holdup_D_seg_vars[k];
                for (unsigned int k = 0; k < (1U + mp_in.nc + 1U) * mp_in.nc; ++k)
                  D_q_c_D_seg_vars[k] = src->D_q_c_D_seg_vars[k];
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

                for (unsigned int id = 0; id < 1U + mp_in.nc + 1U; ++id)
                  {
                    dst->D_rho_avg_D_seg_vars[id] = D_rho_avg_D_seg_vars[id];
                    dst->D_mixture_mass_rate_D_seg_vars[id] = D_mixture_mass_rate_D_seg_vars[id];
                    dst->D_average_volumetric_mixture_velocity_D_seg_vars[id] = D_average_volumetric_mixture_velocity_D_seg_vars[id];
                  }
                for (unsigned int k = 0; k < (1U + mp_in.nc + 1U) * mp_in.np; ++k)
                  dst->D_phase_holdup_D_seg_vars[k] = D_phase_holdup_D_seg_vars[k];
                for (unsigned int k = 0; k < (1U + mp_in.nc + 1U) * mp_in.nc; ++k)
                  dst->D_q_c_D_seg_vars[k] = D_q_c_D_seg_vars[k];
              }
            };

          const df_fd_wsncs_backup_t wsncs_backup (wsncs, mp);

          // The FD probes below use the same persistent lagged DF snapshot as the
          // production residual/Jacobian path.  df_fd_wsncs_backup_t captures the seeded
          // scratch state above and restores it before each test_function() call.

          // prev outputs
          double prev_average_mixture_velocity = 0.;
          double prev_gas_liq_interfacial_tension = 0.;
          //double prev_liquid_hp = 0.;
          double prev_bubble_rise_velocity = 0.;
          double prev_liquid_density = 0.;
          double prev_diametr_dimless = 0.;
          double prev_Kut_number = 0.;
          double prev_flooding_velocity = 0.;
          double prev_ksi = 0.;
          double prev_eta = 0.;
          double prev_gas_phase_velocity = 0.;
          double prev_liquid_phase_velocity = 0.;
          double prev_K_g = 0.;
          double prev_avg_xi = 0.;
          double prev_xi = 0.;
          double prev_sigma_o = 0.;
          double prev_sigma_w = 0.;
          double prev_oil_water_interfacial_tension = 0.;
          double prev_bubble_rise_velocity_OW = 0.;
          double prev_C_OW_buf = 0.;
          double prev_V_d_OW_buf = 0.;
          double prev_oil_velocity_buf = 0.;
          double prev_water_velocity_buf = 0.;

          double prev_alpha_g = 0.;
          double prev_beta_o = 0.;
          double prev_alpha_o = 0.;
          double prev_alpha_w = 0.;
          double prev_R_g = 0.;
          double prev_R_o = 0.;
          unsigned int prev_active_flags = 0U;
          int prev_inner_it = 0;

          // next outputs
          double next_average_mixture_velocity = 0.;
          double next_gas_liq_interfacial_tension = 0.;
          //double next_liquid_hp = 0.;
          double next_bubble_rise_velocity = 0.;
          double next_liquid_density = 0.;
          double next_diametr_dimless = 0.;
          double next_Kut_number = 0.;
          double next_flooding_velocity = 0.;
          double next_ksi = 0.;
          double next_eta = 0.;
          double next_gas_phase_velocity = 0.;
          double next_liquid_phase_velocity = 0.;
          double next_K_g = 0.;
          double next_avg_xi = 0.;
          double next_xi = 0.;
          double next_sigma_o = 0.;
          double next_sigma_w = 0.;
          double next_oil_water_interfacial_tension = 0.;
          double next_bubble_rise_velocity_OW = 0.;
          double next_C_OW_buf = 0.;
          double next_V_d_OW_buf = 0.;
          double next_oil_velocity_buf = 0.;
          double next_water_velocity_buf = 0.;

          double next_alpha_g = 0.;
          double next_beta_o = 0.;
          double next_alpha_o = 0.;
          double next_alpha_w = 0.;
          double next_R_g = 0.;
          double next_R_o = 0.;
          unsigned int next_active_flags = 0U;
          int next_inner_it = 0;

          (void) prev_average_mixture_velocity;
          (void) next_inner_it;
          (void) prev_alpha_w;
          (void) next_avg_xi;
          (void) prev_alpha_o;
          (void) next_xi;
          (void) next_average_mixture_velocity;
          (void) prev_inner_it;
          (void) next_alpha_o;
          (void) prev_avg_xi;
          (void) next_alpha_w;
          (void) prev_xi;

          // component rates for numerical derivative
          std::vector<double> prev_component_rate_dbg (mp.nc, 0.0);
          std::vector<double> next_component_rate_dbg (mp.nc, 0.0);
          double prev_rho_avg_dbg = 0.0;
          double next_rho_avg_dbg = 0.0;

          // ------------------------ prev
          fully_implicit_element_status element_status_prev_dbg (*element_status);
          fully_implicit_element_status *element_status_prev_dbg_ptr = &element_status_prev_dbg;
          copy_segment_params_to_element_status (seg, element_status_prev_dbg_ptr);

          wsncs_backup.restore (wsncs, mp);
          double qtot_prev_dbg = q_tot_buf;
          apply_fd_perturbation (element_status_prev_dbg_ptr, qtot_prev_dbg, -fd_eps);
          wsncs->wsn_mixture_molar_rate = qtot_prev_dbg;

          test_function (rep,
                         seg,
                         wsncs,
                         element_status_prev_dbg_ptr,
                         mp,
                         new_status,
                         i_meshblock,
                         current_therm_comp_input_props,
                         itd,
                         prev_average_mixture_velocity,
                         prev_gas_liq_interfacial_tension,
                         //prev_liquid_hp,
                         prev_bubble_rise_velocity,
                         prev_liquid_density,
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
                         prev_C_OW_buf,
                         prev_V_d_OW_buf,
                         prev_oil_velocity_buf,
                         prev_water_velocity_buf,
                         prev_alpha_g,
                         prev_beta_o,
                         prev_alpha_o,
                         prev_alpha_w,
                         prev_R_g,
                         prev_R_o,
                         prev_active_flags,
                         prev_inner_it);

          double prev_C_0 = wsncs->wsn_C_0;
          double prev_drift_velocity = wsncs->wsn_drift_velocity;
          prev_rho_avg_dbg = wsncs->rho_avg_DF;
          for (unsigned int ic = 0; ic < mp.nc; ++ic)
            prev_component_rate_dbg[ic] = wsncs->wsn_component_rate[ic];

          // ------------------------ next
          fully_implicit_element_status element_status_next_dbg (*element_status);
          fully_implicit_element_status *element_status_next_dbg_ptr = &element_status_next_dbg;
          copy_segment_params_to_element_status (seg, element_status_next_dbg_ptr);

          wsncs_backup.restore (wsncs, mp);
          double qtot_next_dbg = q_tot_buf;
          apply_fd_perturbation (element_status_next_dbg_ptr, qtot_next_dbg, +fd_eps);
          wsncs->wsn_mixture_molar_rate = qtot_next_dbg;

          test_function (rep,
                         seg,
                         wsncs,
                         element_status_next_dbg_ptr,
                         mp,
                         new_status,
                         i_meshblock,
                         current_therm_comp_input_props,
                         itd,
                         next_average_mixture_velocity,
                         next_gas_liq_interfacial_tension,
                         //next_liquid_hp,
                         next_bubble_rise_velocity,
                         next_liquid_density,
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
                         next_C_OW_buf,
                         next_V_d_OW_buf,
                         next_oil_velocity_buf,
                         next_water_velocity_buf,
                         next_alpha_g,
                         next_beta_o,
                         next_alpha_o,
                         next_alpha_w,
                         next_R_g,
                         next_R_o,
                         next_active_flags,
                         next_inner_it);

          double next_C_0 = wsncs->wsn_C_0;
          double next_drift_velocity = wsncs->wsn_drift_velocity;
          next_rho_avg_dbg = wsncs->rho_avg_DF;
          for (unsigned int ic = 0; ic < mp.nc; ++ic)
            next_component_rate_dbg[ic] = wsncs->wsn_component_rate[ic];

          // ------------------------ restore
          wsncs_backup.restore (wsncs, mp);
          wsncs->wsn_mixture_molar_rate = q_tot_buf;

          // ------------------------ numerical derivatives
          double C_0_num_derivative = (next_C_0 - prev_C_0) / (2.0 * fd_eps);
          double Drift_velocity_num_derivative = (next_drift_velocity - prev_drift_velocity) / (2.0 * fd_eps);

          double alpha_g_num_derivative = (next_alpha_g - prev_alpha_g) / (2.0 * fd_eps);
          double alpha_o_num_derivative = (next_alpha_o - prev_alpha_o) / (2.0 * fd_eps);
          double beta_o_num_derivative  = (next_beta_o - prev_beta_o) / (2.0 * fd_eps); // derived quantity only
          //double alpha_w_num_derivative = (next_alpha_w - prev_alpha_w) / (2.0 * fd_eps);

          //double average_mixture_velocity_numerical = (next_average_mixture_velocity - prev_average_mixture_velocity) / (2.0 * fd_eps);
          double gas_liq_interfacial_tension_numerical = (next_gas_liq_interfacial_tension - prev_gas_liq_interfacial_tension) / (2.0 * fd_eps);
          double bubble_rise_velocity_numerical = (next_bubble_rise_velocity - prev_bubble_rise_velocity) / (2.0 * fd_eps);
          double liquid_density_numerical = (next_liquid_density - prev_liquid_density) / (2.0 * fd_eps);
          double diametr_dimless_numerical = (next_diametr_dimless - prev_diametr_dimless) / (2.0 * fd_eps);
          double Kut_number_numerical = (next_Kut_number - prev_Kut_number) / (2.0 * fd_eps);
          double flooding_velocity_numerical = (next_flooding_velocity - prev_flooding_velocity) / (2.0 * fd_eps);
          double ksi_numerical = (next_ksi - prev_ksi) / (2.0 * fd_eps);
          double eta_numerical = (next_eta - prev_eta) / (2.0 * fd_eps);
          double gas_phase_velocity_numerical = (next_gas_phase_velocity - prev_gas_phase_velocity) / (2.0 * fd_eps);
          double liquid_phase_velocity_numerical = (next_liquid_phase_velocity - prev_liquid_phase_velocity) / (2.0 * fd_eps);
          double K_g_numerical = (next_K_g - prev_K_g) / (2.0 * fd_eps);
          //double avg_xi_numerical = (next_avg_xi - prev_avg_xi) / (2.0 * fd_eps);
          //double xi_numerical = (next_xi - prev_xi) / (2.0 * fd_eps);
          double sigma_o_num = (next_sigma_o - prev_sigma_o) / (2.0 * fd_eps);
          double sigma_w_num = (next_sigma_w - prev_sigma_w) / (2.0 * fd_eps);
          double liquid_hp_num = 0.5 * ((1.0 - prev_alpha_g) + (1.0 - next_alpha_g));
          double rho_avg_numerical = (next_rho_avg_dbg - prev_rho_avg_dbg) / (2.0 * fd_eps);
          double oil_water_interfacial_tension_num = (next_oil_water_interfacial_tension - prev_oil_water_interfacial_tension) / (2.0 * fd_eps);
          double bubble_rise_velocity_OW_num = (next_bubble_rise_velocity_OW - prev_bubble_rise_velocity_OW) / (2.0 * fd_eps);
          double C_OW_num = (next_C_OW_buf - prev_C_OW_buf) / (2.0 * fd_eps);
          double V_d_OW_num = (next_V_d_OW_buf - prev_V_d_OW_buf) / (2.0 * fd_eps);
          double oil_velocity_num = (next_oil_velocity_buf - prev_oil_velocity_buf) / (2.0 * fd_eps);
          double water_velocity_num = (next_water_velocity_buf - prev_water_velocity_buf) / (2.0 * fd_eps);

                  copy_segment_params_to_element_status (seg, element_status);
                  // IMPORTANT:
                  // Do NOT overwrite the current segment thermodynamic state with the upwind/parent
                  // composition inside the local DF block. Upwinding should be applied only to the
                  // convective transport closure at the face, not to the segment-local flash state.
                  // Otherwise the local residual block becomes independent of the current segment
                  // component unknowns in from_parent_to_child flow, which structurally zeroes the
                  // composition columns and can make the outer Newton matrix rank-deficient.
                  const bool use_upwind_state_inside_df = false;
                  if (use_upwind_state_inside_df
                      && wsncs->wsn_flow_dir == segment_flow_direction_t::from_parent_to_child)
                    {
                      set_flow_direction_dependent_segment_params_to_element_status (seg, element_status, new_status);
                    }

                 if (element_status->component_N_tot > tnm::min_compare)
                   {
                     if (auto err = run_flash <true> (rep, i_meshblock, element_status, current_therm_comp_input_props, itd); err != segments_solver_err_t::none)
                       return;
                   }

                  // Persistent lagged DF state was loaded once above.
                  // Do not read previous Newton values from wsncs here: wsncs is a scratch
                  // assembly object and is overwritten/reinitialized between outer evaluations.

                  //if (!(seg.wsn->wsn_index == TOP_SEG_INDEX))

                    fill_wsncs_from_element_status (wsncs, element_status, mp);
                    wsncs->wsn_mixture_molar_rate = q_tot_buf;

                  const unsigned int nseg_vars = 1U + mp.nc + 1U;
                  // Composition columns are active for the local DF block because the local
                  // flash/holdup state is evaluated from the CURRENT segment composition in both
                  // flow directions. Upwinding is handled separately in the transport closure.
                  const bool current_seg_component_columns_inactive = false;
                  auto current_seg_comp_col_inactive =
                    [&] (unsigned int id) -> bool
                    {
                      return current_seg_component_columns_inactive
                             && (id > 0U)
                             && (id < 1U + mp.nc);
                    };

                  wsncs->wsn_mmw = 0.0;
                  wsncs->wsn_mixture_mass_rate = 0.0;

                  if (seg.wsn->wsn_index == TOP_SEG_INDEX)
                    {
                      /*wsncs->wsn_C_0 = 1.0;
                      wsncs->wsn_drift_velocity = 0.0;
                      wsncs->wsn_C_0_OW = 1.0;
                      wsncs->wsn_drift_velocity_OW = 0.0;

                      for (unsigned int id = 0; id < nseg_vars; ++id)
                        {
                          wsncs->D_C0_D_seg_vars[id] = 0.0;
                          wsncs->D_drift_velocity_D_seg_vars[id] = 0.0;
                          wsncs->D_C0_OW_D_seg_vars[id] = 0.0;
                          wsncs->D_drift_velocity_OW_D_seg_vars[id] = 0.0;
                          wsncs->D_mixture_mass_rate_D_seg_vars[id] = 0.0;
                        }

                        for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                          {
                            wsncs->wsn_component_rate[ic] =
                                seg.wsncs->wsn_mixture_molar_rate * component_z_for_flow[ic];

                            for (unsigned int id = 0; id < nseg_vars; ++id)
                              wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id] = 0.0;

                            wsncs->wsn_mixture_mass_rate +=
                                wsncs->wsn_component_rate[ic] * component_molar_weights[ic];

                            wsncs->D_mixture_mass_rate_D_seg_vars[nseg_vars - 1] = 0.;

                            wsncs->wsn_mmw += component_z_for_flow[ic] * component_molar_weights[ic];
                          }*/
                    }
                      else
                          {
                          double error = 100.;
                          int it = 0;
                          int max_it = 10;

                          const bool prev_df_holdup_input_valid = df_lagged_input_state_valid;
                          // Do NOT freeze absolute superficial velocities.  Lag only the
                          // DF holdup/phase split; jmix and jsp_in are recomputed from
                          // the current Newton q_tot to preserve the q_tot Jacobian column.
                          const bool use_frozen_df_superficial_inputs = false;

                          const double alpha_g_lagged_df_input = df_lagged_input_state.alpha_g;
                          const double alpha_o_lagged_df_input = df_lagged_input_state.alpha_o;
                          const double alpha_w_lagged_df_input = df_lagged_input_state.alpha_w;

                          const double average_volumetric_mixture_velocity_from_rate_raw =
                              tnav_div (wsncs->wsn_mixture_molar_rate,
                                        seg.wsn->pipe_props.area * element_status->avg_xi * internal_const::DAYS_TO_SEC ());

                          double average_volumetric_mixture_velocity_raw =
                              average_volumetric_mixture_velocity_from_rate_raw;

                          double df_flow_sign =
                              (average_volumetric_mixture_velocity_raw < 0.0) ? -1.0 : 1.0;
                          const double average_volumetric_mixture_velocity_physical = average_volumetric_mixture_velocity_raw;
                          const double average_volumetric_mixture_velocity_from_rate_physical =
                              average_volumetric_mixture_velocity_from_rate_raw;
                          double average_volumetric_mixture_velocity =
                              df_flow_sign * average_volumetric_mixture_velocity_raw; // local DF frame: j >= 0

                              for (unsigned int id = 0; id < 1U + mp.nc + 1U; id++)
                                {
                                  double water = element_status->component_N[0];
                                  double oil = element_status->component_N[1];
                                  double gas = element_status->component_N[2];
                                  (void) water;
                                  (void) oil;
                                  (void) gas;

                                  if (current_seg_comp_col_inactive (id))
                                    {
                                      wsncs->D_average_volumetric_mixture_velocity_D_seg_vars[id] = 0.0;
                                      continue;
                                    }

                                  double D_average_volumetric_mixture_velocity_raw = 0.0;
                                  if (id < 1U + mp.nc) //p^j, z^c_j
                                    D_average_volumetric_mixture_velocity_raw =
                                        -tnav_div (wsncs->wsn_mixture_molar_rate,
                                                   seg.wsn->pipe_props.area * internal_const::DAYS_TO_SEC ()
                                                   * element_status->avg_xi * element_status->avg_xi)
                                        * element_status->avg_D_xi[id];
                                  else
                                    D_average_volumetric_mixture_velocity_raw =
                                        tnav_div (1., seg.wsn->pipe_props.area * internal_const::DAYS_TO_SEC () * element_status->avg_xi);

                                  wsncs->D_average_volumetric_mixture_velocity_D_seg_vars[id] =
                                      df_flow_sign * D_average_volumetric_mixture_velocity_raw;
                                }

                          (void) error;
                          (void) it;
                          (void) max_it;

                          double mixture_superficial_velocity =
                              average_volumetric_mixture_velocity; /// local DF input j [m/s]

                          std::vector<double> D_mixture_superficial_velocity_D_seg_vars (nseg_vars, 0.);
                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              D_mixture_superficial_velocity_D_seg_vars[id]
                                  = wsncs->D_average_volumetric_mixture_velocity_D_seg_vars[id]; /// internal_const::DAYS_TO_SEC ();
                            }


                          // No hard clipping of holdups inside the local DF iterator.
                          // Only denominators are protected to avoid NaN/Inf.
                          const double df_den_eps = tnm::min_compare;
                          auto df_safe_nonzero =
                            [&] (double v) -> double
                            {
                              if (!std::isfinite (v))
                                return (v < 0.0) ? -df_den_eps : df_den_eps;
                              if (fabs (v) < df_den_eps)
                                return (v < 0.0) ? -df_den_eps : df_den_eps;
                              return v;
                            };

                          double drift_incl_mult = 1.0;
                          {
                            const double seg_length_abs = fabs (seg.wsn->pipe_props.length);
                            const double seg_depth_abs = fabs (seg.wsn->pipe_props.depth_change);
                            if (seg_length_abs > tnm::min_compare)
                              {
                                double cos_theta = seg_depth_abs / seg_length_abs;
                                if (cos_theta < 0.0)
                                  cos_theta = 0.0;
                                else if (cos_theta > 1.0)
                                  cos_theta = 1.0;

                                if (cos_theta >= 1.0 - tnm::min_compare)
                                  drift_incl_mult = 1.0;
                                else
                                  {
                                    const double sin_theta = sqrt (std::max (0.0, 1.0 - cos_theta * cos_theta));
                                    drift_incl_mult = sqrt (cos_theta) * tnav_pow (1.0 + sin_theta, 2);
                                  }
                              }
                          }

                          const double alpha_g_flash_init = element_status->phase_S[PHASE_GAS];
                          const double alpha_o_flash_init = element_status->phase_S[PHASE_OIL];
                          const double alpha_w_flash_init = element_status->phase_S[PHASE_WATER];

                          bool phase_inputs_from_lagged_map = prev_df_holdup_input_valid;
                          bool lagged_phase_input_regularized = false;
                          double lagged_phase_input_lambda = phase_inputs_from_lagged_map ? 1.0 : 0.0;
                          double lagged_phase_input_det = 0.0;

                          double alpha_g_seed_raw = phase_inputs_from_lagged_map
                                                      ? alpha_g_lagged_df_input
                                                      : alpha_g_flash_init;
                          double alpha_o_seed_raw = phase_inputs_from_lagged_map
                                                      ? alpha_o_lagged_df_input
                                                      : alpha_o_flash_init;
                          double alpha_w_seed_raw = phase_inputs_from_lagged_map
                                                      ? alpha_w_lagged_df_input
                                                      : alpha_w_flash_init;

                          {
                            const double sum_input = alpha_g_seed_raw + alpha_o_seed_raw + alpha_w_seed_raw;
                            lagged_phase_input_det = sum_input;
                            if (!std::isfinite (sum_input) || sum_input <= tnm::min_compare)
                              {
                                phase_inputs_from_lagged_map = false;
                                lagged_phase_input_lambda = 0.0;
                                alpha_g_seed_raw = alpha_g_flash_init;
                                alpha_o_seed_raw = alpha_o_flash_init;
                                alpha_w_seed_raw = alpha_w_flash_init;
                              }
                            else if (fabs (sum_input - 1.0) > 1.0e-10)
                              {
                                alpha_g_seed_raw /= sum_input;
                                alpha_o_seed_raw /= sum_input;
                                alpha_w_seed_raw /= sum_input;
                                lagged_phase_input_regularized = true;
                              }
                          }

                          double gas_superficial_velocity_input_raw =
                              alpha_g_seed_raw * average_volumetric_mixture_velocity_raw;
                          double oil_superficial_velocity_input_raw =
                              alpha_o_seed_raw * average_volumetric_mixture_velocity_raw;
                          double water_superficial_velocity_input_raw =
                              alpha_w_seed_raw * average_volumetric_mixture_velocity_raw;

                          // HOLDUP_SPLIT_CURRENT_JMIX mode:
                          // jsp_in = alpha_input * current_jmix.  The previously stored
                          // absolute jsp values are logged only and are not used as inputs.

                          double gas_superficial_velocity_input = df_flow_sign * gas_superficial_velocity_input_raw;
                          double oil_superficial_velocity_input = df_flow_sign * oil_superficial_velocity_input_raw;
                          double water_superficial_velocity_input = df_flow_sign * water_superficial_velocity_input_raw;

                          std::vector<double> D_gas_superficial_velocity_input_D_seg_vars (nseg_vars, 0.0);
                          std::vector<double> D_oil_superficial_velocity_input_D_seg_vars (nseg_vars, 0.0);
                          std::vector<double> D_water_superficial_velocity_input_D_seg_vars (nseg_vars, 0.0);

                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              const double D_jmix = D_mixture_superficial_velocity_D_seg_vars[id];

                              // If the phase split comes from previous Newton DF state,
                              // alpha_input is frozen but current jmix is not.  Therefore
                              // the lagged branch uses D_jsp = alpha_prevDF * D_jmix.
                              // If no previous DF state exists, use the flash derivatives.
                              double D_alpha_g_input = 0.0;
                              double D_alpha_o_input = 0.0;
                              double D_alpha_w_input = 0.0;
                              if (!phase_inputs_from_lagged_map && id < 1U + mp.nc)
                                {
                                  D_alpha_g_input = element_status->phase_D_S[id * mp.np + PHASE_GAS];
                                  D_alpha_o_input = element_status->phase_D_S[id * mp.np + PHASE_OIL];
                                  D_alpha_w_input = element_status->phase_D_S[id * mp.np + PHASE_WATER];
                                }

                              D_gas_superficial_velocity_input_D_seg_vars[id] =
                                  alpha_g_seed_raw * D_jmix + mixture_superficial_velocity * D_alpha_g_input;
                              D_oil_superficial_velocity_input_D_seg_vars[id] =
                                  alpha_o_seed_raw * D_jmix + mixture_superficial_velocity * D_alpha_o_input;
                              D_water_superficial_velocity_input_D_seg_vars[id] =
                                  alpha_w_seed_raw * D_jmix + mixture_superficial_velocity * D_alpha_w_input;
                            }

                          const double liquid_superficial_velocity_input =
                              oil_superficial_velocity_input + water_superficial_velocity_input;

                          const double alpha_l_seed_local = alpha_o_seed_raw + alpha_w_seed_raw;
                          const double beta_input =
                              (alpha_l_seed_local > df_den_eps)
                                ? std::max (0.0, std::min (1.0, alpha_o_seed_raw / alpha_l_seed_local))
                                : 0.5;

                          const double simplex_eps = 1.0e-12;
                          auto clamp01 = [&] (double v) -> double
                            {
                              if (!std::isfinite (v)) return 0.0;
                              if (v < 0.0) return 0.0;
                              if (v > 1.0) return 1.0;
                              return v;
                            };

                          auto project_df_simplex = [&] (double &alpha_g_inout, double &alpha_o_inout)
                            {
                              if (!std::isfinite (alpha_g_inout)) alpha_g_inout = alpha_g_flash_init;
                              if (!std::isfinite (alpha_o_inout)) alpha_o_inout = alpha_o_flash_init;

                              if (alpha_g_inout < simplex_eps)
                                alpha_g_inout = simplex_eps;
                              if (alpha_g_inout > 1.0 - simplex_eps)
                                alpha_g_inout = 1.0 - simplex_eps;

                              double alpha_l_local = 1.0 - alpha_g_inout;
                              if (alpha_l_local < simplex_eps)
                                {
                                  alpha_g_inout = 1.0 - simplex_eps;
                                  alpha_l_local = simplex_eps;
                                }

                              if (!std::isfinite (alpha_o_inout))
                                alpha_o_inout = 0.0;

                              if (alpha_o_inout < 0.0)
                                alpha_o_inout = 0.0;
                              if (alpha_o_inout > alpha_l_local)
                                alpha_o_inout = alpha_l_local;
                            };

                          double alpha_g_newton = alpha_g_seed_raw;
                          double alpha_o_newton = alpha_o_seed_raw;
                          project_df_simplex (alpha_g_newton, alpha_o_newton);
                          if (1.0 - alpha_g_newton < 10.0 * simplex_eps)
                            {
                              alpha_g_newton = 1.0 - 10.0 * simplex_eps;
                              alpha_o_newton = beta_input * (1.0 - alpha_g_newton);
                            }
                          project_df_simplex (alpha_g_newton, alpha_o_newton);

                          // -----------------------------------------------------------------------------
                          // A.3 Value-only evaluator at fixed (alpha_g, alpha_o).
                          //     This follows the same Eclipse/Shi closures as the old code.
                          // -----------------------------------------------------------------------------
                          struct df_local_value_state_t
                            {
                              phase_holdups_DF holdups = phase_holdups_DF (0.0, 0.0, 0.0, 0.0);
                              phase_vel_DF     vels    = phase_vel_DF (0.0, 0.0, 0.0, 0.0);

                              double sigma_o = 0.0; // dynes/cm
                              double sigma_w = 0.0; // dynes/cm
                              double gas_liq_interfacial_tension = 0.0;
                              double liq_density = 0.0;
                              double gas_density = 0.0;
                              double bubble_rise_velocity = 0.0;
                              double diametr_dimless = 0.0;
                              double Kut_number = 0.0;
                              double flooding_velocity = 0.0;
                              double ksi = 0.0;
                              double eta = 0.0;
                              double C0 = 1.0;
                              double K_g = 0.0;
                              double drift_velocity = 0.0;

                              double oil_water_interfacial_tension = 0.0;
                              double bubble_rise_velocity_OW = 0.0;
                              double C0_OW = 1.0;
                              double drift_velocity_OW = 0.0;
                            };

                          auto evaluate_df_value_alphao = [&] (double alpha_g_trial,
                                                               double alpha_o_trial,
                                                               df_local_value_state_t &S)
                            {
                              project_df_simplex (alpha_g_trial, alpha_o_trial);

                              const double alpha_l_trial = 1.0 - alpha_g_trial;
                              const double alpha_w_trial = std::max (0.0, alpha_l_trial - alpha_o_trial);
                              const double beta_o_trial = (alpha_l_trial > df_den_eps) ? clamp01 (alpha_o_trial / alpha_l_trial) : beta_input;

                              S.holdups.gas = alpha_g_trial;
                              S.holdups.liquid = alpha_l_trial;
                              S.holdups.oil = alpha_o_trial;
                              S.holdups.water = alpha_w_trial;

                              double D_sigma_o_dummy = 0.0;
                              double D_sigma_w_dummy = 0.0;

                              S.sigma_o = pipe_gas_oil_interfacial_tension_and_deriv (
                                  45.5,
                                  element_status->p * converter_metric_to_field.pressure_mult (),
                                  160.0,
                                  D_sigma_o_dummy);

                              S.sigma_w = pipe_gas_wat_interfacial_tension_and_deriv (
                                  element_status->p * converter_metric_to_field.pressure_mult (),
                                  160.0,
                                  D_sigma_w_dummy);

                              const double sigma_o_SI = surf_mult * S.sigma_o;
                              const double sigma_w_SI = surf_mult * S.sigma_w;

                              S.gas_liq_interfacial_tension =
                                  beta_o_trial * sigma_o_SI + (1.0 - beta_o_trial) * sigma_w_SI;

                              S.liq_density =
                                  beta_o_trial * element_status->phase_rho[PHASE_OIL]
                                  + (1.0 - beta_o_trial) * element_status->phase_rho[PHASE_WATER];

                              S.gas_density = element_status->phase_rho[PHASE_GAS];

                              S.bubble_rise_velocity = 0.0;
                              if (S.gas_liq_interfacial_tension > df_den_eps && S.liq_density > df_den_eps)
                                {
                                  S.bubble_rise_velocity =
                                      tnav_pow (
                                          tnav_div (S.gas_liq_interfacial_tension * internal_const::grav_metric ()
                                                    * fabs (S.liq_density - S.gas_density),
                                                    S.liq_density * S.liq_density),
                                          0.25);
                                }

                              S.diametr_dimless = 0.0;
                              if (S.liq_density - S.gas_density > 0.0 && S.gas_liq_interfacial_tension > df_den_eps)
                                {
                                  S.diametr_dimless =
                                      sqrt (tnav_div (internal_const::grav_metric () * (S.liq_density - S.gas_density),
                                                      S.gas_liq_interfacial_tension))
                                      * seg.wsn->pipe_props.diameter;
                                }

                              double lin_interp_dummy = 0.0;
                              S.Kut_number = compute_critical_Kutateladze_number_by_diametr (S.diametr_dimless, lin_interp_dummy);

                              S.flooding_velocity = S.Kut_number * sqrt (tnav_div (S.liq_density, std::max (S.gas_density, df_den_eps))) * S.bubble_rise_velocity;

                              const double A = 1.2;
                              const double B = 0.3;
                              const double F_v = 1.0;
                              const double V_sgf = df_safe_nonzero (S.flooding_velocity);

                              S.ksi = std::max (alpha_g_trial,
                                                tnav_div (F_v * alpha_g_trial * fabs (mixture_superficial_velocity), V_sgf));

                              S.eta = (S.ksi - B) / (1.0 - B);
                              if (!std::isfinite (S.eta))
                                S.eta = (S.ksi > B) ? 1.0 : 0.0;
                              else if (S.eta < 0.0)
                                S.eta = 0.0;
                              else if (S.eta > 1.0)
                                S.eta = 1.0;

                              S.C0 = A / (1.0 + (A - 1.0) * S.eta * S.eta);

                              const double K_g_low  = 1.53 / S.C0;
                              const double K_g_high = S.Kut_number;
                              if (alpha_g_trial < 0.2)
                                S.K_g = K_g_low;
                              else if (alpha_g_trial > 0.4)
                                S.K_g = K_g_high;
                              else
                                S.K_g = interpolate_y_against_x (alpha_g_trial, 0.2, 0.4, K_g_low, K_g_high);

                              {
                                const double sqrt_gas_over_liq = sqrt (tnav_div (std::max (S.gas_density, df_den_eps), std::max (S.liq_density, df_den_eps)));
                                const double numerator_gd = (1.0 - alpha_g_trial * S.C0) * S.C0 * S.K_g * S.bubble_rise_velocity;
                                const double denominator_gd = 1.0 + alpha_g_trial * S.C0 * (sqrt_gas_over_liq - 1.0);
                                S.drift_velocity = drift_incl_mult * tnav_div (numerator_gd, df_safe_nonzero (denominator_gd));
                              }

                              S.vels.gas = S.C0 * mixture_superficial_velocity + S.drift_velocity;

                              if (1.0 - alpha_g_trial > df_den_eps)
                                {
                                  S.vels.liquid =
                                      tnav_div (1.0 - alpha_g_trial * S.C0, 1.0 - alpha_g_trial) * mixture_superficial_velocity
                                      - tnav_div (alpha_g_trial, 1.0 - alpha_g_trial) * S.drift_velocity;
                                }
                              else
                                {
                                  S.vels.liquid = mixture_superficial_velocity;
                                }

                              const double B1_OW = 0.4;
                              const double B2_OW = 0.7;
                              const double A_OW  = 1.2;

                              if (beta_o_trial < B1_OW)
                                S.C0_OW = A_OW;
                              else if (beta_o_trial > B2_OW)
                                S.C0_OW = 1.0;
                              else
                                S.C0_OW = interpolate_y_against_x (beta_o_trial, B1_OW, B2_OW, A_OW, 1.0);

                              double D_go_dummy = 0.0;
                              double D_gw_dummy = 0.0;

                              const double gas_oil_interfacial_tension =
                                  surf_mult * pipe_gas_oil_interfacial_tension_and_deriv (
                                      45.5,
                                      element_status->p * converter_metric_to_field.pressure_mult (),
                                      160.0,
                                      D_go_dummy);

                              const double gas_wat_interfacial_tension =
                                  surf_mult * pipe_gas_wat_interfacial_tension_and_deriv (
                                      element_status->p * converter_metric_to_field.pressure_mult (),
                                      160.0,
                                      D_gw_dummy);

                              S.oil_water_interfacial_tension =
                                  fabs (gas_oil_interfacial_tension * beta_o_trial - gas_wat_interfacial_tension * (1.0 - beta_o_trial));

                              S.bubble_rise_velocity_OW = 0.0;
                              if (S.oil_water_interfacial_tension > df_den_eps
                                  && element_status->phase_rho[PHASE_WATER] > df_den_eps)
                                {
                                  S.bubble_rise_velocity_OW =
                                      tnav_pow (
                                          S.oil_water_interfacial_tension * internal_const::grav_metric ()
                                          * (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL])
                                          / (element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER]),
                                          0.25);
                                }

                              S.drift_velocity_OW = drift_incl_mult * 1.53 * S.bubble_rise_velocity_OW * tnav_pow (1.0 - beta_o_trial, 2);

                              S.vels.oil = S.C0_OW * S.vels.liquid + S.drift_velocity_OW;

                              if (1.0 - beta_o_trial > df_den_eps)
                                {
                                  S.vels.water =
                                      tnav_div (1.0 - beta_o_trial * S.C0_OW, 1.0 - beta_o_trial) * S.vels.liquid
                                      - tnav_div (beta_o_trial, 1.0 - beta_o_trial) * S.drift_velocity_OW;
                                }
                              else
                                {
                                  S.vels.water = S.vels.liquid;
                                }
                            };

                          // -----------------------------------------------------------------------------

                          // -----------------------------------------------------------------------------
                          // A.4 Two-stage staged DF solve: gas/liquid first, then oil/water.
                          //     This mirrors the Shi / ECLIPSE description more closely than the
                          //     previous coupled (alpha_g, alpha_o) Newton and isolates the gas/liquid
                          //     branch for debugging.
                          // -----------------------------------------------------------------------------
                          phase_holdups_DF prev_holdups (0.0, 0.0, 0.0, 0.0);
                          phase_vel_DF prev_vels (0.0, 0.0, 0.0, 0.0);
                          phase_holdups_DF new_holdups (0.0, 0.0, 0.0, 0.0);
                          phase_vel_DF new_vels (0.0, 0.0, 0.0, 0.0);

                          const double beta_gl_fixed = beta_input;
                          const double beta_min = simplex_eps;
                          const double beta_max = 1.0 - simplex_eps;

                          auto evaluate_gl_value = [&] (double alpha_g_trial,
                                                        df_local_value_state_t &S)
                            {
                              const double alpha_g_clamped = std::max (simplex_eps,
                                                                       std::min (1.0 - simplex_eps, alpha_g_trial));
                              const double alpha_o_trial = beta_gl_fixed * (1.0 - alpha_g_clamped);
                              evaluate_df_value_alphao (alpha_g_clamped, alpha_o_trial, S);
                            };

                          auto compute_rg_only = [&] (double alpha_g_trial,
                                                      df_local_value_state_t *S_out = nullptr) -> double
                            {
                              df_local_value_state_t S_tmp;
                              evaluate_gl_value (alpha_g_trial, S_tmp);
                              if (S_out)
                                *S_out = S_tmp;
                              return S_tmp.holdups.gas * S_tmp.vels.gas - gas_superficial_velocity_input;
                            };

                          auto evaluate_ow_value = [&] (double alpha_g_fixed,
                                                        double beta_trial,
                                                        df_local_value_state_t &S)
                            {
                              const double alpha_g_clamped = std::max (simplex_eps,
                                                                       std::min (1.0 - simplex_eps, alpha_g_fixed));
                              const double alpha_l_fixed = std::max (0.0, 1.0 - alpha_g_clamped);
                              const double beta_clamped = clamp01 (beta_trial);
                              const double alpha_o_trial = alpha_l_fixed * beta_clamped;
                              evaluate_df_value_alphao (alpha_g_clamped, alpha_o_trial, S);
                            };

                          auto compute_ro_only = [&] (double alpha_g_fixed,
                                                      double beta_trial,
                                                      df_local_value_state_t *S_out = nullptr) -> double
                            {
                              df_local_value_state_t S_tmp;
                              evaluate_ow_value (alpha_g_fixed, beta_trial, S_tmp);
                              if (S_out)
                                *S_out = S_tmp;
                              return S_tmp.holdups.oil * S_tmp.vels.oil - oil_superficial_velocity_input;
                            };

                          auto find_first_root_bracket = [&] (const std::function<double(double)> &f,
                                                              double lo,
                                                              double hi,
                                                              int nscan,
                                                              double &x_left,
                                                              double &x_right,
                                                              double &x_best,
                                                              double &f_best) -> bool
                            {
                              x_left = lo;
                              x_right = hi;
                              x_best = lo;
                              f_best = f (lo);
                              double best_abs = fabs (f_best);

                              double x_prev = lo;
                              double f_prev = f_best;
                              for (int is = 1; is <= nscan; ++is)
                                {
                                  const double t = static_cast<double> (is) / static_cast<double> (nscan);
                                  const double x_now = lo + (hi - lo) * t;
                                  const double f_now = f (x_now);
                                  if (fabs (f_now) < best_abs)
                                    {
                                      best_abs = fabs (f_now);
                                      x_best = x_now;
                                      f_best = f_now;
                                    }
                                  if (std::isfinite (f_prev) && std::isfinite (f_now) && f_prev * f_now <= 0.0)
                                    {
                                      x_left = x_prev;
                                      x_right = x_now;
                                      return true;
                                    }
                                  x_prev = x_now;
                                  f_prev = f_now;
                                }
                              return false;
                            };
                          (void) find_first_root_bracket;

                          struct df_root_bracket_t
                            {
                              double xl = 0.0;
                              double xr = 0.0;
                              double fl = 0.0;
                              double fr = 0.0;
                            };

                          auto collect_root_brackets = [&] (const std::function<double(double)> &f,
                                                            double lo,
                                                            double hi,
                                                            int nscan,
                                                            std::vector<df_root_bracket_t> &brackets,
                                                            double &x_best,
                                                            double &f_best)
                            {
                              brackets.clear ();
                              x_best = lo;
                              f_best = f (lo);
                              double best_abs = fabs (f_best);

                              double x_prev = lo;
                              double f_prev = f_best;
                              for (int is = 1; is <= nscan; ++is)
                                {
                                  const double t = static_cast<double> (is) / static_cast<double> (nscan);
                                  const double x_now = lo + (hi - lo) * t;
                                  const double f_now = f (x_now);
                                  if (fabs (f_now) < best_abs)
                                    {
                                      best_abs = fabs (f_now);
                                      x_best = x_now;
                                      f_best = f_now;
                                    }
                                  if (std::isfinite (f_prev) && std::isfinite (f_now) && f_prev * f_now <= 0.0)
                                    {
                                      df_root_bracket_t B;
                                      B.xl = x_prev;
                                      B.xr = x_now;
                                      B.fl = f_prev;
                                      B.fr = f_now;
                                      brackets.push_back (B);
                                    }
                                  x_prev = x_now;
                                  f_prev = f_now;
                                }
                            };

                          auto select_nearest_root_bracket = [&] (const std::vector<df_root_bracket_t> &brackets,
                                                                  double x_target) -> int
                            {
                              if (brackets.empty ())
                                return -1;
                              int best_idx = 0;
                              double best_dist = 1.0e300;
                              for (size_t ib = 0; ib < brackets.size (); ++ib)
                                {
                                  const double xl = brackets[ib].xl;
                                  const double xr = brackets[ib].xr;
                                  double dist = 0.0;
                                  if (x_target < xl)
                                    dist = xl - x_target;
                                  else if (x_target > xr)
                                    dist = x_target - xr;
                                  else
                                    dist = 0.0;
                                  if (dist < best_dist)
                                    {
                                      best_dist = dist;
                                      best_idx = static_cast<int> (ib);
                                    }
                                }
                              return best_idx;
                            };

                          auto solve_scalar_root = [&] (const std::function<double(double)> &f,
                                                        const std::function<void(int, double, double, double, double, double, double)> &iter_logger,
                                                        double x_left,
                                                        double x_right,
                                                        double x_seed,
                                                        double tol,
                                                        int max_it_local,
                                                        double &x_out,
                                                        double &f_out,
                                                        int &it_out) -> bool
                            {
                              double xl = x_left;
                              double xr = x_right;
                              double fl = f (xl);
                              double fr = f (xr);
                              if (!(std::isfinite (fl) && std::isfinite (fr)) || fl * fr > 0.0)
                                {
                                  x_out = x_seed;
                                  f_out = f (x_out);
                                  it_out = 0;
                                  if (iter_logger)
                                    iter_logger (it_out, x_out, f_out, xl, xr, fl, fr);
                                  return false;
                                }

                              double x = std::max (xl, std::min (xr, x_seed));
                              it_out = 0;
                              for (int it_loc = 0; it_loc < max_it_local; ++it_loc)
                                {
                                  ++it_out;
                                  const double fx = f (x);
                                  f_out = fx;
                                  if (iter_logger)
                                    iter_logger (it_out, x, fx, xl, xr, fl, fr);
                                  if (fabs (fx) <= tol)
                                    {
                                      x_out = x;
                                      return true;
                                    }

                                  const double eps_x = std::max (1.0e-8, 1.0e-6 * std::max (1.0, fabs (x)));
                                  const double xp = std::min (xr, x + eps_x);
                                  const double xm = std::max (xl, x - eps_x);
                                  const double fp = f (xp);
                                  const double fm = f (xm);
                                  const double denom = std::max (xp - xm, 1.0e-14);
                                  const double dfdx = (fp - fm) / denom;

                                  double x_trial = x;
                                  if (std::isfinite (dfdx) && fabs (dfdx) > 1.0e-14)
                                    x_trial = x - fx / dfdx;
                                  if (!(x_trial > xl && x_trial < xr) || !std::isfinite (x_trial))
                                    x_trial = 0.5 * (xl + xr);

                                  const double f_trial = f (x_trial);
                                  if (fl * f_trial <= 0.0)
                                    {
                                      xr = x_trial;
                                      fr = f_trial;
                                    }
                                  else
                                    {
                                      xl = x_trial;
                                      fl = f_trial;
                                    }
                                  x = x_trial;
                                }

                              x_out = x;
                              f_out = f (x);
                              if (iter_logger)
                                iter_logger (it_out, x_out, f_out, xl, xr, fl, fr);
                              return (fabs (f_out) <= tol);
                            };

                          auto compute_full_residuals = [&] (const df_local_value_state_t &S,
                                                             double &Rg_out,
                                                             double &Ro_out,
                                                             double &Rw_out,
                                                             double &res_norm_out)
                            {
                              Rg_out = S.holdups.gas * S.vels.gas - gas_superficial_velocity_input;
                              Ro_out = S.holdups.oil * S.vels.oil - oil_superficial_velocity_input;
                              Rw_out = S.holdups.water * S.vels.water - water_superficial_velocity_input;

                              const double Rg_rel = fabs (Rg_out) / std::max (1.0, fabs (gas_superficial_velocity_input));
                              const double Ro_rel = fabs (Ro_out) / std::max (1.0, fabs (oil_superficial_velocity_input));
                              const double Rw_rel = fabs (Rw_out) / std::max (1.0, fabs (water_superficial_velocity_input));
                              res_norm_out = std::sqrt (Rg_rel * Rg_rel + Ro_rel * Ro_rel + Rw_rel * Rw_rel);
                            };

                          auto log_df_iter_state = [&] (const char *stage_name,
                                                        int it_loc,
                                                        double x_value,
                                                        double fx_value,
                                                        double xl_value,
                                                        double xr_value,
                                                        double fl_value,
                                                        double fr_value,
                                                        const df_local_value_state_t &S_dbg)
                            {
                              double Rg_dbg = 0.0;
                              double Ro_dbg = 0.0;
                              double Rw_dbg = 0.0;
                              double res_dbg = 0.0;
                              compute_full_residuals (S_dbg, Rg_dbg, Ro_dbg, Rw_dbg, res_dbg);
                              const double beta_dbg = (S_dbg.holdups.liquid > df_den_eps)
                                                        ? (S_dbg.holdups.oil / S_dbg.holdups.liquid)
                                                        : 0.0;
                              PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                   "MERGEN_DF_ITER: Well {}: Seg_idx = {}, stage = {}, it_loc = {}, x = {}, fx = {}, xl = {}, xr = {}, fl = {}, fr = {}, "
                                   "alpha_g = {}, alpha_l = {}, alpha_o = {}, alpha_w = {}, beta_o = {}, vg = {}, vl = {}, vo = {}, vw = {}, C0 = {}, Vd = {}, C0_OW = {}, Vd_OW = {}, "
                                   "Rg = {}, Ro = {}, Rw = {}, res_norm = {}\n",
                                   wcb_wis->get_well_name (), seg.wsn->wsn_index, stage_name, it_loc,
                                   x_value, fx_value, xl_value, xr_value, fl_value, fr_value,
                                   S_dbg.holdups.gas, S_dbg.holdups.liquid, S_dbg.holdups.oil, S_dbg.holdups.water, beta_dbg,
                                   S_dbg.vels.gas, S_dbg.vels.liquid, S_dbg.vels.oil, S_dbg.vels.water,
                                   S_dbg.C0, S_dbg.drift_velocity, S_dbg.C0_OW, S_dbg.drift_velocity_OW,
                                   Rg_dbg, Ro_dbg, Rw_dbg, res_dbg);
                            };

                          auto apply_df_flow_sign_to_state = [&] (df_local_value_state_t &S)
                            {
                              S.vels.gas *= df_flow_sign;
                              S.vels.liquid *= df_flow_sign;
                              S.vels.oil *= df_flow_sign;
                              S.vels.water *= df_flow_sign;
                              S.drift_velocity *= df_flow_sign;
                              S.drift_velocity_OW *= df_flow_sign;
                            };

                          double alpha_g_gl = std::max (simplex_eps, std::min (1.0 - simplex_eps, alpha_g_seed_raw));
                          double beta_ow = clamp01 (beta_input);
                          double gl_best = alpha_g_gl;
                          double gl_best_res = 0.0;
                          double gl_left = simplex_eps;
                          double gl_right = 1.0 - simplex_eps;
                          int gl_it = 0;
                          int ow_it = 0;

                          std::vector<df_root_bracket_t> gl_brackets;
                          collect_root_brackets (
                              [&] (double ag) { return compute_rg_only (ag, nullptr); },
                              simplex_eps, 1.0 - simplex_eps, 64,
                              gl_brackets, gl_best, gl_best_res);
                          const bool gl_root_exists = !gl_brackets.empty ();

                          double Rg_stage = 0.0;
                          if (gl_root_exists)
                            {
                              const int ibest_gl = select_nearest_root_bracket (gl_brackets, alpha_g_gl);
                              gl_left = gl_brackets[ibest_gl].xl;
                              gl_right = gl_brackets[ibest_gl].xr;
                              PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                   "MERGEN_DF_BRACKET_SELECT: Well {}: Seg_idx = {}, stage = GL, idx = {}, xl = {}, xr = {}, target = {}\n",
                                   wcb_wis->get_well_name (), seg.wsn->wsn_index, ibest_gl, gl_left, gl_right, alpha_g_gl);

                              const std::function<void(int, double, double, double, double, double, double)> gl_iter_logger =
                                  [&] (int it_loc, double x_value, double fx_value, double xl_value, double xr_value, double fl_value, double fr_value)
                                    {
                                      df_local_value_state_t S_dbg;
                                      compute_rg_only (x_value, &S_dbg);
                                      log_df_iter_state ("GL", it_loc, x_value, fx_value, xl_value, xr_value, fl_value, fr_value, S_dbg);
                                    };
                              const double alpha_g_seed_local = std::max (gl_left, std::min (gl_right, alpha_g_gl));
                              solve_scalar_root ([&] (double ag) { return compute_rg_only (ag, nullptr); },
                                                 gl_iter_logger,
                                                 gl_left, gl_right, alpha_g_seed_local, 1.0e-12, 32,
                                                 alpha_g_gl, Rg_stage, gl_it);
                            }
                          else
                            {
                              alpha_g_gl = gl_best;
                              df_local_value_state_t S_dbg;
                              Rg_stage = compute_rg_only (alpha_g_gl, &S_dbg);
                              log_df_iter_state ("GL_NO_BRACKET", 0, alpha_g_gl, Rg_stage, gl_left, gl_right, gl_best_res, gl_best_res, S_dbg);
                            }

                          const double alpha_l_gl = std::max (0.0, 1.0 - alpha_g_gl);
                          if (alpha_l_gl > 1.0e-10 && fabs (liquid_superficial_velocity_input) > 1.0e-14)
                            {
                              const double beta_target = clamp01 (beta_input);
                              const double Ro_target = compute_ro_only (alpha_g_gl, beta_target, nullptr);
                              PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                   "MERGEN_DF_TARGET: Well {}: Seg_idx = {}, stage = OW, x_target = {}, f_target = {}\n",
                                   wcb_wis->get_well_name (), seg.wsn->wsn_index, beta_target, Ro_target);

                              double beta_best = beta_target;
                              double beta_best_res = 0.0;
                              std::vector<df_root_bracket_t> ow_brackets;
                              collect_root_brackets (
                                  [&] (double b) { return compute_ro_only (alpha_g_gl, b, nullptr); },
                                  beta_min, beta_max, 64,
                                  ow_brackets, beta_best, beta_best_res);

                              PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                   "MERGEN_DF_BRACKETS: Well {}: Seg_idx = {}, stage = OW, target = {}, n = {}\n",
                                   wcb_wis->get_well_name (), seg.wsn->wsn_index, beta_target, ow_brackets.size ());
                              for (size_t ib = 0; ib < ow_brackets.size (); ++ib)
                                {
                                  const double in_interval = (beta_target >= ow_brackets[ib].xl && beta_target <= ow_brackets[ib].xr) ? 1.0 : 0.0;
                                  const double dist = (beta_target < ow_brackets[ib].xl)
                                                        ? (ow_brackets[ib].xl - beta_target)
                                                        : ((beta_target > ow_brackets[ib].xr)
                                                           ? (beta_target - ow_brackets[ib].xr)
                                                           : 0.0);
                                  PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                       "MERGEN_DF_BRACKET: Well {}: Seg_idx = {}, stage = OW, idx = {}, xl = {}, xr = {}, fl = {}, fr = {}, in_interval = {}, dist = {}\n",
                                       wcb_wis->get_well_name (), seg.wsn->wsn_index, static_cast<int> (ib),
                                       ow_brackets[ib].xl, ow_brackets[ib].xr, ow_brackets[ib].fl, ow_brackets[ib].fr,
                                       in_interval, dist);
                                }

                              double Ro_stage = 0.0;
                              if (std::isfinite (Ro_target) && fabs (Ro_target) <= 1.0e-12)
                                {
                                  beta_ow = beta_target;
                                  df_local_value_state_t S_dbg;
                                  Ro_stage = compute_ro_only (alpha_g_gl, beta_ow, &S_dbg);
                                  log_df_iter_state ("OW_TARGET_ACCEPT", 0, beta_ow, Ro_stage, beta_target, beta_target, Ro_stage, Ro_stage, S_dbg);
                                }
                              else if (!ow_brackets.empty ())
                                {
                                  const int ibest = select_nearest_root_bracket (ow_brackets, beta_target);
                                  const double beta_left = ow_brackets[ibest].xl;
                                  const double beta_right = ow_brackets[ibest].xr;
                                  PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                       "MERGEN_DF_BRACKET_SELECT: Well {}: Seg_idx = {}, stage = OW, idx = {}, xl = {}, xr = {}, target = {}\n",
                                       wcb_wis->get_well_name (), seg.wsn->wsn_index, ibest, beta_left, beta_right, beta_target);
                                  const std::function<void(int, double, double, double, double, double, double)> ow_iter_logger =
                                      [&] (int it_loc, double x_value, double fx_value, double xl_value, double xr_value, double fl_value, double fr_value)
                                        {
                                          df_local_value_state_t S_dbg;
                                          evaluate_ow_value (alpha_g_gl, x_value, S_dbg);
                                          log_df_iter_state ("OW", it_loc, x_value, fx_value, xl_value, xr_value, fl_value, fr_value, S_dbg);
                                        };
                                  const double beta_seed_local = std::max (beta_left, std::min (beta_right, beta_target));
                                  solve_scalar_root ([&] (double b) { return compute_ro_only (alpha_g_gl, b, nullptr); },
                                                     ow_iter_logger,
                                                     beta_left, beta_right, beta_seed_local, 1.0e-12, 32,
                                                     beta_ow, Ro_stage, ow_it);
                                }
                              else
                                {
                                  beta_ow = beta_best;
                                  df_local_value_state_t S_dbg;
                                  Ro_stage = compute_ro_only (alpha_g_gl, beta_ow, &S_dbg);
                                  log_df_iter_state ("OW_NO_BRACKET", 0, beta_ow, Ro_stage, beta_min, beta_max, beta_best_res, beta_best_res, S_dbg);
                                }
                              (void) Ro_stage;
                            }
                          else
                            {
                              beta_ow = clamp01 (beta_input);
                            }

                          df_local_value_state_t state_now;
                          evaluate_ow_value (alpha_g_gl, beta_ow, state_now);

                          double Rg_final = 0.0;
                          double Ro_final = 0.0;
                          double Rw_final = 0.0;
                          double res_norm_final = 0.0;
                          compute_full_residuals (state_now, Rg_final, Ro_final, Rw_final, res_norm_final);

                          error = res_norm_final;
                          it = gl_it + ow_it;

                          apply_df_flow_sign_to_state (state_now);
                          prev_holdups.copy_operator (&state_now.holdups);
                          prev_vels.copy_operator (&state_now.vels);
                          new_holdups.copy_operator (&state_now.holdups);
                          new_vels.copy_operator (&state_now.vels);
// Export final value-path scalars for logs / FD debug / post-processing.
                          evaluate_df_value_alphao (prev_holdups.gas, prev_holdups.oil, state_now);
                          apply_df_flow_sign_to_state (state_now);
                          new_holdups.copy_operator (&state_now.holdups);
                          new_vels.copy_operator (&state_now.vels);

                          const bool local_df_converged = (error <= 1.0e-8);
                          const bool ow_stage_active_dbg = (new_holdups.liquid > 1.0e-6);

                          // Diagnostics only: do NOT modify the physical inputs.
                          // These values help determine whether the current superficial
                          // gas input has a sign compatible with the gas velocity predicted
                          // by the DF closure near alpha_g -> 0.
                          double gas_velocity_at_alpha_floor_dbg = 0.0;
                          bool gas_input_sign_mismatch_dbg = false;
                          {
                            df_local_value_state_t state_floor_dbg;
                            double alpha_o_floor_dbg = std::max (simplex_eps,
                                                                 std::min (1.0 - simplex_eps,
                                                                           beta_input * (1.0 - simplex_eps)));
                            evaluate_df_value_alphao (simplex_eps, alpha_o_floor_dbg, state_floor_dbg);
                            apply_df_flow_sign_to_state (state_floor_dbg);
                            gas_velocity_at_alpha_floor_dbg = state_floor_dbg.vels.gas;
                            gas_input_sign_mismatch_dbg =
                                (gas_superficial_velocity_input_raw * gas_velocity_at_alpha_floor_dbg < 0.0);
                          }

                          const double gas_liq_interfacial_tension_dbg = state_now.gas_liq_interfacial_tension;
                          const double liq_density_dbg = state_now.liq_density;
                          const double gas_density_dbg = state_now.gas_density;
                          const double bubble_rise_velocity_dbg = state_now.bubble_rise_velocity;
                          const double diametr_dimless_dbg = state_now.diametr_dimless;
                          const double Kut_number_dbg = state_now.Kut_number;
                          const double flooding_velocity_dbg = state_now.flooding_velocity;
                          const double ksi_dbg = state_now.ksi;
                          //const double eta_dbg = state_now.eta;
                          const double C_0_dbg = state_now.C0;
                          const double K_g_dbg = state_now.K_g;
                          const double drift_velocity_dbg = state_now.drift_velocity;
                          //const double gas_phase_velocity_dbg = state_now.vels.gas;
                          //const double liquid_phase_velocity_dbg = state_now.vels.liquid;
                          //const double sigma_o_dbg = state_now.sigma_o;
                          //const double sigma_w_dbg = state_now.sigma_w;
                          //const double wat_oil_interfacial_tension_dbg = state_now.oil_water_interfacial_tension;
                          //const double bubble_rise_velocity_OW_dbg = state_now.bubble_rise_velocity_OW;
                          const double wsn_C0_OW_dbg = state_now.C0_OW;
                          const double drift_velocity_OW_dbg = state_now.drift_velocity_OW;
                          //const double oil_phase_velocity_dbg = state_now.vels.oil;
                          //const double water_phase_velocity_dbg = state_now.vels.water;

                          wsncs->wsn_C_0 = C_0_dbg;
                          wsncs->wsn_drift_velocity = drift_velocity_dbg;
                          wsncs->wsn_C_0_OW = wsn_C0_OW_dbg;
                          wsncs->wsn_drift_velocity_OW = drift_velocity_OW_dbg;

                          PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                               "MERGEN: Well {}: it = {}, Seg_idx = {}, error = {}, average_volumetric_mixture_velocity = {}, average_volumetric_mixture_velocity_from_rate = {}, element_status->avg_xi = {}, "
                               "seg.wsn->pipe_props.area = {}, wsncs->wsn_mixture_molar_rate = {}, wsncs->wsn_C_0 = {}, wsncs->wsn_drift_velocity = {}, "
                               "gas_liq_interfacial_tension = {}, gas_density = {}, liquid_density = {}, diametr_dimless = {}, bubble_rise_velocity = {}, ksi = {}, "
                               "Kut_number = {}, flooding_velocity = {}, K_g = {}, element_status->phase_S[PHASE_GAS] = {}, "
                               "seg.wsn->pipe_props.diameter = {}, phase_S_WAT = {}, phase_S_OIL = {}, phase_S_GAS = {}, "
                               "alpha_input_WAT = {}, alpha_input_OIL = {}, alpha_input_GAS = {}, "
                               "jsg_in = {}, jso_in = {}, jsw_in = {}, "
                               "phase_inputs_from_lagged_map = {}, lagged_input_mode = HOLDUP_SPLIT_CURRENT_JMIX, lagged_phase_input_regularized = {}, lagged_lambda = {}, lagged_det = {}, "
                               "df_lagged_state_valid = {}, df_lagged_generation = {}, use_frozen_df_superficial_inputs = {}, "
                               "df_lagged_alpha_GAS = {}, df_lagged_alpha_OIL = {}, df_lagged_alpha_WAT = {}, "
                               "df_lagged_jsg_raw = {}, df_lagged_jso_raw = {}, df_lagged_jsw_raw = {}, df_lagged_jmix_raw = {}, "
                               "gas_root_sign_mismatch = {}, vg_floor = {}, local_df_converged = {}, ow_stage_active = {}, "
                               "wsncs->wsn_C_0_OW = {}, wsncs->wsn_drift_velocity_OW = {}, "
                               "new_holdups.gas = {}, new_holdups.liquid = {}, new_holdups.oil = {}, new_holdups.water = {}\n",
                               wcb_wis->get_well_name (), it, seg.wsn->wsn_index, error, average_volumetric_mixture_velocity_physical,
                               average_volumetric_mixture_velocity_from_rate_physical, element_status->avg_xi,
                               seg.wsn->pipe_props.area, wsncs->wsn_mixture_molar_rate, wsncs->wsn_C_0, wsncs->wsn_drift_velocity,
                               gas_liq_interfacial_tension_dbg, gas_density_dbg, liq_density_dbg, diametr_dimless_dbg, bubble_rise_velocity_dbg, ksi_dbg,
                               Kut_number_dbg, flooding_velocity_dbg, K_g_dbg, element_status->phase_S[PHASE_GAS],
                               seg.wsn->pipe_props.diameter, element_status->phase_S[PHASE_WATER], element_status->phase_S[PHASE_OIL], element_status->phase_S[PHASE_GAS],
                               alpha_w_seed_raw, alpha_o_seed_raw, alpha_g_seed_raw,
                               gas_superficial_velocity_input_raw, oil_superficial_velocity_input_raw, water_superficial_velocity_input_raw,
                               phase_inputs_from_lagged_map ? 1 : 0, lagged_phase_input_regularized ? 1 : 0,
                               lagged_phase_input_lambda, lagged_phase_input_det,
                               df_lagged_input_state_valid ? 1 : 0, df_lagged_input_state.generation,
                               use_frozen_df_superficial_inputs ? 1 : 0,
                               df_lagged_input_state.alpha_g, df_lagged_input_state.alpha_o, df_lagged_input_state.alpha_w,
                               df_lagged_input_state.jsg_raw, df_lagged_input_state.jso_raw, df_lagged_input_state.jsw_raw, df_lagged_input_state.jmix_raw,
                               gas_input_sign_mismatch_dbg ? 1 : 0, gas_velocity_at_alpha_floor_dbg, local_df_converged ? 1 : 0,
                               ow_stage_active_dbg ? 1 : 0,
                               wsncs->wsn_C_0_OW, wsncs->wsn_drift_velocity_OW,
                               new_holdups.gas, new_holdups.liquid, new_holdups.oil, new_holdups.water);

                          const double dbg_Rg_iter = new_holdups.gas   * new_vels.gas   - gas_superficial_velocity_input_raw;
                          const double dbg_Ro_iter = new_holdups.oil   * new_vels.oil   - oil_superficial_velocity_input_raw;
                          const double dbg_Rw_iter = new_holdups.water * new_vels.water - water_superficial_velocity_input_raw;
                          PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                               "MERGEN: Well {}: Seg_idx = {}, local residuals: Rg = {}, Ro = {}, Rw = {}\n",
                               wcb_wis->get_well_name (), seg.wsn->wsn_index, dbg_Rg_iter, dbg_Ro_iter, dbg_Rw_iter);

                      // ============================================================================
                      // POST-PROCESSING AFTER FIXED-POINT CONVERGENCE
                      // ============================================================================

                      // Final converged holdups
                      const double alpha_g = prev_holdups.gas;
                      const double alpha_l = prev_holdups.liquid;

                      double beta_o = 0.0;
                      if (fabs (alpha_l) > df_den_eps)
                        beta_o = prev_holdups.oil / alpha_l;

                      const double alpha_o = prev_holdups.oil;
                      const double alpha_w = prev_holdups.water;

                      // Regularization for the degenerate oil/water split when liquid holdup collapses.
                      // In that regime beta_o = alpha_o / alpha_l is not a numerically useful variable:
                      // its derivatives blow up like 1/alpha_l and 1/alpha_l^2 even though the oil/water
                      // partition has negligible physical influence on the gas-dominated DF state.
                      // We therefore freeze the chain rule through beta_o when alpha_l is too small.
                      const double alpha_l_beta_eps = 1.0e-6;
                      const bool ow_stage_active = (alpha_l > alpha_l_beta_eps);

                      // Persist actual DF holdups as segment-state seed for the next outer iteration.
                      wsncs->phase_S[PHASE_GAS] = alpha_g;
                      wsncs->phase_S[PHASE_OIL] = alpha_o;
                      wsncs->phase_S[PHASE_WATER] = alpha_w;

                      // IMPORTANT: once DF holdups replace the flash holdups in wsncs->phase_S,
                      // the associated derivative/state arrays must be made consistent as well.
                      // Otherwise the outer Newton sees a mixed state:
                      //   values phase_S -> from DF,
                      //   phase_D_S / avg_D_xi / avg_D_rho -> still from flash.
                      // This inconsistency becomes severe exactly in the user-forced constant-split
                      // runs, because the converged DF holdups may be far away from flash phase_S.

                      // -----------------------------
                      // Direct partial derivatives at FIXED (alpha_g, beta_o)
                      // -----------------------------
                      std::vector<double> D_gas_liq_interfacial_tension_partial_D_seg_vars (nseg_vars, 0.);
                      std::vector<double> D_liq_density_partial_D_seg_vars                 (nseg_vars, 0.);
                      std::vector<double> D_bubble_rise_velocity_partial_D_seg_vars        (nseg_vars, 0.);
                      std::vector<double> D_diametr_dimless_partial_D_seg_vars             (nseg_vars, 0.);
                      std::vector<double> D_Kut_number_partial_D_seg_vars                  (nseg_vars, 0.);
                      std::vector<double> D_flooding_velocity_partial_D_seg_vars           (nseg_vars, 0.);
                      std::vector<double> D_ksi_partial_D_seg_vars                         (nseg_vars, 0.);
                      std::vector<double> D_eta_partial_D_seg_vars                         (nseg_vars, 0.);
                      std::vector<double> D_K_g_partial_D_seg_vars                         (nseg_vars, 0.);
                      std::vector<double> D_C0_partial_D_seg_vars                          (nseg_vars, 0.);
                      std::vector<double> D_drift_velocity_partial_D_seg_vars              (nseg_vars, 0.);
                      std::vector<double> D_gas_phase_velocity_partial_D_seg_vars           (nseg_vars, 0.);
                      std::vector<double> D_liquid_phase_velocity_partial_D_seg_vars        (nseg_vars, 0.);
                      std::vector<double> D_drift_velocity_OW_partial_D_seg_vars           (nseg_vars, 0.);
                      std::vector<double> D_oil_phase_velocity_partial_D_seg_vars           (nseg_vars, 0.);
                      std::vector<double> D_water_phase_velocity_partial_D_seg_vars         (nseg_vars, 0.);

                      // Holdup derivatives in the final segment-variable basis.
                      // These are filled after solving the local 2x2 implicit DF system
                      // for [alpha_g, alpha_o], then reused in several later blocks.
                      std::vector<double> D_alpha_g_D_seg_vars                             (nseg_vars, 0.);
                      std::vector<double> D_alpha_l_D_seg_vars                             (nseg_vars, 0.);
                      std::vector<double> D_alpha_o_D_seg_vars                             (nseg_vars, 0.);
                      std::vector<double> D_alpha_w_D_seg_vars                             (nseg_vars, 0.);
                      std::vector<double> D_beta_o_D_seg_vars                              (nseg_vars, 0.);

                        const double surf_tension_mult = internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT ();

                        double sigma_o_raw = 0.;      // dynes/cm
                        double sigma_w_raw = 0.;      // dynes/cm
                        double D_sigma_o_D_p_psi = 0.; // correlation derivative, later converted to d(dynes/cm)/d(bar)
                        double D_sigma_w_D_p_psi = 0.; // correlation derivative, later converted to d(dynes/cm)/d(bar)

                        sigma_o_raw = pipe_gas_oil_interfacial_tension_and_deriv (
                            45.5,
                            element_status->p * converter_metric_to_field.pressure_mult (),
                            160.,
                            D_sigma_o_D_p_psi);

                        sigma_w_raw = pipe_gas_wat_interfacial_tension_and_deriv (
                            element_status->p * converter_metric_to_field.pressure_mult (),
                            160.,
                            D_sigma_w_D_p_psi);

                        D_sigma_o_D_p_psi *= converter_metric_to_field.pressure_mult ();
                        D_sigma_w_D_p_psi *= converter_metric_to_field.pressure_mult ();

                            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                 "Seg {} var {}: D_sigma_o_D_p ana={}, num={}, relerr={}\n",
                                 seg.wsn->wsn_index, fd_var_id,
                                 D_sigma_o_D_p_psi, sigma_o_num, fabs (D_sigma_o_D_p_psi - sigma_o_num));

                            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                 "Seg {} var {}: D_sigma_w_D_p ana={}, num={}, relerr={}\n",
                                 seg.wsn->wsn_index, fd_var_id,
                                 D_sigma_w_D_p_psi, sigma_w_num, fabs (D_sigma_w_D_p_psi - sigma_w_num));

                            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                 "Seg {} var {}: alpha_l value={}, avg(prev,next)={}, absdiff={}\n",
                                 seg.wsn->wsn_index, fd_var_id,
                                 alpha_l, liquid_hp_num, fabs (alpha_l - liquid_hp_num));

                        // Convert to N/m.
                        double sigma_o = surf_tension_mult * sigma_o_raw;
                        double sigma_w = surf_tension_mult * sigma_w_raw;

                        // ECLIPSE TDM 2020 uses combined-liquid properties weighted by the
                        // current oil and water volume fractions alpha_o and alpha_w (Eq. 8.82),
                        // not by the fixed superficial-input split. Using holdup-weighted liquid
                        // properties is important when the local DF solve moves far away from the
                        // flash/input phase split.
                        double gas_liq_interfacial_tension =
                            beta_o * sigma_o + (1.0 - beta_o) * sigma_w;

                        std::fill (D_gas_liq_interfacial_tension_partial_D_seg_vars.begin (),
                                   D_gas_liq_interfacial_tension_partial_D_seg_vars.end (), 0.0);
                        for (unsigned int id = 0; id < 1U + mp.nc; ++id)
                          {
                            const double D_sigma_o_x = (id == 0) ? (surf_tension_mult * D_sigma_o_D_p_psi) : 0.0;
                            const double D_sigma_w_x = (id == 0) ? (surf_tension_mult * D_sigma_w_D_p_psi) : 0.0;
                            D_gas_liq_interfacial_tension_partial_D_seg_vars[id] =
                                beta_o * D_sigma_o_x
                                + (1.0 - beta_o) * D_sigma_w_x
                                + (sigma_o - sigma_w) * D_beta_o_D_seg_vars[id];
                          }
                        PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                             "Seg {} var {}: D_gas_liq_interfacial_tension_partial_D_seg_vars[0] ana={}, num={}, relerr={}\n",
                             seg.wsn->wsn_index, fd_var_id,
                             D_gas_liq_interfacial_tension_partial_D_seg_vars[0],
                             gas_liq_interfacial_tension_numerical,
                             fabs (D_gas_liq_interfacial_tension_partial_D_seg_vars[0] - gas_liq_interfacial_tension_numerical));

                        const double D_gas_liq_interfacial_tension_D_beta_o = sigma_o - sigma_w;

                        // ECLIPSE TDM 2020: combined-liquid density weighted by alpha_o and alpha_w.
                        const double liq_density =
                            beta_o * element_status->phase_rho[PHASE_OIL]
                            + (1.0 - beta_o) * element_status->phase_rho[PHASE_WATER];

                        const double gas_density = element_status->phase_rho[PHASE_GAS];

                        for (unsigned int id = 0; id < 1U + mp.nc; ++id)
                          {
                            D_liq_density_partial_D_seg_vars[id] =
                                beta_o * element_status->phase_D_rho[id * mp.np + PHASE_OIL]
                                + (1.0 - beta_o) * element_status->phase_D_rho[id * mp.np + PHASE_WATER]
                                + (element_status->phase_rho[PHASE_OIL] - element_status->phase_rho[PHASE_WATER])
                                      * D_beta_o_D_seg_vars[id];
                          }

                        const double D_liq_density_D_beta_o =
                            element_status->phase_rho[PHASE_OIL] - element_status->phase_rho[PHASE_WATER];

                      // Vc = (sigma * g * |rho_l - rho_g| / rho_l^2)^(1/4)
                      double bubble_rise_velocity =
                          tnav_pow (tnav_div (gas_liq_interfacial_tension * internal_const::grav_metric ()
                                              * fabs (liq_density - gas_density),
                                              liq_density * liq_density), 0.25);

                      for (unsigned int id = 0; id < 1U + mp.nc; ++id)
                        {
                          D_bubble_rise_velocity_partial_D_seg_vars[id]
                              = 0.25 * tnav_div (1., tnav_pow (bubble_rise_velocity, 3))
                              * tnav_div (
                                    (D_gas_liq_interfacial_tension_partial_D_seg_vars[id]
                                         * internal_const::grav_metric () * fabs (liq_density - gas_density)
                                     + gas_liq_interfacial_tension * internal_const::grav_metric ()
                                           * tnav_sgn (liq_density - gas_density)
                                           * (D_liq_density_partial_D_seg_vars[id]
                                              - element_status->phase_D_rho[id * mp.np + PHASE_GAS]))
                                        * liq_density
                                        - 2. * gas_liq_interfacial_tension * internal_const::grav_metric ()
                                              * fabs (liq_density - gas_density)
                                              * D_liq_density_partial_D_seg_vars[id],
                                    tnav_pow (liq_density, 3));
                        }

                      double D_bubble_rise_velocity_D_beta_o = 0.;
                      {
                        const double D_delta_rho_D_beta_o = D_liq_density_D_beta_o;

                        D_bubble_rise_velocity_D_beta_o =
                            0.25 * tnav_div (1., tnav_pow (bubble_rise_velocity, 3))
                            * tnav_div (
                                  (D_gas_liq_interfacial_tension_D_beta_o * internal_const::grav_metric ()
                                       * fabs (liq_density - gas_density)
                                   + gas_liq_interfacial_tension * internal_const::grav_metric ()
                                         * tnav_sgn (liq_density - gas_density) * D_delta_rho_D_beta_o)
                                      * liq_density
                                      - 2. * gas_liq_interfacial_tension * internal_const::grav_metric ()
                                            * fabs (liq_density - gas_density) * D_liq_density_D_beta_o,
                                  tnav_pow (liq_density, 3));
                      }

                      // D_hat
                      double diametr_dimless = 0.;
                      if (liq_density - gas_density > 0.)
                        {
                          diametr_dimless =
                              sqrt (tnav_div (internal_const::grav_metric () * (liq_density - gas_density),
                                              gas_liq_interfacial_tension))
                              * seg.wsn->pipe_props.diameter;
                        }

                      for (unsigned int id = 0; id < 1U + mp.nc; ++id)
                        {
                          D_diametr_dimless_partial_D_seg_vars[id]
                              = 0.5 * seg.wsn->pipe_props.diameter
                              * sqrt (tnav_div (gas_liq_interfacial_tension,
                                                internal_const::grav_metric () * (liq_density - gas_density)))
                              * tnav_div (
                                    internal_const::grav_metric ()
                                        * tnav_sgn (liq_density - gas_density)
                                        * (D_liq_density_partial_D_seg_vars[id]
                                           - element_status->phase_D_rho[id * mp.np + PHASE_GAS])
                                        * gas_liq_interfacial_tension
                                        - internal_const::grav_metric ()
                                              * fabs (liq_density - gas_density)
                                              * D_gas_liq_interfacial_tension_partial_D_seg_vars[id],
                                    tnav_pow (gas_liq_interfacial_tension, 2));
                        }

                      double D_diametr_dimless_D_beta_o = 0.;
                      if (liq_density - gas_density > 0.)
                        {
                          D_diametr_dimless_D_beta_o =
                              0.5 * seg.wsn->pipe_props.diameter
                              * sqrt (tnav_div (gas_liq_interfacial_tension,
                                                internal_const::grav_metric () * (liq_density - gas_density)))
                              * tnav_div (
                                    internal_const::grav_metric ()
                                        * tnav_sgn (liq_density - gas_density)
                                        * D_liq_density_D_beta_o
                                        * gas_liq_interfacial_tension
                                        - internal_const::grav_metric ()
                                              * fabs (liq_density - gas_density)
                                              * D_gas_liq_interfacial_tension_D_beta_o,
                                    tnav_pow (gas_liq_interfacial_tension, 2));
                        }

                      double linear_interpolation_derivative = 0.;
                      double Kut_number =
                          compute_critical_Kutateladze_number_by_diametr (diametr_dimless,
                                                                          linear_interpolation_derivative);

                      for (unsigned int id = 0; id < 1U + mp.nc; ++id)
                        {
                          D_Kut_number_partial_D_seg_vars[id]
                              = linear_interpolation_derivative * D_diametr_dimless_partial_D_seg_vars[id];
                        }

                      const double D_Kut_number_D_beta_o =
                          linear_interpolation_derivative * D_diametr_dimless_D_beta_o;

                      // Vsgf = Ku * sqrt(rho_l / rho_g) * Vc
                      double flooding_velocity =
                          Kut_number * sqrt (liq_density / gas_density) * bubble_rise_velocity;

                      for (unsigned int id = 0; id < 1U + mp.nc; ++id)
                        {
                          const double first =
                              D_Kut_number_partial_D_seg_vars[id] * sqrt (liq_density / gas_density)
                              * bubble_rise_velocity;

                          const double second =
                              Kut_number * 0.5 * tnav_div (1., tnav_pow (liq_density / gas_density, 0.5))
                              * bubble_rise_velocity
                              * (D_liq_density_partial_D_seg_vars[id] * gas_density
                                 - element_status->phase_D_rho[id * mp.np + PHASE_GAS] * liq_density)
                              / tnav_pow (gas_density, 2);

                          const double third =
                              Kut_number * sqrt (liq_density / gas_density)
                              * D_bubble_rise_velocity_partial_D_seg_vars[id];

                          D_flooding_velocity_partial_D_seg_vars[id] = first + second + third;
                        }

                      double D_flooding_velocity_D_beta_o = 0.;
                      {
                        const double first =
                            D_Kut_number_D_beta_o * sqrt (liq_density / gas_density)
                            * bubble_rise_velocity;

                        const double second =
                            Kut_number * 0.5 * tnav_div (1., tnav_pow (liq_density / gas_density, 0.5))
                            * bubble_rise_velocity
                            * (D_liq_density_D_beta_o * gas_density) / tnav_pow (gas_density, 2);

                        const double third =
                            Kut_number * sqrt (liq_density / gas_density)
                            * D_bubble_rise_velocity_D_beta_o;

                        D_flooding_velocity_D_beta_o = first + second + third;
                      }


                      // --------------------------------------
                      // C0(alpha_g, Vm, Vsgf) and direct partials
                      // --------------------------------------
                      const double A = 1.2;
                      const double B = 0.3;
                      const double F_v = 1.0;
                      const double V_sgf = df_safe_nonzero (flooding_velocity);

                      double ksi_second =
                          tnav_div (F_v * alpha_g * fabs (mixture_superficial_velocity), V_sgf);
                      double ksi = std::max (alpha_g, ksi_second);
                      unsigned int ksi_flag = 0;

                      double D_ksi_D_alpha_g = 0.;
                      double D_ksi_D_beta_o  = 0.;

                      if (alpha_g > ksi_second)
                        {
                          ksi_flag = 1;
                          std::fill (D_ksi_partial_D_seg_vars.begin (), D_ksi_partial_D_seg_vars.end (), 0.);
                          D_ksi_D_alpha_g = 1.;
                          D_ksi_D_beta_o  = 0.;
                        }
                      else
                        {
                          ksi_flag = 2;

                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              D_ksi_partial_D_seg_vars[id]
                                  = tnav_div (
                                        F_v * alpha_g
                                            * (tnav_sgn (mixture_superficial_velocity)
                                                   * D_mixture_superficial_velocity_D_seg_vars[id] * V_sgf
                                               - fabs (mixture_superficial_velocity)
                                                     * D_flooding_velocity_partial_D_seg_vars[id]),
                                        tnav_pow (V_sgf, 2));
                            }

                          D_ksi_D_alpha_g = tnav_div (F_v * fabs (mixture_superficial_velocity), V_sgf);
                          D_ksi_D_beta_o  = -tnav_div (F_v * alpha_g * fabs (mixture_superficial_velocity)
                                                       * D_flooding_velocity_D_beta_o,
                                                       tnav_pow (V_sgf, 2));
                        }

                      double eta = (ksi - B) / (1. - B);
                      double D_eta_D_alpha_g = 0.;
                      double D_eta_D_beta_o  = 0.;

                      for (unsigned int id = 0; id < nseg_vars; ++id)
                        D_eta_partial_D_seg_vars[id] = D_ksi_partial_D_seg_vars[id] / (1. - B);

                      D_eta_D_alpha_g = D_ksi_D_alpha_g / (1. - B);
                      D_eta_D_beta_o  = D_ksi_D_beta_o  / (1. - B);

                      if (!std::isfinite (eta))
                        {
                          eta = (ksi > B) ? 1.0 : 0.0;
                          std::fill (D_eta_partial_D_seg_vars.begin (), D_eta_partial_D_seg_vars.end (), 0.0);
                          D_eta_D_alpha_g = 0.0;
                          D_eta_D_beta_o  = 0.0;
                        }
                      else if (eta < 0.0)
                        {
                          eta = 0.0;
                          std::fill (D_eta_partial_D_seg_vars.begin (), D_eta_partial_D_seg_vars.end (), 0.0);
                          D_eta_D_alpha_g = 0.0;
                          D_eta_D_beta_o  = 0.0;
                        }
                      else if (eta > 1.0)
                        {
                          eta = 1.0;
                          std::fill (D_eta_partial_D_seg_vars.begin (), D_eta_partial_D_seg_vars.end (), 0.0);
                          D_eta_D_alpha_g = 0.0;
                          D_eta_D_beta_o  = 0.0;
                        }

                      double C_0 = A / (1. + (A - 1.) * eta * eta);

                      double D_C0_D_alpha_g = 0.;
                      double D_C0_D_beta_o  = 0.;

                      for (unsigned int id = 0; id < nseg_vars; ++id)
                        {
                          D_C0_partial_D_seg_vars[id]
                              = -A * (A - 1.) * 2. * eta * D_eta_partial_D_seg_vars[id]
                              / tnav_pow (1. + (A - 1.) * eta * eta, 2);
                        }

                      D_C0_D_alpha_g =
                          -A * (A - 1.) * 2. * eta * D_eta_D_alpha_g
                          / tnav_pow (1. + (A - 1.) * eta * eta, 2);

                      D_C0_D_beta_o =
                          -A * (A - 1.) * 2. * eta * D_eta_D_beta_o
                          / tnav_pow (1. + (A - 1.) * eta * eta, 2);

                      wsncs->wsn_C_0 = C_0_dbg;

                      // --------------------------------------
                      // K_g(alpha_g, beta_o)
                      // --------------------------------------
                      const double a1_Kg = 0.2;
                      const double a2_Kg = 0.4;

                      double K_g_low  = tnav_div (1.53, C_0);
                      double K_g_high = Kut_number;
                      double K_g = 0.;

                      std::vector<double> D_K_g_low_partial_D_seg_vars (nseg_vars, 0.);
                      for (unsigned int id = 0; id < nseg_vars; ++id)
                        {
                          D_K_g_low_partial_D_seg_vars[id]
                              = tnav_div (-1.53, C_0 * C_0) * D_C0_partial_D_seg_vars[id];
                        }

                      const double D_K_g_low_D_alpha_g = tnav_div (-1.53, C_0 * C_0) * D_C0_D_alpha_g;
                      const double D_K_g_low_D_beta_o  = tnav_div (-1.53, C_0 * C_0) * D_C0_D_beta_o;

                      double D_K_g_D_alpha_g = 0.;
                      double D_K_g_D_beta_o  = 0.;

                      if (alpha_g < a1_Kg)
                        {
                          K_g = K_g_low;

                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            D_K_g_partial_D_seg_vars[id] = D_K_g_low_partial_D_seg_vars[id];

                          D_K_g_D_alpha_g = D_K_g_low_D_alpha_g;
                          D_K_g_D_beta_o  = D_K_g_low_D_beta_o;
                        }
                      else if (alpha_g > a2_Kg)
                        {
                          K_g = K_g_high;

                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            D_K_g_partial_D_seg_vars[id] = D_Kut_number_partial_D_seg_vars[id];

                          D_K_g_D_alpha_g = 0.;
                          D_K_g_D_beta_o  = D_Kut_number_D_beta_o;
                        }
                      else
                        {
                          const double t = (alpha_g - a1_Kg) / (a2_Kg - a1_Kg);

                          K_g = (1. - t) * K_g_low + t * K_g_high;

                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              D_K_g_partial_D_seg_vars[id]
                                  = (1. - t) * D_K_g_low_partial_D_seg_vars[id]
                                  + t * D_Kut_number_partial_D_seg_vars[id];
                            }

                          D_K_g_D_alpha_g =
                              (K_g_high - K_g_low) / (a2_Kg - a1_Kg)
                              + (1. - t) * D_K_g_low_D_alpha_g;

                          D_K_g_D_beta_o =
                              (1. - t) * D_K_g_low_D_beta_o
                              + t * D_Kut_number_D_beta_o;
                        }

                      // --------------------------------------
                      // Gas drift velocity Vd(alpha_g, beta_o)
                      // --------------------------------------
                      const double sqrt_gas_over_liq = sqrt (tnav_div (gas_density, liq_density));

                      std::vector<double> D_sqrt_gas_over_liq_partial_D_seg_vars (nseg_vars, 0.);
                      for (unsigned int id = 0; id < 1U + mp.nc; ++id)
                        {
                          D_sqrt_gas_over_liq_partial_D_seg_vars[id]
                              = 0.5 * tnav_div (1., sqrt_gas_over_liq)
                              * tnav_div (element_status->phase_D_rho[id * mp.np + PHASE_GAS] * liq_density
                                          - gas_density * D_liq_density_partial_D_seg_vars[id],
                                          tnav_pow (liq_density, 2));
                        }

                      const double D_sqrt_gas_over_liq_D_beta_o =
                          0.5 * tnav_div (1., sqrt_gas_over_liq)
                          * tnav_div (-gas_density * D_liq_density_D_beta_o,
                                      tnav_pow (liq_density, 2));

                      const double numerator_gd =
                          (1. - alpha_g * C_0) * C_0 * K_g * bubble_rise_velocity;

                      const double denominator_gd =
                          1. + alpha_g * C_0 * (sqrt_gas_over_liq - 1.);

                      double drift_velocity = tnav_div (numerator_gd, denominator_gd);
                      drift_velocity *= drift_incl_mult;
                      wsncs->wsn_drift_velocity = drift_velocity_dbg;

                      double D_numerator_gd_D_alpha_g =
                          (-C_0 - alpha_g * D_C0_D_alpha_g) * C_0 * K_g * bubble_rise_velocity
                          + (1. - alpha_g * C_0) * D_C0_D_alpha_g * K_g * bubble_rise_velocity
                          + (1. - alpha_g * C_0) * C_0 * D_K_g_D_alpha_g * bubble_rise_velocity;

                      double D_denominator_gd_D_alpha_g =
                          (C_0 + alpha_g * D_C0_D_alpha_g) * (sqrt_gas_over_liq - 1.);

                      double D_numerator_gd_D_beta_o =
                          (-alpha_g * D_C0_D_beta_o) * C_0 * K_g * bubble_rise_velocity
                          + (1. - alpha_g * C_0) * D_C0_D_beta_o * K_g * bubble_rise_velocity
                          + (1. - alpha_g * C_0) * C_0 * D_K_g_D_beta_o * bubble_rise_velocity
                          + (1. - alpha_g * C_0) * C_0 * K_g * D_bubble_rise_velocity_D_beta_o;

                      double D_denominator_gd_D_beta_o =
                          alpha_g * D_C0_D_beta_o * (sqrt_gas_over_liq - 1.)
                          + alpha_g * C_0 * D_sqrt_gas_over_liq_D_beta_o;

                      double D_drift_velocity_D_alpha_g =
                          drift_incl_mult * tnav_div (D_numerator_gd_D_alpha_g * denominator_gd
                                    - numerator_gd * D_denominator_gd_D_alpha_g,
                                    denominator_gd * denominator_gd);

                      double D_drift_velocity_D_beta_o =
                          drift_incl_mult * tnav_div (D_numerator_gd_D_beta_o * denominator_gd
                                    - numerator_gd * D_denominator_gd_D_beta_o,
                                    denominator_gd * denominator_gd);

                      for (unsigned int id = 0; id < nseg_vars; ++id)
                        {
                          const double D_num =
                              (-alpha_g * D_C0_partial_D_seg_vars[id]) * C_0 * K_g * bubble_rise_velocity
                              + (1. - alpha_g * C_0) * D_C0_partial_D_seg_vars[id] * K_g * bubble_rise_velocity
                              + (1. - alpha_g * C_0) * C_0 * D_K_g_partial_D_seg_vars[id] * bubble_rise_velocity
                              + (1. - alpha_g * C_0) * C_0 * K_g * D_bubble_rise_velocity_partial_D_seg_vars[id];

                          const double D_den =
                              alpha_g * D_C0_partial_D_seg_vars[id] * (sqrt_gas_over_liq - 1.)
                              + alpha_g * C_0 * D_sqrt_gas_over_liq_partial_D_seg_vars[id];

                          D_drift_velocity_partial_D_seg_vars[id] =
                              drift_incl_mult * tnav_div (D_num * denominator_gd - numerator_gd * D_den,
                                        denominator_gd * denominator_gd);
                        }

                      // --------------------------------------
                      // Gas / liquid actual velocities
                      // --------------------------------------
                      double gas_phase_velocity = C_0 * mixture_superficial_velocity + drift_velocity;

                      double D_gas_phase_velocity_D_alpha_g =
                          D_C0_D_alpha_g * mixture_superficial_velocity + D_drift_velocity_D_alpha_g;

                      double D_gas_phase_velocity_D_beta_o =
                          D_C0_D_beta_o * mixture_superficial_velocity + D_drift_velocity_D_beta_o;

                      for (unsigned int id = 0; id < nseg_vars; ++id)
                        {
                          D_gas_phase_velocity_partial_D_seg_vars[id]
                              = D_C0_partial_D_seg_vars[id] * mixture_superficial_velocity
                              + C_0 * D_mixture_superficial_velocity_D_seg_vars[id]
                              + D_drift_velocity_partial_D_seg_vars[id];
                        }

                      double liquid_phase_velocity = 0.0;
                      double D_liquid_phase_velocity_D_alpha_g = 0.0;
                      double D_liquid_phase_velocity_D_beta_o = 0.0;

                      if (alpha_g > 1.0)
                        {
                          liquid_phase_velocity = 0.0;
                          D_liquid_phase_velocity_D_alpha_g = 0.0;
                          D_liquid_phase_velocity_D_beta_o = 0.0;
                          std::fill (D_liquid_phase_velocity_partial_D_seg_vars.begin (),
                                     D_liquid_phase_velocity_partial_D_seg_vars.end (), 0.0);
                        }
                      else
                        {
                          const double f_gl = tnav_div (1. - alpha_g * C_0, 1. - alpha_g);
                          const double g_gl = tnav_div (alpha_g, 1. - alpha_g);

                          liquid_phase_velocity =
                              f_gl * mixture_superficial_velocity - g_gl * drift_velocity;

                          const double D_f_gl_D_alpha_g =
                              tnav_div (1. - C_0 - alpha_g * (1. - alpha_g) * D_C0_D_alpha_g,
                                        tnav_pow (1. - alpha_g, 2));

                          const double D_g_gl_D_alpha_g =
                              tnav_div (1., tnav_pow (1. - alpha_g, 2));

                          const double D_f_gl_D_beta_o =
                              -tnav_div (alpha_g * D_C0_D_beta_o, (1. - alpha_g));

                          D_liquid_phase_velocity_D_alpha_g =
                              D_f_gl_D_alpha_g * mixture_superficial_velocity
                              - D_g_gl_D_alpha_g * drift_velocity
                              - g_gl * D_drift_velocity_D_alpha_g;

                          D_liquid_phase_velocity_D_beta_o =
                              D_f_gl_D_beta_o * mixture_superficial_velocity
                              - g_gl * D_drift_velocity_D_beta_o;

                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              D_liquid_phase_velocity_partial_D_seg_vars[id]
                                  = -tnav_div (alpha_g * D_C0_partial_D_seg_vars[id], (1. - alpha_g))
                                        * mixture_superficial_velocity
                                    + f_gl * D_mixture_superficial_velocity_D_seg_vars[id]
                                    - g_gl * D_drift_velocity_partial_D_seg_vars[id];
                            }
                        }

                      // --------------------------------------
                      // Oil/water stage (beta_o in liquid)
                      // --------------------------------------
                      const double B1_OW = 0.4;
                      const double B2_OW = 0.7;
                      const double A_OW  = 1.2;

                      double wsn_C0_OW = 0.;
                      double D_wsn_C0_OW_D_beta_o = 0.;

                      if (beta_o < B1_OW)
                        {
                          wsn_C0_OW = A_OW;
                          D_wsn_C0_OW_D_beta_o = 0.;
                        }
                      else if (beta_o > B2_OW)
                        {
                          wsn_C0_OW = 1.;
                          D_wsn_C0_OW_D_beta_o = 0.;
                        }
                      else
                        {
                          wsn_C0_OW = interpolate_y_against_x (beta_o, B1_OW, B2_OW, A_OW, 1.);
                          D_wsn_C0_OW_D_beta_o = -(A_OW - 1.) / (B2_OW - B1_OW);
                        }

                      double D_gas_oil_interfacial_tension_D_p = 0.;
                      double gas_oil_interfacial_tension =
                          internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT ()
                          * pipe_gas_oil_interfacial_tension_and_deriv (
                                45.5,
                                element_status->p * converter_metric_to_field.pressure_mult (),
                                160.,
                                D_gas_oil_interfacial_tension_D_p);

                      double D_gas_wat_interfacial_tension_D_p = 0.;
                      double gas_wat_interfacial_tension =
                          internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT ()
                          * pipe_gas_wat_interfacial_tension_and_deriv (
                                element_status->p * converter_metric_to_field.pressure_mult (),
                                160.,
                                D_gas_wat_interfacial_tension_D_p);

                      double wat_oil_interfacial_tension_arg =
                          gas_oil_interfacial_tension * beta_o
                          - gas_wat_interfacial_tension * (1. - beta_o);

                      double wat_oil_interfacial_tension = fabs (wat_oil_interfacial_tension_arg);

                      std::vector<double> D_wat_oil_interfacial_tension_partial_D_seg_vars (nseg_vars, 0.);
                      for (unsigned int id = 0; id < nseg_vars; ++id)
                        {
                          double D_sigma_go = 0.;
                          double D_sigma_gw = 0.;

                          if (id == 0)
                            {
                              D_sigma_go =
                                    internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT ()
                                    * converter_metric_to_field.pressure_mult ()
                                    * D_gas_oil_interfacial_tension_D_p;

                                D_sigma_gw =
                                    internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT ()
                                    * converter_metric_to_field.pressure_mult ()
                                    * D_gas_wat_interfacial_tension_D_p;
                            }

                          D_wat_oil_interfacial_tension_partial_D_seg_vars[id]
                              = tnav_sgn (wat_oil_interfacial_tension_arg)
                              * (beta_o * D_sigma_go - (1. - beta_o) * D_sigma_gw);
                        }

                      const double D_wat_oil_interfacial_tension_D_beta_o =
                          tnav_sgn (wat_oil_interfacial_tension_arg)
                          * (gas_oil_interfacial_tension + gas_wat_interfacial_tension);

                      double bubble_rise_velocity_OW =
                          tnav_pow (
                              wat_oil_interfacial_tension * internal_const::grav_metric ()
                                  * (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL])
                                  / (element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER]),
                              0.25);

                      std::vector<double> D_bubble_rise_velocity_OW_partial_D_seg_vars (nseg_vars, 0.);
                      for (unsigned int id = 0; id < 1U + mp.nc; ++id)
                        {
                          D_bubble_rise_velocity_OW_partial_D_seg_vars[id]
                              = 0.25
                              * tnav_div (
                                    1.,
                                    tnav_pow (
                                        wat_oil_interfacial_tension * internal_const::grav_metric ()
                                            * (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL])
                                            / (element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER]),
                                        0.75))
                              * tnav_div (
                                    (D_wat_oil_interfacial_tension_partial_D_seg_vars[id]
                                         * internal_const::grav_metric ()
                                         * (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL])
                                     + wat_oil_interfacial_tension * internal_const::grav_metric ()
                                           * (element_status->phase_D_rho[id * mp.np + PHASE_WATER]
                                              - element_status->phase_D_rho[id * mp.np + PHASE_OIL]))
                                        * element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER]
                                        - wat_oil_interfacial_tension * internal_const::grav_metric ()
                                              * (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL])
                                              * 2. * element_status->phase_rho[PHASE_WATER]
                                              * element_status->phase_D_rho[id * mp.np + PHASE_WATER],
                                    tnav_pow (element_status->phase_rho[PHASE_WATER], 4));
                        }

                      double D_bubble_rise_velocity_OW_D_beta_o =
                          0.25
                          * tnav_div (
                                1.,
                                tnav_pow (
                                    wat_oil_interfacial_tension * internal_const::grav_metric ()
                                        * (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL])
                                        / (element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER]),
                                    0.75))
                          * tnav_div (
                                D_wat_oil_interfacial_tension_D_beta_o
                                    * internal_const::grav_metric ()
                                    * (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL]),
                                tnav_pow (element_status->phase_rho[PHASE_WATER], 2));

                      double drift_velocity_OW =
                          1.53 * bubble_rise_velocity_OW * tnav_pow (1. - beta_o, 2);
                      drift_velocity_OW *= drift_incl_mult;

                      for (unsigned int id = 0; id < nseg_vars; ++id)
                        {
                          D_drift_velocity_OW_partial_D_seg_vars[id]
                              = drift_incl_mult * 1.53 * D_bubble_rise_velocity_OW_partial_D_seg_vars[id]
                                * tnav_pow (1. - beta_o, 2);
                        }

                      double D_drift_velocity_OW_D_beta_o =
                          drift_incl_mult * (1.53 * D_bubble_rise_velocity_OW_D_beta_o * tnav_pow (1. - beta_o, 2)
                          - 1.53 * bubble_rise_velocity_OW * 2. * (1. - beta_o));

                      wsncs->wsn_C_0_OW = wsn_C0_OW_dbg;
                      wsncs->wsn_drift_velocity_OW = drift_velocity_OW_dbg;

                      // Oil / water velocities
                      double oil_phase_velocity = 0.0;
                      double D_oil_phase_velocity_D_alpha_g = 0.0;
                      double D_oil_phase_velocity_D_beta_o = 0.0;

                      double water_phase_velocity = 0.0;
                      double D_water_phase_velocity_D_alpha_g = 0.0;
                      double D_water_phase_velocity_D_beta_o = 0.0;

                      const bool beta_o_is_one = (1.0 - beta_o <= tnm::min_compare);

                      if (beta_o_is_one)
                        {
                          // Limit beta_o -> 1.
                          // Water holdup tends to zero, but the water phase velocity itself
                          // tends to the liquid-phase velocity. Hard-zeroing the velocity here
                          // destroys d(alpha_w * V_w)/dx and makes q_c Jacobian rows singular.
                          wsn_C0_OW = 1.0;
                          D_wsn_C0_OW_D_beta_o = 0.0;
                          drift_velocity_OW = 0.0;
                          D_drift_velocity_OW_D_beta_o = 0.0;
                          std::fill (D_drift_velocity_OW_partial_D_seg_vars.begin (),
                                     D_drift_velocity_OW_partial_D_seg_vars.end (), 0.0);

                          oil_phase_velocity = liquid_phase_velocity;
                          D_oil_phase_velocity_D_alpha_g = D_liquid_phase_velocity_D_alpha_g;
                          D_oil_phase_velocity_D_beta_o = D_liquid_phase_velocity_D_beta_o;

                          water_phase_velocity = liquid_phase_velocity;
                          D_water_phase_velocity_D_alpha_g = D_liquid_phase_velocity_D_alpha_g;
                          D_water_phase_velocity_D_beta_o = D_liquid_phase_velocity_D_beta_o;

                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              D_oil_phase_velocity_partial_D_seg_vars[id] =
                                  D_liquid_phase_velocity_partial_D_seg_vars[id];
                              D_water_phase_velocity_partial_D_seg_vars[id] =
                                  D_liquid_phase_velocity_partial_D_seg_vars[id];
                            }
                        }
                      else
                        {
                          oil_phase_velocity =
                              wsn_C0_OW * liquid_phase_velocity + drift_velocity_OW;

                          D_oil_phase_velocity_D_alpha_g =
                              wsn_C0_OW * D_liquid_phase_velocity_D_alpha_g;

                          D_oil_phase_velocity_D_beta_o =
                              D_wsn_C0_OW_D_beta_o * liquid_phase_velocity
                              + wsn_C0_OW * D_liquid_phase_velocity_D_beta_o
                              + D_drift_velocity_OW_D_beta_o;

                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              D_oil_phase_velocity_partial_D_seg_vars[id] =
                                  wsn_C0_OW * D_liquid_phase_velocity_partial_D_seg_vars[id]
                                  + D_drift_velocity_OW_partial_D_seg_vars[id];
                            }

                          const double f_ow = tnav_div (1. - beta_o * wsn_C0_OW, 1. - beta_o);
                          const double g_ow = tnav_div (beta_o, 1. - beta_o);

                          water_phase_velocity =
                              f_ow * liquid_phase_velocity - g_ow * drift_velocity_OW;

                          const double D_f_ow_D_beta_o =
                              tnav_div (1. - wsn_C0_OW - beta_o * (1. - beta_o) * D_wsn_C0_OW_D_beta_o,
                                        tnav_pow (1. - beta_o, 2));

                          const double D_g_ow_D_beta_o =
                              tnav_div (1., tnav_pow (1. - beta_o, 2));

                          D_water_phase_velocity_D_alpha_g =
                              f_ow * D_liquid_phase_velocity_D_alpha_g;

                          D_water_phase_velocity_D_beta_o =
                              D_f_ow_D_beta_o * liquid_phase_velocity
                              + f_ow * D_liquid_phase_velocity_D_beta_o
                              - D_g_ow_D_beta_o * drift_velocity_OW
                              - g_ow * D_drift_velocity_OW_D_beta_o;

                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              D_water_phase_velocity_partial_D_seg_vars[id] =
                                  f_ow * D_liquid_phase_velocity_partial_D_seg_vars[id]
                                  - g_ow * D_drift_velocity_OW_partial_D_seg_vars[id];
                            }
                        }

                      wsncs->wsn_C_0_OW = wsn_C0_OW_dbg;
                      wsncs->wsn_drift_velocity_OW = drift_velocity_OW_dbg;

                      // -----------------------------
                      // Derivatives of holdups from the IMPLICIT local DF system
                      // in the NEW basis u = [alpha_g, alpha_o].
                      //
                      // Residuals:
                      //   Rg(alpha_g, alpha_o, x) = alpha_g * Vg(alpha_g, beta_o) - j_g(x) = 0
                      //   Ro(alpha_g, alpha_o, x) = alpha_o * Vo(alpha_g, beta_o) - j_o(x) = 0
                      //   alpha_w = 1 - alpha_g - alpha_o,
                      //   beta_o  = alpha_o / (1 - alpha_g).
                      // -----------------------------
                      const double alpha_l_safe = std::max (alpha_l, alpha_l_beta_eps);
                      const double D_beta_o_D_alpha_g_at_alpha_o =
                          ow_stage_active ? (alpha_o / (alpha_l_safe * alpha_l_safe)) : 0.0;
                      const double D_beta_o_D_alpha_o_at_alpha_g =
                          ow_stage_active ? (1.0 / alpha_l_safe) : 0.0;

                      const double D_gas_phase_velocity_D_alpha_g_at_alpha_o =
                          D_gas_phase_velocity_D_alpha_g + D_gas_phase_velocity_D_beta_o * D_beta_o_D_alpha_g_at_alpha_o;
                      const double D_gas_phase_velocity_D_alpha_o =
                          D_gas_phase_velocity_D_beta_o * D_beta_o_D_alpha_o_at_alpha_g;

                      const double D_oil_phase_velocity_D_alpha_g_at_alpha_o =
                          D_oil_phase_velocity_D_alpha_g + D_oil_phase_velocity_D_beta_o * D_beta_o_D_alpha_g_at_alpha_o;
                      const double D_oil_phase_velocity_D_alpha_o =
                          D_oil_phase_velocity_D_beta_o * D_beta_o_D_alpha_o_at_alpha_g;

                      double J11 = gas_phase_velocity + alpha_g * D_gas_phase_velocity_D_alpha_g_at_alpha_o;
                      double J12 = alpha_g * D_gas_phase_velocity_D_alpha_o;
                      double J21 = alpha_o * D_oil_phase_velocity_D_alpha_g_at_alpha_o;
                      double J22 = oil_phase_velocity + alpha_o * D_oil_phase_velocity_D_alpha_o;

                      double detJ = J11 * J22 - J12 * J21;
                      const double J_scale = std::max (1.0,
                                                       std::max (std::max (fabs (J11), fabs (J12)),
                                                                 std::max (fabs (J21), fabs (J22))));
                      const double jac_reg = 1.0e-12 * J_scale;
                      const bool used_small_j_fallback = (!std::isfinite (detJ) || fabs (detJ) < jac_reg * jac_reg);
                      if (used_small_j_fallback)
                        {
                          J11 += jac_reg;
                          J22 += jac_reg;
                          detJ = J11 * J22 - J12 * J21;
                        }

                      for (unsigned int id = 0; id < nseg_vars; ++id)
                        {
                          const double Rg_x =
                              alpha_g * D_gas_phase_velocity_partial_D_seg_vars[id]
                              - D_gas_superficial_velocity_input_D_seg_vars[id];

                          const double Ro_x =
                              alpha_o * D_oil_phase_velocity_partial_D_seg_vars[id]
                              - D_oil_superficial_velocity_input_D_seg_vars[id];

                          D_alpha_g_D_seg_vars[id] =
                              tnav_div (-J22 * Rg_x + J12 * Ro_x, detJ);

                          D_alpha_o_D_seg_vars[id] =
                              tnav_div ( J21 * Rg_x - J11 * Ro_x, detJ);

                          D_alpha_l_D_seg_vars[id] = -D_alpha_g_D_seg_vars[id];
                          D_alpha_w_D_seg_vars[id] = -D_alpha_g_D_seg_vars[id] - D_alpha_o_D_seg_vars[id];

                          if (ow_stage_active)
                            {
                              D_beta_o_D_seg_vars[id] =
                                  tnav_div (D_alpha_o_D_seg_vars[id] * alpha_l - alpha_o * D_alpha_l_D_seg_vars[id],
                                            alpha_l_safe * alpha_l_safe);
                            }
                          else
                            {
                              D_beta_o_D_seg_vars[id] = 0.0;
                            }
                        }

                            // -----------------------------
                            // TOTAL derivatives = direct partial + chain through alpha_g, beta_o
                            // -----------------------------
                            for (unsigned int id = 0; id < nseg_vars; ++id)
                              {
                                wsncs->D_C0_D_seg_vars[id] =
                                    D_C0_partial_D_seg_vars[id]
                                    + D_C0_D_alpha_g * D_alpha_g_D_seg_vars[id]
                                    + D_C0_D_beta_o  * D_beta_o_D_seg_vars[id];

                                wsncs->D_drift_velocity_D_seg_vars[id] =
                                    D_drift_velocity_partial_D_seg_vars[id]
                                    + D_drift_velocity_D_alpha_g * D_alpha_g_D_seg_vars[id]
                                    + D_drift_velocity_D_beta_o  * D_beta_o_D_seg_vars[id];

                                wsncs->D_C0_OW_D_seg_vars[id] =
                                    D_wsn_C0_OW_D_beta_o * D_beta_o_D_seg_vars[id];

                                wsncs->D_drift_velocity_OW_D_seg_vars[id] =
                                    D_drift_velocity_OW_partial_D_seg_vars[id]
                                    + D_drift_velocity_OW_D_beta_o * D_beta_o_D_seg_vars[id];
                              }

                      // total phase velocity derivatives
                      std::vector<double> D_gas_phase_velocity_D_seg_vars_final    (nseg_vars, 0.);
                      std::vector<double> D_liquid_phase_velocity_D_seg_vars_final (nseg_vars, 0.);
                      std::vector<double> D_oil_phase_velocity_D_seg_vars_final    (nseg_vars, 0.);
                      std::vector<double> D_water_phase_velocity_D_seg_vars_final  (nseg_vars, 0.);

                      for (unsigned int id = 0; id < nseg_vars; ++id)
                        {
                          if (current_seg_comp_col_inactive (id))
                            {
                              D_alpha_g_D_seg_vars[id] = 0.0;
                              D_beta_o_D_seg_vars[id] = 0.0;
                              D_alpha_l_D_seg_vars[id] = 0.0;
                              D_alpha_o_D_seg_vars[id] = 0.0;
                              D_alpha_w_D_seg_vars[id] = 0.0;
                              wsncs->D_C0_D_seg_vars[id] = 0.0;
                              wsncs->D_drift_velocity_D_seg_vars[id] = 0.0;
                              wsncs->D_C0_OW_D_seg_vars[id] = 0.0;
                              wsncs->D_drift_velocity_OW_D_seg_vars[id] = 0.0;
                              D_gas_phase_velocity_D_seg_vars_final[id] = 0.0;
                              D_liquid_phase_velocity_D_seg_vars_final[id] = 0.0;
                              D_oil_phase_velocity_D_seg_vars_final[id] = 0.0;
                              D_water_phase_velocity_D_seg_vars_final[id] = 0.0;
                              continue;
                            }

                          D_gas_phase_velocity_D_seg_vars_final[id] =
                              D_gas_phase_velocity_partial_D_seg_vars[id]
                              + D_gas_phase_velocity_D_alpha_g * D_alpha_g_D_seg_vars[id]
                              + D_gas_phase_velocity_D_beta_o  * D_beta_o_D_seg_vars[id];

                          D_liquid_phase_velocity_D_seg_vars_final[id] =
                              D_liquid_phase_velocity_partial_D_seg_vars[id]
                              + D_liquid_phase_velocity_D_alpha_g * D_alpha_g_D_seg_vars[id]
                              + D_liquid_phase_velocity_D_beta_o  * D_beta_o_D_seg_vars[id];

                          D_oil_phase_velocity_D_seg_vars_final[id] =
                              D_oil_phase_velocity_partial_D_seg_vars[id]
                              + D_oil_phase_velocity_D_alpha_g * D_alpha_g_D_seg_vars[id]
                              + D_oil_phase_velocity_D_beta_o  * D_beta_o_D_seg_vars[id];

                          D_water_phase_velocity_D_seg_vars_final[id] =
                              D_water_phase_velocity_partial_D_seg_vars[id]
                              + D_water_phase_velocity_D_alpha_g * D_alpha_g_D_seg_vars[id]
                              + D_water_phase_velocity_D_beta_o  * D_beta_o_D_seg_vars[id];

                          wsncs->D_drift_velocity_D_seg_vars[id] *= df_flow_sign;
                          wsncs->D_drift_velocity_OW_D_seg_vars[id] *= df_flow_sign;
                          D_gas_phase_velocity_D_seg_vars_final[id] *= df_flow_sign;
                          D_liquid_phase_velocity_D_seg_vars_final[id] *= df_flow_sign;
                          D_oil_phase_velocity_D_seg_vars_final[id] *= df_flow_sign;
                          D_water_phase_velocity_D_seg_vars_final[id] *= df_flow_sign;
                        }

                      gas_phase_velocity *= df_flow_sign;
                      liquid_phase_velocity *= df_flow_sign;
                      oil_phase_velocity *= df_flow_sign;
                      water_phase_velocity *= df_flow_sign;
                      drift_velocity *= df_flow_sign;
                      drift_velocity_OW *= df_flow_sign;

                      // ============================================================================
                      // COMPONENT RATES MUST BE COMPUTED ONCE, AFTER FIXED-POINT
                      // ============================================================================

                          std::vector<double> D_rho_avg_D_seg_vars (nseg_vars, 0.0);
                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              const double D_rho_g = (id < 1U + mp.nc && !current_seg_comp_col_inactive (id)) ? element_status->phase_D_rho[id * mp.np + PHASE_GAS]   : 0.0;
                              const double D_rho_o = (id < 1U + mp.nc && !current_seg_comp_col_inactive (id)) ? element_status->phase_D_rho[id * mp.np + PHASE_OIL]   : 0.0;
                              const double D_rho_w = (id < 1U + mp.nc && !current_seg_comp_col_inactive (id)) ? element_status->phase_D_rho[id * mp.np + PHASE_WATER] : 0.0;

                              D_rho_avg_D_seg_vars[id] =
                                  D_rho_g * alpha_g + element_status->phase_rho[PHASE_GAS]   * D_alpha_g_D_seg_vars[id]
                                + D_rho_o * alpha_o + element_status->phase_rho[PHASE_OIL]   * D_alpha_o_D_seg_vars[id]
                                + D_rho_w * alpha_w + element_status->phase_rho[PHASE_WATER] * D_alpha_w_D_seg_vars[id];
                            }

                          double rho_avg = 0.;
                          for (unsigned int ip : range (mp.np))
                            {
                              double phase_holdup = 0.;
                              if (ip == PHASE_GAS)
                                phase_holdup = alpha_g;
                              else if (ip == PHASE_OIL)
                                phase_holdup = alpha_o;
                              else if (ip == PHASE_WATER)
                                phase_holdup = alpha_w;
                              else
                                phase_holdup = 0.;
                              rho_avg += element_status->phase_rho[ip] * phase_holdup;
                            }
                          wsncs->rho_avg_DF = rho_avg;


                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            wsncs->D_rho_avg_D_seg_vars[id] = D_rho_avg_D_seg_vars[id];

                          // Synchronize DF-exported holdups with the state arrays copied earlier
                          // from flash. These arrays are used outside the local DF block as part of
                          // the global Jacobian assembly, so they must represent the SAME state.
                          for (unsigned int id = 0; id < 1U + mp.nc; ++id)
                            {
                              wsncs->phase_D_S[id * mp.np + PHASE_GAS]   = D_alpha_g_D_seg_vars[id];
                              wsncs->phase_D_S[id * mp.np + PHASE_OIL]   = D_alpha_o_D_seg_vars[id];
                              wsncs->phase_D_S[id * mp.np + PHASE_WATER] = D_alpha_w_D_seg_vars[id];

                              wsncs->avg_D_xi[id] =
                                    element_status->phase_xi[PHASE_GAS]   * D_alpha_g_D_seg_vars[id]
                                  + element_status->phase_xi[PHASE_OIL]   * D_alpha_o_D_seg_vars[id]
                                  + element_status->phase_xi[PHASE_WATER] * D_alpha_w_D_seg_vars[id]
                                  + alpha_g * element_status->phase_D_xi[id * mp.np + PHASE_GAS]
                                  + alpha_o * element_status->phase_D_xi[id * mp.np + PHASE_OIL]
                                  + alpha_w * element_status->phase_D_xi[id * mp.np + PHASE_WATER];

                              wsncs->avg_D_rho[id] = D_rho_avg_D_seg_vars[id];
                            }

                          PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                 "Seg {} var {}: alpha_water = {}, alpha_oil = {}, alpha_gas = {}, alpha_liq = {}, \n"
                                 "water_velo = {}, oil_velo = {}, gas_velo = {}, liquid_velo = {}, beta_o = {}, \n"
                                 "D_alpha_w_D_seg_vars[0] = {}, D_alpha_w_D_seg_vars[1] = {}, D_alpha_w_D_seg_vars[2] = {},"
                                 "D_alpha_w_D_seg_vars[3] = {}, D_alpha_w_D_seg_vars[4] = {}, inner_error = {}, rho_avg = {} \n",
                                 seg.wsn->wsn_index, fd_var_id, alpha_w, alpha_o, alpha_g, alpha_l,
                                 water_phase_velocity, oil_phase_velocity, gas_phase_velocity, liquid_phase_velocity, beta_o,
                                 D_alpha_w_D_seg_vars[0], D_alpha_w_D_seg_vars[1], D_alpha_w_D_seg_vars[2], D_alpha_w_D_seg_vars[3],
                                 D_alpha_w_D_seg_vars[4], error, wsncs->rho_avg_DF);

                      PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                           "Seg {} smallJ: detJ={}, J11={}, J12={}, J21={}, J22={}, fallback={}, flow_dir={}, comp_cols_inactive={}, state_upwinded={}\n",
                           seg.wsn->wsn_index, detJ, J11, J12, J21, J22,
                           used_small_j_fallback ? 1 : 0,
                           static_cast<int> (wsncs->wsn_flow_dir),
                           current_seg_component_columns_inactive ? 1 : 0,
                           use_upwind_state_inside_df ? 1 : 0);

                      for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                        {
                          wsncs->wsn_component_rate[ic] = 0.;
                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id] = 0.;
                        }


                      for (unsigned int ip = 0; ip < mp.np; ++ip)
                        {
                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            if (ip == PHASE_GAS)
                              wsncs->D_phase_holdup_D_seg_vars[mp.np * id + ip] = D_alpha_g_D_seg_vars[id];
                            else if (ip == PHASE_OIL)
                              wsncs->D_phase_holdup_D_seg_vars[mp.np * id + ip] = D_alpha_o_D_seg_vars[id];
                            else if (ip == PHASE_WATER)
                              wsncs->D_phase_holdup_D_seg_vars[mp.np * id + ip] = D_alpha_w_D_seg_vars[id];
                        }

                      const double area_day = seg.wsn->pipe_props.area * internal_const::DAYS_TO_SEC (); // same units as phase velocities in this block
                      std::vector<double> D_phase_molar_rate_D_seg_var (nseg_vars * mp.np, 0.0);
                      for (unsigned int ip = 0; ip < mp.np; ++ip)
                        {
                          double phase_velocity = 0.;
                          const std::vector<double> *D_phase_velocity_ptr = nullptr;
                          double phase_holdup = 0.;
                          const std::vector<double> *D_phase_holdup_ptr = nullptr;

                          if (ip == PHASE_GAS)
                            {
                              phase_velocity = gas_phase_velocity;
                              D_phase_velocity_ptr = &D_gas_phase_velocity_D_seg_vars_final;
                              phase_holdup = alpha_g;
                              D_phase_holdup_ptr = &D_alpha_g_D_seg_vars;
                            }
                          else if (ip == PHASE_OIL)
                            {
                              phase_velocity = oil_phase_velocity;
                              D_phase_velocity_ptr = &D_oil_phase_velocity_D_seg_vars_final;
                              phase_holdup = alpha_o;
                              D_phase_holdup_ptr = &D_alpha_o_D_seg_vars;
                            }
                          else // PHASE_WATER
                            {
                              phase_velocity = water_phase_velocity;
                              D_phase_velocity_ptr = &D_water_phase_velocity_D_seg_vars_final;
                              phase_holdup = alpha_w;
                              D_phase_holdup_ptr = &D_alpha_w_D_seg_vars;
                            }

                          const double phase_superficial_flux = phase_velocity * phase_holdup * area_day;
                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              if (current_seg_comp_col_inactive (id))
                                {
                                  D_phase_molar_rate_D_seg_var[id * mp.np + ip] = 0.0;
                                  continue;
                                }

                              D_phase_molar_rate_D_seg_var[id * mp.np + ip] =
                                  ((*D_phase_holdup_ptr)[id] * phase_velocity
                                   + phase_holdup * (*D_phase_velocity_ptr)[id])
                                  * area_day;
                            }

                          for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                            {
                              wsncs->wsn_component_rate[ic] +=
                                  phase_superficial_flux
                                  * element_status->component_phase_x[ic * mp.np + ip]
                                  * element_status->phase_xi[ip];

                              for (unsigned int id = 0; id < nseg_vars; ++id)
                                {
                                  if (id < 1U + mp.nc)
                                    {
                                      if (current_seg_comp_col_inactive (id))
                                        continue;

                                      wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id] +=
                                          D_phase_molar_rate_D_seg_var[id * mp.np + ip]
                                              * element_status->component_phase_x[ic * mp.np + ip]
                                              * element_status->phase_xi[ip]
                                          + phase_superficial_flux
                                              * (element_status->component_phase_D_x[(id * mp.nc + ic) * mp.np + ip]
                                                 * element_status->phase_xi[ip]
                                                 + element_status->component_phase_x[ic * mp.np + ip]
                                                   * element_status->phase_D_xi[id * mp.np + ip]);
                                    }
                                  else
                                    {
                                      wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id] +=
                                          D_phase_molar_rate_D_seg_var[id * mp.np + ip]
                                          * element_status->component_phase_x[ic * mp.np + ip]
                                          * element_status->phase_xi[ip];
                                    }
                                }
                            }
                        }

                        if (seg.wsn->wsn_index != TOP_SEG_INDEX)
                          {
                            wsncs->wsn_mixture_mass_rate = 0.0;
                            wsncs->wsn_mmw = 0.0;
                            for (unsigned int id = 0; id < nseg_vars; ++id)
                              wsncs->D_mixture_mass_rate_D_seg_vars[id] = 0.0;

                            for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                              {
                                wsncs->wsn_mixture_mass_rate +=
                                    wsncs->wsn_component_rate[ic] * component_molar_weights[ic];

                                for (unsigned int id = 0; id < nseg_vars; ++id)
                                  {
                                    wsncs->D_mixture_mass_rate_D_seg_vars[id] +=
                                        wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id]
                                        * component_molar_weights[ic];
                                  }
                                wsncs->wsn_mmw += component_z_for_flow[ic] * component_molar_weights[ic];
                              }
                          }

        // COMPARISON OF ANALYTICAL VS NUMERICAL DERIVATIVES
        // ============================================================================

        auto rel_err_dbg =
          [] (double ana, double num) -> double
          {
            double denom = std::max (1.e-12, std::max (fabs (ana), fabs (num)));
            return fabs (ana - num) / denom;
          };

        auto build_phase_mask_dbg = [&] (const fully_implicit_element_status *es) -> unsigned int
          {
            unsigned int mask = 0U;
            if (!es)
              return mask;
            for (unsigned int ip = 0; ip < mp.np; ++ip)
              {
                if (std::isfinite (es->phase_S[ip]) && fabs (es->phase_S[ip]) > 1.0e-12)
                  mask |= (1U << ip);
              }
            return mask;
          };

        const unsigned int prev_flash_phase_mask = build_phase_mask_dbg (element_status_prev_dbg_ptr);
        const unsigned int next_flash_phase_mask = build_phase_mask_dbg (element_status_next_dbg_ptr);

        const bool same_active_set = (prev_active_flags == next_active_flags);
        const bool inner_converged =
            (fabs (prev_R_g) < 1.e-4 && fabs (prev_R_o) < 1.e-4 &&
             fabs (next_R_g) < 1.e-4 && fabs (next_R_o) < 1.e-4);
        const bool same_flash_phase_set = (prev_flash_phase_mask == next_flash_phase_mask);

        if (!same_flash_phase_set)
          {
            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: FD skipped because phase mask changed prev_mask={}, next_mask={}\n",
                 seg.wsn->wsn_index, fd_var_id, prev_flash_phase_mask, next_flash_phase_mask);
          }

        if (same_active_set && inner_converged && same_flash_phase_set && seg.wsn->wsn_index != TOP_SEG_INDEX)
          {
            double D_sigma_gl_total_ana =
                D_gas_liq_interfacial_tension_partial_D_seg_vars[fd_var_id]
                + D_gas_liq_interfacial_tension_D_beta_o * D_beta_o_D_seg_vars[fd_var_id];

            double D_liq_density_total_ana =
                D_liq_density_partial_D_seg_vars[fd_var_id]
                + D_liq_density_D_beta_o * D_beta_o_D_seg_vars[fd_var_id];

            double D_bubble_total_ana =
                D_bubble_rise_velocity_partial_D_seg_vars[fd_var_id]
                + D_bubble_rise_velocity_D_beta_o * D_beta_o_D_seg_vars[fd_var_id];

            double D_diametr_total_ana =
                D_diametr_dimless_partial_D_seg_vars[fd_var_id]
                + D_diametr_dimless_D_beta_o * D_beta_o_D_seg_vars[fd_var_id];

            double D_Kut_total_ana =
                D_Kut_number_partial_D_seg_vars[fd_var_id]
                + D_Kut_number_D_beta_o * D_beta_o_D_seg_vars[fd_var_id];

            double D_flooding_total_ana =
                D_flooding_velocity_partial_D_seg_vars[fd_var_id]
                + D_flooding_velocity_D_beta_o * D_beta_o_D_seg_vars[fd_var_id];

            double D_ksi_total_ana =
                D_ksi_partial_D_seg_vars[fd_var_id]
                + D_ksi_D_alpha_g * D_alpha_g_D_seg_vars[fd_var_id]
                + D_ksi_D_beta_o  * D_beta_o_D_seg_vars[fd_var_id];

            double D_eta_total_ana =
                D_eta_partial_D_seg_vars[fd_var_id]
                + D_eta_D_alpha_g * D_alpha_g_D_seg_vars[fd_var_id]
                + D_eta_D_beta_o  * D_beta_o_D_seg_vars[fd_var_id];

            double D_Kg_total_ana =
                D_K_g_partial_D_seg_vars[fd_var_id]
                + D_K_g_D_alpha_g * D_alpha_g_D_seg_vars[fd_var_id]
                + D_K_g_D_beta_o  * D_beta_o_D_seg_vars[fd_var_id];

            double D_sigma_ow_total_ana =
                D_wat_oil_interfacial_tension_partial_D_seg_vars[fd_var_id]
                + D_wat_oil_interfacial_tension_D_beta_o * D_beta_o_D_seg_vars[fd_var_id];

            double D_bubble_ow_total_ana =
                D_bubble_rise_velocity_OW_partial_D_seg_vars[fd_var_id]
                + D_bubble_rise_velocity_OW_D_beta_o * D_beta_o_D_seg_vars[fd_var_id];

            double D_C0_ow_total_ana =
                D_wsn_C0_OW_D_beta_o * D_beta_o_D_seg_vars[fd_var_id];

            double D_Vd_ow_total_ana =
                D_drift_velocity_OW_partial_D_seg_vars[fd_var_id]
                + D_drift_velocity_OW_D_beta_o * D_beta_o_D_seg_vars[fd_var_id];

            // -------------------- mixed check: analytic partials + numerical holdup derivatives
            double D_C0_mixed_num_h =
                D_C0_partial_D_seg_vars[fd_var_id]
                + D_C0_D_alpha_g * alpha_g_num_derivative
                + D_C0_D_beta_o  * beta_o_num_derivative;

            double D_Vd_mixed_num_h =
                D_drift_velocity_partial_D_seg_vars[fd_var_id]
                + D_drift_velocity_D_alpha_g * alpha_g_num_derivative
                + D_drift_velocity_D_beta_o  * beta_o_num_derivative;

            double D_Vg_mixed_num_h =
                D_gas_phase_velocity_partial_D_seg_vars[fd_var_id]
                + D_gas_phase_velocity_D_alpha_g * alpha_g_num_derivative
                + D_gas_phase_velocity_D_beta_o  * beta_o_num_derivative;

            double D_Vl_mixed_num_h =
                D_liquid_phase_velocity_partial_D_seg_vars[fd_var_id]
                + D_liquid_phase_velocity_D_alpha_g * alpha_g_num_derivative
                + D_liquid_phase_velocity_D_beta_o  * beta_o_num_derivative;

            double Rg_x_dbg =
                alpha_g * D_gas_phase_velocity_partial_D_seg_vars[fd_var_id]
                - D_gas_superficial_velocity_input_D_seg_vars[fd_var_id];

            double Ro_x_dbg =
                alpha_o * D_oil_phase_velocity_partial_D_seg_vars[fd_var_id]
                - D_oil_superficial_velocity_input_D_seg_vars[fd_var_id];

            const double alpha_l_safe_dbg = df_safe_nonzero (alpha_l);
            const double D_beta_o_D_alpha_g_dbg = alpha_o / (alpha_l_safe_dbg * alpha_l_safe_dbg);
            const double D_beta_o_D_alpha_o_dbg = 1.0 / alpha_l_safe_dbg;

            const double J11_dbg =
                gas_phase_velocity
                + alpha_g * (D_gas_phase_velocity_D_alpha_g + D_gas_phase_velocity_D_beta_o * D_beta_o_D_alpha_g_dbg);

            const double J12_dbg =
                alpha_g * D_gas_phase_velocity_D_beta_o * D_beta_o_D_alpha_o_dbg;

            const double J21_dbg =
                alpha_o * (D_oil_phase_velocity_D_alpha_g + D_oil_phase_velocity_D_beta_o * D_beta_o_D_alpha_g_dbg);

            const double J22_dbg =
                oil_phase_velocity + alpha_o * D_oil_phase_velocity_D_beta_o * D_beta_o_D_alpha_o_dbg;

            double defect_Rg_ana =
                J11_dbg * D_alpha_g_D_seg_vars[fd_var_id]
                + J12_dbg * D_alpha_o_D_seg_vars[fd_var_id]
                + Rg_x_dbg;

            double defect_Ro_ana =
                J21_dbg * D_alpha_g_D_seg_vars[fd_var_id]
                + J22_dbg * D_alpha_o_D_seg_vars[fd_var_id]
                + Ro_x_dbg;

            double defect_Rg_num =
                J11_dbg * alpha_g_num_derivative
                + J12_dbg * alpha_o_num_derivative
                + Rg_x_dbg;

            double defect_Ro_num =
                J21_dbg * alpha_g_num_derivative
                + J22_dbg * alpha_o_num_derivative
                + Ro_x_dbg;

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: alpha_g ana={}, num={}, relerr={}, alpha_o ana={}, num={}, relerr={}, beta_o ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_alpha_g_D_seg_vars[fd_var_id], alpha_g_num_derivative,
                 rel_err_dbg (D_alpha_g_D_seg_vars[fd_var_id], alpha_g_num_derivative),
                 D_alpha_o_D_seg_vars[fd_var_id], alpha_o_num_derivative,
                 rel_err_dbg (D_alpha_o_D_seg_vars[fd_var_id], alpha_o_num_derivative),
                 D_beta_o_D_seg_vars[fd_var_id], beta_o_num_derivative,
                 rel_err_dbg (D_beta_o_D_seg_vars[fd_var_id], beta_o_num_derivative));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: IFT defect ana: Rg={}, Ro={} ; num-h defect: Rg={}, Ro={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 defect_Rg_ana, defect_Ro_ana,
                 defect_Rg_num, defect_Ro_num);

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: sigma_gl ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_sigma_gl_total_ana, gas_liq_interfacial_tension_numerical,
                 rel_err_dbg (D_sigma_gl_total_ana, gas_liq_interfacial_tension_numerical));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: rho_l ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_liq_density_total_ana, liquid_density_numerical,
                 rel_err_dbg (D_liq_density_total_ana, liquid_density_numerical));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: rho_avg_DF ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_rho_avg_D_seg_vars[fd_var_id], rho_avg_numerical,
                 rel_err_dbg (D_rho_avg_D_seg_vars[fd_var_id], rho_avg_numerical));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: Vc ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_bubble_total_ana, bubble_rise_velocity_numerical,
                 rel_err_dbg (D_bubble_total_ana, bubble_rise_velocity_numerical));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: Dhat ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_diametr_total_ana, diametr_dimless_numerical,
                 rel_err_dbg (D_diametr_total_ana, diametr_dimless_numerical));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: Ku ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_Kut_total_ana, Kut_number_numerical,
                 rel_err_dbg (D_Kut_total_ana, Kut_number_numerical));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: Vsgf ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_flooding_total_ana, flooding_velocity_numerical,
                 rel_err_dbg (D_flooding_total_ana, flooding_velocity_numerical));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: ksi ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_ksi_total_ana, ksi_numerical,
                 rel_err_dbg (D_ksi_total_ana, ksi_numerical));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: eta ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_eta_total_ana, eta_numerical,
                 rel_err_dbg (D_eta_total_ana, eta_numerical));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: Kg ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_Kg_total_ana, K_g_numerical,
                 rel_err_dbg (D_Kg_total_ana, K_g_numerical));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: C0 full ana={}, full num={}, mixed(num holdup)={}, relerr(full)={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 wsncs->D_C0_D_seg_vars[fd_var_id], C_0_num_derivative, D_C0_mixed_num_h,
                 rel_err_dbg (wsncs->D_C0_D_seg_vars[fd_var_id], C_0_num_derivative));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: Vd full ana={}, full num={}, mixed(num holdup)={}, relerr(full)={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 wsncs->D_drift_velocity_D_seg_vars[fd_var_id], Drift_velocity_num_derivative, D_Vd_mixed_num_h,
                 rel_err_dbg (wsncs->D_drift_velocity_D_seg_vars[fd_var_id], Drift_velocity_num_derivative));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: Vg full ana={}, full num={}, mixed(num holdup)={}, relerr(full)={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_gas_phase_velocity_D_seg_vars_final[fd_var_id], gas_phase_velocity_numerical, D_Vg_mixed_num_h,
                 rel_err_dbg (D_gas_phase_velocity_D_seg_vars_final[fd_var_id], gas_phase_velocity_numerical));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: Vl full ana={}, full num={}, mixed(num holdup)={}, relerr(full)={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_liquid_phase_velocity_D_seg_vars_final[fd_var_id], liquid_phase_velocity_numerical, D_Vl_mixed_num_h,
                 rel_err_dbg (D_liquid_phase_velocity_D_seg_vars_final[fd_var_id], liquid_phase_velocity_numerical));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: sigma_ow ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_sigma_ow_total_ana, oil_water_interfacial_tension_num,
                 rel_err_dbg (D_sigma_ow_total_ana, oil_water_interfacial_tension_num));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: Vc_ow ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_bubble_ow_total_ana, bubble_rise_velocity_OW_num,
                 rel_err_dbg (D_bubble_ow_total_ana, bubble_rise_velocity_OW_num));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: C0_ow ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_C0_ow_total_ana, C_OW_num,
                 rel_err_dbg (D_C0_ow_total_ana, C_OW_num));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: Vd_ow ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_Vd_ow_total_ana, V_d_OW_num,
                 rel_err_dbg (D_Vd_ow_total_ana, V_d_OW_num));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: Vo ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_oil_phase_velocity_D_seg_vars_final[fd_var_id], oil_velocity_num,
                 rel_err_dbg (D_oil_phase_velocity_D_seg_vars_final[fd_var_id], oil_velocity_num));

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: Vw ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_water_phase_velocity_D_seg_vars_final[fd_var_id], water_velocity_num,
                 rel_err_dbg (D_water_phase_velocity_D_seg_vars_final[fd_var_id], water_velocity_num));

            // component rates
            for (auto ic = mp.nc0; ic < mp.nc; ++ic)
              {
                double D_qc_num =
                    (next_component_rate_dbg[ic] - prev_component_rate_dbg[ic]) / (2.0 * fd_eps);

                double D_qc_ana =
                    wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + fd_var_id];

                PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                     "Seg {} var {} ic {}: qc ana={}, num={}, relerr={}\n",
                     seg.wsn->wsn_index, fd_var_id, ic,
                     D_qc_ana, D_qc_num, rel_err_dbg (D_qc_ana, D_qc_num));

                // Numerical derivative is printed only for comparison.
                // Do NOT overwrite the analytical Jacobian entry in the live solver state.
                //wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + fd_var_id] = D_qc_num;
                }

            // -----------------------------------------------------------------
            // ALL-VARIABLES FD DEBUG SUMMARY
            // -----------------------------------------------------------------
            const df_fd_wsncs_backup_t wsncs_dbg_final_state (wsncs, mp);

            auto fd_var_name_dbg = [&] (unsigned int vid) -> std::string
              {
                if (vid == 0)
                  return "p";
                if (vid < 1U + mp.nc)
                  return std::string ("N[") + std::to_string (vid - 1) + "]";
                return "q_tot";
              };

            auto apply_fd_local = [&] (fully_implicit_element_status *es, double &qtot_work, unsigned int local_vid, double delta)
              {
                if (local_vid == 0)
                  es->p += delta;
                else if (local_vid < 1U + mp.nc)
                  {
                    es->component_N[local_vid - 1] += delta;
                    es->component_N_tot += delta;
                  }
                else
                  qtot_work += delta;
              };

            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} ALLVARS checker seed: incoming_prev_df_state (not converged_final_state), flow_dir={}, comp_cols_inactive={}, state_upwinded={}\n",
                 seg.wsn->wsn_index, static_cast<int> (wsncs->wsn_flow_dir),
                 current_seg_component_columns_inactive ? 1 : 0,
                 use_upwind_state_inside_df ? 1 : 0);

            const unsigned int phase_mask_base_all = build_phase_mask_dbg (element_status);
            const double rho_base_all = wsncs->rho_avg_DF;
            std::vector<double> qc_base_all (mp.nc, 0.0);
            for (unsigned int ic = 0; ic < mp.nc; ++ic)
              qc_base_all[ic] = wsncs->wsn_component_rate[ic];

            std::vector<double> D_alpha_g_patched = D_alpha_g_D_seg_vars;
            std::vector<double> D_beta_o_patched = D_beta_o_D_seg_vars;
            std::vector<double> D_alpha_o_patched = D_alpha_o_D_seg_vars;
            std::vector<double> D_alpha_w_patched = D_alpha_w_D_seg_vars;
            std::vector<double> D_alpha_l_patched = D_alpha_l_D_seg_vars;
            std::vector<double> D_rho_avg_patched = D_rho_avg_D_seg_vars;
            std::vector<double> D_qc_patched ((1U + mp.nc + 1U) * mp.nc, 0.0);
            for (auto ic = mp.nc0; ic < mp.nc; ++ic)
              for (unsigned int id = 0; id < nseg_vars; ++id)
                D_qc_patched[(1U + mp.nc + 1U) * ic + id] =
                    wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id];

            bool applied_phase_boundary_override = false;

            for (unsigned int dbg_vid = 0; dbg_vid < nseg_vars_fd; ++dbg_vid)
              {
                double eps_dbg = 1.e-3;
                if (dbg_vid == 0)
                  eps_dbg = 1.e-3;
                else if (dbg_vid < 1U + mp.nc)
                  eps_dbg = 1.e-4 * std::max (1.0, fabs (element_status->component_N[dbg_vid - 1]));
                else
                  eps_dbg = 1.e-6 * std::max (1.0, fabs (q_tot_buf));

                fully_implicit_element_status es_prev_all (*element_status);
                fully_implicit_element_status *es_prev_all_ptr = &es_prev_all;
                copy_segment_params_to_element_status (seg, es_prev_all_ptr);
                wsncs_backup.restore (wsncs, mp);
                double qtot_prev_all = q_tot_buf;
                apply_fd_local (es_prev_all_ptr, qtot_prev_all, dbg_vid, -eps_dbg);
                wsncs->wsn_mixture_molar_rate = qtot_prev_all;

                double pa_vm=0., pa_sig=0., pa_vc=0., pa_rhol=0., pa_dhat=0., pa_ku=0., pa_vsgf=0., pa_ksi=0., pa_eta=0., pa_vg=0., pa_vl=0., pa_kg=0., pa_avgxi=0., pa_xi=0., pa_so=0., pa_sw=0., pa_sow=0., pa_vcow=0., pa_cow=0., pa_vdow=0., pa_vo=0., pa_vw=0.;
                double pa_ag=0., pa_bo=0., pa_ao=0., pa_aw=0., pa_rg=0., pa_ro=0.; unsigned int pa_flags=0u; int pa_it=0;
                test_function (rep, seg, wsncs, es_prev_all_ptr, mp, new_status, i_meshblock, current_therm_comp_input_props, itd,
                               pa_vm, pa_sig, pa_vc, pa_rhol, pa_dhat, pa_ku, pa_vsgf, pa_ksi, pa_eta, pa_vg, pa_vl, pa_kg, pa_avgxi, pa_xi, pa_so, pa_sw,
                               pa_sow, pa_vcow, pa_cow, pa_vdow, pa_vo, pa_vw, pa_ag, pa_bo, pa_ao, pa_aw, pa_rg, pa_ro, pa_flags, pa_it);
                std::vector<double> qc_prev_all (mp.nc, 0.0);
                for (unsigned int ic = 0; ic < mp.nc; ++ic) qc_prev_all[ic] = wsncs->wsn_component_rate[ic];
                const double rho_prev_all = wsncs->rho_avg_DF;
                const unsigned int phase_mask_prev_all = build_phase_mask_dbg (es_prev_all_ptr);
                wsncs_dbg_final_state.restore (wsncs, mp);
                wsncs->wsn_mixture_molar_rate = q_tot_buf;

                fully_implicit_element_status es_next_all (*element_status);
                fully_implicit_element_status *es_next_all_ptr = &es_next_all;
                copy_segment_params_to_element_status (seg, es_next_all_ptr);
                wsncs_backup.restore (wsncs, mp);
                double qtot_next_all = q_tot_buf;
                apply_fd_local (es_next_all_ptr, qtot_next_all, dbg_vid, +eps_dbg);
                wsncs->wsn_mixture_molar_rate = qtot_next_all;

                double na_vm=0., na_sig=0., na_vc=0., na_rhol=0., na_dhat=0., na_ku=0., na_vsgf=0., na_ksi=0., na_eta=0., na_vg=0., na_vl=0., na_kg=0., na_avgxi=0., na_xi=0., na_so=0., na_sw=0., na_sow=0., na_vcow=0., na_cow=0., na_vdow=0., na_vo=0., na_vw=0.;
                double na_ag=0., na_bo=0., na_ao=0., na_aw=0., na_rg=0., na_ro=0.; unsigned int na_flags=0u; int na_it=0;
                test_function (rep, seg, wsncs, es_next_all_ptr, mp, new_status, i_meshblock, current_therm_comp_input_props, itd,
                               na_vm, na_sig, na_vc, na_rhol, na_dhat, na_ku, na_vsgf, na_ksi, na_eta, na_vg, na_vl, na_kg, na_avgxi, na_xi, na_so, na_sw,
                               na_sow, na_vcow, na_cow, na_vdow, na_vo, na_vw, na_ag, na_bo, na_ao, na_aw, na_rg, na_ro, na_flags, na_it);
                std::vector<double> qc_next_all (mp.nc, 0.0);
                for (unsigned int ic = 0; ic < mp.nc; ++ic) qc_next_all[ic] = wsncs->wsn_component_rate[ic];
                const double rho_next_all = wsncs->rho_avg_DF;
                const unsigned int phase_mask_next_all = build_phase_mask_dbg (es_next_all_ptr);

                wsncs_dbg_final_state.restore (wsncs, mp);
                wsncs->wsn_mixture_molar_rate = q_tot_buf;

                const double d_alpha_num = (na_ag - pa_ag) / (2.0 * eps_dbg);
                const double d_beta_num = (na_bo - pa_bo) / (2.0 * eps_dbg);
                const double d_rho_num = (rho_next_all - rho_prev_all) / (2.0 * eps_dbg);
                const double d_c0_num = (wsncs->wsn_C_0 - wsncs->wsn_C_0) ;
                (void) d_c0_num;

                if (phase_mask_prev_all != phase_mask_next_all)
                  {
                    bool used_one_sided_override = false;
                    const bool prev_matches_base = (phase_mask_prev_all == phase_mask_base_all);
                    const bool next_matches_base = (phase_mask_next_all == phase_mask_base_all);

                    if (prev_matches_base || next_matches_base)
                      {
                        const double inv_eps_dbg = 1.0 / eps_dbg;

                        if (prev_matches_base && !next_matches_base)
                          {
                            D_alpha_g_patched[dbg_vid] = (alpha_g - pa_ag) * inv_eps_dbg;
                            D_beta_o_patched[dbg_vid] = (beta_o - pa_bo) * inv_eps_dbg;
                            D_rho_avg_patched[dbg_vid] = (rho_base_all - rho_prev_all) * inv_eps_dbg;
                            for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                              {
                                D_qc_patched[(1U + mp.nc + 1U) * ic + dbg_vid] =
                                    (qc_base_all[ic] - qc_prev_all[ic]) * inv_eps_dbg;
                              }
                            used_one_sided_override = true;
                            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                 "Seg {} ALLVARS {} crossed phase mask (prev={}, next={}); using backward one-sided FD override for production Jacobian\n",
                                 seg.wsn->wsn_index, fd_var_name_dbg (dbg_vid), phase_mask_prev_all, phase_mask_next_all);
                          }
                        else if (next_matches_base && !prev_matches_base)
                          {
                            D_alpha_g_patched[dbg_vid] = (na_ag - alpha_g) * inv_eps_dbg;
                            D_beta_o_patched[dbg_vid] = (na_bo - beta_o) * inv_eps_dbg;
                            D_rho_avg_patched[dbg_vid] = (rho_next_all - rho_base_all) * inv_eps_dbg;
                            for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                              {
                                D_qc_patched[(1U + mp.nc + 1U) * ic + dbg_vid] =
                                    (qc_next_all[ic] - qc_base_all[ic]) * inv_eps_dbg;
                              }
                            used_one_sided_override = true;
                            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                 "Seg {} ALLVARS {} crossed phase mask (prev={}, next={}); using forward one-sided FD override for production Jacobian\n",
                                 seg.wsn->wsn_index, fd_var_name_dbg (dbg_vid), phase_mask_prev_all, phase_mask_next_all);
                          }

                        if (used_one_sided_override)
                          {
                            const double D_alpha_l_one_sided = -D_alpha_g_patched[dbg_vid];
                            D_alpha_l_patched[dbg_vid] = D_alpha_l_one_sided;

                            if (fabs (alpha_l) > tnm::min_compare)
                              {
                                D_alpha_o_patched[dbg_vid] =
                                    D_alpha_l_one_sided * beta_o + alpha_l * D_beta_o_patched[dbg_vid];
                                D_alpha_w_patched[dbg_vid] =
                                    D_alpha_l_one_sided * (1.0 - beta_o) - alpha_l * D_beta_o_patched[dbg_vid];
                              }
                            else
                              {
                                D_alpha_o_patched[dbg_vid] = 0.0;
                                D_alpha_w_patched[dbg_vid] = 0.0;
                              }

                            applied_phase_boundary_override = true;
                            continue;
                          }
                      }

                    PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                         "Seg {} ALLVARS {} skipped because phase mask changed prev_mask={}, next_mask={}\n",
                         seg.wsn->wsn_index, fd_var_name_dbg (dbg_vid), phase_mask_prev_all, phase_mask_next_all);
                    continue;
                  }

                const double d_alpha_o_num = (na_ao - pa_ao) / (2.0 * eps_dbg);

                D_alpha_g_patched[dbg_vid] = d_alpha_num;
                D_alpha_o_patched[dbg_vid] = d_alpha_o_num;
                D_beta_o_patched[dbg_vid] = d_beta_num;
                D_alpha_l_patched[dbg_vid] = -d_alpha_num;
                D_alpha_w_patched[dbg_vid] = -d_alpha_num - d_alpha_o_num;
                D_rho_avg_patched[dbg_vid] = d_rho_num;
                for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                  {
                    const double d_qc_num_all = (qc_next_all[ic] - qc_prev_all[ic]) / (2.0 * eps_dbg);
                    D_qc_patched[(1U + mp.nc + 1U) * ic + dbg_vid] = d_qc_num_all;
                  }

                PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                     "Seg {} ALLVARS {}: alpha_g a={} n={} r={} | alpha_o a={} n={} r={} | beta_o a={} n={} r={} | rho_avg a={} n={} r={}\n",
                     seg.wsn->wsn_index, fd_var_name_dbg (dbg_vid),
                     D_alpha_g_D_seg_vars[dbg_vid], d_alpha_num, rel_err_dbg (D_alpha_g_D_seg_vars[dbg_vid], d_alpha_num),
                     D_alpha_o_D_seg_vars[dbg_vid], d_alpha_o_num, rel_err_dbg (D_alpha_o_D_seg_vars[dbg_vid], d_alpha_o_num),
                     D_beta_o_D_seg_vars[dbg_vid], d_beta_num, rel_err_dbg (D_beta_o_D_seg_vars[dbg_vid], d_beta_num),
                     D_rho_avg_D_seg_vars[dbg_vid], d_rho_num, rel_err_dbg (D_rho_avg_D_seg_vars[dbg_vid], d_rho_num));

                for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                  {
                    const double d_qc_num_all = (qc_next_all[ic] - qc_prev_all[ic]) / (2.0 * eps_dbg);
                    const double d_qc_ana_all = wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + dbg_vid];
                    PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                         "Seg {} ALLVARS {} ic {}: qc a={} n={} r={}\n",
                         seg.wsn->wsn_index, fd_var_name_dbg (dbg_vid), ic,
                         d_qc_ana_all, d_qc_num_all, rel_err_dbg (d_qc_ana_all, d_qc_num_all));
                  }
              }

            if (applied_phase_boundary_override)
              {
                PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                     "Seg {} computed phase-boundary one-sided FD diagnostics but did NOT export them to production Jacobian\n",
                     seg.wsn->wsn_index);
                wsncs_dbg_final_state.restore (wsncs, mp);
                wsncs->wsn_mixture_molar_rate = q_tot_buf;
              }
          }

                          // Store the converged production DF state for the next
                          // Big-Newton evaluation. This is deliberately after the FD
                          // diagnostics, so all numerical derivative probes for the
                          // current Jacobian used the same incoming lagged snapshot.
                          if (local_df_converged)
                            {
                              df_lagged_phase_input_state_t df_lagged_output_state;
                              df_lagged_output_state.valid = true;
                              df_lagged_output_state.alpha_g = alpha_g;
                              df_lagged_output_state.alpha_o = alpha_o;
                              df_lagged_output_state.alpha_w = alpha_w;
                              df_lagged_output_state.jsg_raw = alpha_g * gas_phase_velocity;
                              df_lagged_output_state.jso_raw = alpha_o * oil_phase_velocity;
                              df_lagged_output_state.jsw_raw = alpha_w * water_phase_velocity;
                              df_lagged_output_state.jmix_raw = df_lagged_output_state.jsg_raw
                                                               + df_lagged_output_state.jso_raw
                                                               + df_lagged_output_state.jsw_raw;
                              df_lagged_output_state.rho_avg = rho_avg;
                              df_lagged_output_state.qtot_raw = q_tot_buf;
                              df_store_lagged_state (df_lagged_output_state);

                              PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                   "MERGEN_DF_LAGGED_STORE: Well {}: Seg_idx = {}, generation_next = {}, alpha_g = {}, alpha_o = {}, alpha_w = {}, jsg_raw = {}, jso_raw = {}, jsw_raw = {}, jmix_raw = {}\n",
                                   wcb_wis->get_well_name (), seg.wsn->wsn_index,
                                   df_lagged_input_state_valid ? (df_lagged_input_state.generation + 1UL) : 1UL,
                                   df_lagged_output_state.alpha_g, df_lagged_output_state.alpha_o, df_lagged_output_state.alpha_w,
                                   df_lagged_output_state.jsg_raw, df_lagged_output_state.jso_raw,
                                   df_lagged_output_state.jsw_raw, df_lagged_output_state.jmix_raw);
                            }
                          }
        }

      if (seg.wsn->wsn_index == TOP_SEG_INDEX)
        {
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
                    int &dbg_inner_it)
{
  (void) rep;
  (void) new_status;
  (void) i_meshblock;
  (void) current_therm_comp_input_props;
  (void) itd;

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

  // Lagged DF input mode for the finite-difference/debug evaluator.
  // The previous DF holdups are captured before fill_wsncs_from_element_status()
  // overwrites wsncs->phase_S with the current flash values.
  const bool use_upwind_state_inside_df = false;
  if (use_upwind_state_inside_df
      && wsncs->wsn_flow_dir == segment_flow_direction_t::from_parent_to_child)
    {
      set_flow_direction_dependent_segment_params_to_element_status (seg, element_status, new_status);
    }

  if (element_status->component_N_tot > tnm::min_compare)
    {
      if (auto err = run_flash <true> (rep,
                                       i_meshblock,
                                       element_status,
                                       current_therm_comp_input_props,
                                       itd);
          err != segments_solver_err_t::none)
        {
          return;
        }
    }

  double alpha_g_lagged_df_input = 0.0;
  double alpha_o_lagged_df_input = 0.0;
  double alpha_w_lagged_df_input = 0.0;
  bool prev_df_holdup_input_valid = false;

  // test_function() is used by finite-difference diagnostics and must evaluate the
  // same lagged holdup split as the production value/Jacobian path.  The production
  // code seeds wsncs before df_fd_wsncs_backup_t is captured; restore() then brings
  // these values back before every +/- eps call.  jmix is recomputed from the
  // current perturbed q_tot.
  {
    const double ag_prev = wsncs->phase_S[PHASE_GAS];
    const double ao_prev = wsncs->phase_S[PHASE_OIL];
    const double aw_prev = wsncs->phase_S[PHASE_WATER];
    const double sum_prev = ag_prev + ao_prev + aw_prev;

    prev_df_holdup_input_valid =
        (wsncs->rho_avg_DF > tnm::min_compare)
        && std::isfinite (ag_prev)
        && std::isfinite (ao_prev)
        && std::isfinite (aw_prev)
        && (ag_prev >= -1.0e-10)
        && (ao_prev >= -1.0e-10)
        && (aw_prev >= -1.0e-10)
        && (sum_prev > tnm::min_compare);

    if (prev_df_holdup_input_valid)
      {
        alpha_g_lagged_df_input = std::max (0.0, ag_prev / sum_prev);
        alpha_o_lagged_df_input = std::max (0.0, ao_prev / sum_prev);
        alpha_w_lagged_df_input = std::max (0.0, aw_prev / sum_prev);

        const double sum_norm = alpha_g_lagged_df_input
                                + alpha_o_lagged_df_input
                                + alpha_w_lagged_df_input;
        if (sum_norm > tnm::min_compare)
          {
            alpha_g_lagged_df_input /= sum_norm;
            alpha_o_lagged_df_input /= sum_norm;
            alpha_w_lagged_df_input /= sum_norm;
          }
        else
          {
            prev_df_holdup_input_valid = false;
            alpha_g_lagged_df_input = 0.0;
            alpha_o_lagged_df_input = 0.0;
            alpha_w_lagged_df_input = 0.0;
          }
      }

  }

  const double q_tot_local = wsncs->wsn_mixture_molar_rate;
  fill_wsncs_from_element_status (wsncs, element_status, mp);
  wsncs->wsn_mixture_molar_rate = q_tot_local;

  const double area = seg.wsn->pipe_props.area * internal_const::DAYS_TO_SEC ();
  const double diameter = seg.wsn->pipe_props.diameter;
  const double surf_mult = internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT ();

  const double average_volumetric_velocity_from_rate_raw =
      tnav_div (wsncs->wsn_mixture_molar_rate, area * element_status->avg_xi);
  double average_volumetric_velocity_raw = average_volumetric_velocity_from_rate_raw;
  const double df_flow_sign = (average_volumetric_velocity_raw < 0.0) ? -1.0 : 1.0;
  average_volumetric_velocity = df_flow_sign * average_volumetric_velocity_raw; // local DF frame: j >= 0
  avg_xi = element_status->avg_xi;
  xi = element_status->phase_xi[PHASE_GAS];

  double mixture_superficial_velocity = average_volumetric_velocity;
  const double df_den_eps = tnm::min_compare;

  auto df_safe_nonzero =
    [&] (double v) -> double
    {
      if (!std::isfinite (v))
        return (v < 0.0) ? -df_den_eps : df_den_eps;
      if (fabs (v) < df_den_eps)
        return (v < 0.0) ? -df_den_eps : df_den_eps;
      return v;
    };

  double drift_incl_mult = 1.0;
  {
    const double seg_length_abs = fabs (seg.wsn->pipe_props.length);
    const double seg_depth_abs = fabs (seg.wsn->pipe_props.depth_change);
    if (seg_length_abs > tnm::min_compare)
      {
        double cos_theta = seg_depth_abs / seg_length_abs;
        if (cos_theta < 0.0)
          cos_theta = 0.0;
        else if (cos_theta > 1.0)
          cos_theta = 1.0;

        if (cos_theta >= 1.0 - tnm::min_compare)
          drift_incl_mult = 1.0;
        else
          {
            const double sin_theta = sqrt (std::max (0.0, 1.0 - cos_theta * cos_theta));
            drift_incl_mult = sqrt (cos_theta) * tnav_pow (1.0 + sin_theta, 2);
          }
      }
  }

  wsncs->wsn_mmw = 0.0;
  wsncs->wsn_mixture_mass_rate = 0.0;
  for (auto ic = mp.nc0; ic < mp.nc; ++ic)
    wsncs->wsn_component_rate[ic] = 0.0;

  if (seg.wsn->wsn_index == TOP_SEG_INDEX)
    {
      wsncs->wsn_C_0 = 1.0;
      wsncs->wsn_drift_velocity = 0.0;
      wsncs->wsn_C_0_OW = 1.0;
      wsncs->wsn_drift_velocity_OW = 0.0;

      phase_gas_velocity_buf = mixture_superficial_velocity;
      phase_liquid_velocity_buf = mixture_superficial_velocity;
      oil_velocity_buf = mixture_superficial_velocity;
      water_velocity_buf = mixture_superficial_velocity;

      dbg_alpha_g = element_status->phase_S[PHASE_GAS];
      dbg_alpha_o = element_status->phase_S[PHASE_OIL];
      dbg_alpha_w = element_status->phase_S[PHASE_WATER];
      dbg_beta_o = (dbg_alpha_o + dbg_alpha_w > tnm::min_division)
                     ? dbg_alpha_o / (dbg_alpha_o + dbg_alpha_w)
                     : 0.0;
      dbg_R_g = 0.0;
      dbg_R_o = 0.0;
      dbg_active_flags = (1u << 31);
      dbg_inner_it = 0;
      return;
    }

  const double alpha_g_flash_init = element_status->phase_S[PHASE_GAS];
  const double alpha_o_flash_init = element_status->phase_S[PHASE_OIL];
  const double alpha_w_flash_init = element_status->phase_S[PHASE_WATER];

  bool phase_inputs_from_lagged_map = prev_df_holdup_input_valid;
  bool lagged_phase_input_regularized = false;
  double lagged_phase_input_lambda = phase_inputs_from_lagged_map ? 1.0 : 0.0;
  double lagged_phase_input_det = 0.0;

  double alpha_g_seed_raw = phase_inputs_from_lagged_map
                              ? alpha_g_lagged_df_input
                              : alpha_g_flash_init;
  double alpha_o_seed_raw = phase_inputs_from_lagged_map
                              ? alpha_o_lagged_df_input
                              : alpha_o_flash_init;
  double alpha_w_seed_raw = phase_inputs_from_lagged_map
                              ? alpha_w_lagged_df_input
                              : alpha_w_flash_init;

  {
    const double sum_input = alpha_g_seed_raw + alpha_o_seed_raw + alpha_w_seed_raw;
    lagged_phase_input_det = sum_input;
    if (!std::isfinite (sum_input) || sum_input <= tnm::min_compare)
      {
        phase_inputs_from_lagged_map = false;
        lagged_phase_input_lambda = 0.0;
        alpha_g_seed_raw = alpha_g_flash_init;
        alpha_o_seed_raw = alpha_o_flash_init;
        alpha_w_seed_raw = alpha_w_flash_init;
      }
    else if (fabs (sum_input - 1.0) > 1.0e-10)
      {
        alpha_g_seed_raw /= sum_input;
        alpha_o_seed_raw /= sum_input;
        alpha_w_seed_raw /= sum_input;
        lagged_phase_input_regularized = true;
      }
  }

  double gas_superficial_velocity_input_raw = alpha_g_seed_raw * average_volumetric_velocity_raw;
  double oil_superficial_velocity_input_raw = alpha_o_seed_raw * average_volumetric_velocity_raw;
  double water_superficial_velocity_input_raw = alpha_w_seed_raw * average_volumetric_velocity_raw;


  double gas_superficial_velocity_input = df_flow_sign * gas_superficial_velocity_input_raw;
  double oil_superficial_velocity_input = df_flow_sign * oil_superficial_velocity_input_raw;
  double water_superficial_velocity_input = df_flow_sign * water_superficial_velocity_input_raw;

  (void) phase_inputs_from_lagged_map;
  (void) lagged_phase_input_regularized;
  (void) lagged_phase_input_lambda;
  (void) lagged_phase_input_det;

  const double liquid_superficial_velocity_input =
      oil_superficial_velocity_input + water_superficial_velocity_input;

  auto clamp01 =
    [&] (double v) -> double
    {
      if (!std::isfinite (v)) return 0.0;
      if (v < 0.0) return 0.0;
      if (v > 1.0) return 1.0;
      return v;
    };

  const double simplex_eps = 1.0e-12;
  auto project_df_simplex =
    [&] (double &alpha_g_inout, double &alpha_o_inout)
    {
      if (!std::isfinite (alpha_g_inout))
        alpha_g_inout = alpha_g_flash_init;
      if (!std::isfinite (alpha_o_inout))
        alpha_o_inout = alpha_o_flash_init;

      if (alpha_g_inout < simplex_eps)
        alpha_g_inout = simplex_eps;
      if (alpha_g_inout > 1.0 - simplex_eps)
        alpha_g_inout = 1.0 - simplex_eps;

      double alpha_l_local = 1.0 - alpha_g_inout;
      if (alpha_l_local < simplex_eps)
        {
          alpha_g_inout = 1.0 - simplex_eps;
          alpha_l_local = simplex_eps;
        }

      if (alpha_o_inout < 0.0)
        alpha_o_inout = 0.0;
      if (alpha_o_inout > alpha_l_local)
        alpha_o_inout = alpha_l_local;
    };

  const double alpha_l_seed_local = alpha_o_seed_raw + alpha_w_seed_raw;
  const double beta_input =
      (alpha_l_seed_local > df_den_eps)
        ? clamp01 (alpha_o_seed_raw / alpha_l_seed_local)
        : 0.5;

  struct df_local_value_state_t
    {
      phase_holdups_DF holdups = phase_holdups_DF (0.0, 0.0, 0.0, 0.0);
      phase_vel_DF vels = phase_vel_DF (0.0, 0.0, 0.0, 0.0);

      double sigma_o = 0.0;
      double sigma_w = 0.0;
      double gas_liq_interfacial_tension = 0.0;
      double liq_density = 0.0;
      double gas_density = 0.0;
      double bubble_rise_velocity = 0.0;
      double diametr_dimless = 0.0;
      double Kut_number = 0.0;
      double flooding_velocity = 0.0;
      double ksi = 0.0;
      double eta = 0.0;
      double C0 = 1.0;
      double K_g = 0.0;
      double drift_velocity = 0.0;
      double oil_water_interfacial_tension = 0.0;
      double bubble_rise_velocity_OW = 0.0;
      double C0_OW = 1.0;
      double drift_velocity_OW = 0.0;
    };

  auto evaluate_df_value_alphao =
    [&] (double alpha_g_trial, double alpha_o_trial, df_local_value_state_t &S)
    {
      project_df_simplex (alpha_g_trial, alpha_o_trial);

      const double alpha_l_trial = 1.0 - alpha_g_trial;
      const double alpha_w_trial = std::max (0.0, alpha_l_trial - alpha_o_trial);
      const double beta_o_trial =
          (alpha_l_trial > df_den_eps) ? clamp01 (alpha_o_trial / alpha_l_trial) : beta_input;

      S.holdups.gas = alpha_g_trial;
      S.holdups.liquid = alpha_l_trial;
      S.holdups.oil = alpha_o_trial;
      S.holdups.water = alpha_w_trial;

      double D_sigma_o_dummy = 0.0;
      double D_sigma_w_dummy = 0.0;

      S.sigma_o = pipe_gas_oil_interfacial_tension_and_deriv (
          45.5,
          element_status->p * converter_metric_to_field.pressure_mult (),
          160.0,
          D_sigma_o_dummy);

      S.sigma_w = pipe_gas_wat_interfacial_tension_and_deriv (
          element_status->p * converter_metric_to_field.pressure_mult (),
          160.0,
          D_sigma_w_dummy);

      const double sigma_o_SI = surf_mult * S.sigma_o;
      const double sigma_w_SI = surf_mult * S.sigma_w;

      S.gas_liq_interfacial_tension =
          beta_o_trial * sigma_o_SI + (1.0 - beta_o_trial) * sigma_w_SI;

      S.liq_density =
          beta_o_trial * element_status->phase_rho[PHASE_OIL]
          + (1.0 - beta_o_trial) * element_status->phase_rho[PHASE_WATER];

      S.gas_density = std::max (element_status->phase_rho[PHASE_GAS], df_den_eps);

      S.bubble_rise_velocity = 0.0;
      if (S.gas_liq_interfacial_tension > df_den_eps && S.liq_density > df_den_eps)
        {
          S.bubble_rise_velocity =
              tnav_pow (
                  tnav_div (S.gas_liq_interfacial_tension * internal_const::grav_metric ()
                            * fabs (S.liq_density - S.gas_density),
                            S.liq_density * S.liq_density),
                  0.25);
        }

      S.diametr_dimless = 0.0;
      if (S.liq_density - S.gas_density > 0.0 && S.gas_liq_interfacial_tension > df_den_eps)
        {
          S.diametr_dimless =
              sqrt (tnav_div (internal_const::grav_metric () * (S.liq_density - S.gas_density),
                              S.gas_liq_interfacial_tension))
              * diameter;
        }

      double lin_interp_dummy = 0.0;
      S.Kut_number = compute_critical_Kutateladze_number_by_diametr (S.diametr_dimless, lin_interp_dummy);

      S.flooding_velocity =
          S.Kut_number * sqrt (tnav_div (S.liq_density, std::max (S.gas_density, df_den_eps)))
          * S.bubble_rise_velocity;

      const double A = 1.2;
      const double B = 0.3;
      const double F_v = 1.0;
      const double V_sgf = df_safe_nonzero (S.flooding_velocity);

      S.ksi = std::max (alpha_g_trial,
                        tnav_div (F_v * alpha_g_trial * fabs (mixture_superficial_velocity), V_sgf));

      S.eta = (S.ksi - B) / (1.0 - B);
      if (!std::isfinite (S.eta))
        S.eta = (S.ksi > B) ? 1.0 : 0.0;
      else if (S.eta < 0.0)
        S.eta = 0.0;
      else if (S.eta > 1.0)
        S.eta = 1.0;

      S.C0 = A / (1.0 + (A - 1.0) * S.eta * S.eta);

      const double K_g_low = 1.53 / S.C0;
      const double K_g_high = S.Kut_number;
      if (alpha_g_trial < 0.2)
        S.K_g = K_g_low;
      else if (alpha_g_trial > 0.4)
        S.K_g = K_g_high;
      else
        S.K_g = interpolate_y_against_x (alpha_g_trial, 0.2, 0.4, K_g_low, K_g_high);

      {
        const double sqrt_gas_over_liq =
            sqrt (tnav_div (std::max (S.gas_density, df_den_eps), std::max (S.liq_density, df_den_eps)));
        const double numerator_gd =
            (1.0 - alpha_g_trial * S.C0) * S.C0 * S.K_g * S.bubble_rise_velocity;
        const double denominator_gd =
            1.0 + alpha_g_trial * S.C0 * (sqrt_gas_over_liq - 1.0);
        S.drift_velocity =
            drift_incl_mult * tnav_div (numerator_gd, df_safe_nonzero (denominator_gd));
      }

      S.vels.gas = S.C0 * mixture_superficial_velocity + S.drift_velocity;

      if (1.0 - alpha_g_trial > df_den_eps)
        {
          S.vels.liquid =
              tnav_div (1.0 - alpha_g_trial * S.C0, 1.0 - alpha_g_trial) * mixture_superficial_velocity
              - tnav_div (alpha_g_trial, 1.0 - alpha_g_trial) * S.drift_velocity;
        }
      else
        {
          S.vels.liquid = mixture_superficial_velocity;
        }

      const double B1_OW = 0.4;
      const double B2_OW = 0.7;
      const double A_OW = 1.2;

      if (beta_o_trial < B1_OW)
        S.C0_OW = A_OW;
      else if (beta_o_trial > B2_OW)
        S.C0_OW = 1.0;
      else
        S.C0_OW = interpolate_y_against_x (beta_o_trial, B1_OW, B2_OW, A_OW, 1.0);

      double D_go_dummy = 0.0;
      double D_gw_dummy = 0.0;

      const double gas_oil_interfacial_tension =
          surf_mult * pipe_gas_oil_interfacial_tension_and_deriv (
              45.5,
              element_status->p * converter_metric_to_field.pressure_mult (),
              160.0,
              D_go_dummy);

      const double gas_wat_interfacial_tension =
          surf_mult * pipe_gas_wat_interfacial_tension_and_deriv (
              element_status->p * converter_metric_to_field.pressure_mult (),
              160.0,
              D_gw_dummy);

      S.oil_water_interfacial_tension =
          fabs (gas_oil_interfacial_tension * beta_o_trial - gas_wat_interfacial_tension * (1.0 - beta_o_trial));

      S.bubble_rise_velocity_OW = 0.0;
      if (S.oil_water_interfacial_tension > df_den_eps
          && element_status->phase_rho[PHASE_WATER] > element_status->phase_rho[PHASE_OIL]
          && element_status->phase_rho[PHASE_WATER] > df_den_eps)
        {
          S.bubble_rise_velocity_OW =
              tnav_pow (
                  S.oil_water_interfacial_tension * internal_const::grav_metric ()
                  * (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL])
                  / (element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER]),
                  0.25);
        }

      S.drift_velocity_OW =
          drift_incl_mult * 1.53 * S.bubble_rise_velocity_OW * tnav_pow (1.0 - beta_o_trial, 2);

      S.vels.oil = S.C0_OW * S.vels.liquid + S.drift_velocity_OW;

      if (1.0 - beta_o_trial > df_den_eps)
        {
          S.vels.water =
              tnav_div (1.0 - beta_o_trial * S.C0_OW, 1.0 - beta_o_trial) * S.vels.liquid
              - tnav_div (beta_o_trial, 1.0 - beta_o_trial) * S.drift_velocity_OW;
        }
      else
        {
          S.vels.water = S.vels.liquid;
        }
    };


  auto compute_residuals_alphao =
    [&] (const df_local_value_state_t &S,
         double &Rg_out,
         double &Ro_out,
         double &Rw_out,
         double &res_norm_out)
    {
      Rg_out = S.holdups.gas * S.vels.gas - gas_superficial_velocity_input;
      Ro_out = S.holdups.oil * S.vels.oil - oil_superficial_velocity_input;
      Rw_out = S.holdups.water * S.vels.water - water_superficial_velocity_input;

      const double Rg_rel = fabs (Rg_out) / std::max (1.0, fabs (gas_superficial_velocity_input));
      const double Ro_rel = fabs (Ro_out) / std::max (1.0, fabs (oil_superficial_velocity_input));
      const double Rw_rel = fabs (Rw_out) / std::max (1.0, fabs (water_superficial_velocity_input));
      res_norm_out = sqrt (Rg_rel * Rg_rel + Ro_rel * Ro_rel + Rw_rel * Rw_rel);
    };

  // Two-stage DF solve used by test_function as well:
  // gas/liquid root first at fixed beta_input, then oil/water root at fixed alpha_l.
  const double beta_gl_fixed = beta_input;
  const double beta_min = simplex_eps;
  const double beta_max = 1.0 - simplex_eps;

  auto evaluate_gl_value = [&] (double alpha_g_trial, df_local_value_state_t &S)
    {
      const double alpha_g_clamped = std::max (simplex_eps,
                                               std::min (1.0 - simplex_eps, alpha_g_trial));
      const double alpha_o_trial = beta_gl_fixed * (1.0 - alpha_g_clamped);
      evaluate_df_value_alphao (alpha_g_clamped, alpha_o_trial, S);
    };

  auto compute_rg_only = [&] (double alpha_g_trial, df_local_value_state_t *S_out = nullptr) -> double
    {
      df_local_value_state_t S_tmp;
      evaluate_gl_value (alpha_g_trial, S_tmp);
      if (S_out)
        *S_out = S_tmp;
      return S_tmp.holdups.gas * S_tmp.vels.gas - gas_superficial_velocity_input;
    };

  auto evaluate_ow_value = [&] (double alpha_g_fixed, double beta_trial, df_local_value_state_t &S)
    {
      const double alpha_g_clamped = std::max (simplex_eps,
                                               std::min (1.0 - simplex_eps, alpha_g_fixed));
      const double alpha_l_fixed = std::max (0.0, 1.0 - alpha_g_clamped);
      const double beta_clamped = clamp01 (beta_trial);
      const double alpha_o_trial = alpha_l_fixed * beta_clamped;
      evaluate_df_value_alphao (alpha_g_clamped, alpha_o_trial, S);
    };

  auto compute_ro_only = [&] (double alpha_g_fixed, double beta_trial, df_local_value_state_t *S_out = nullptr) -> double
    {
      df_local_value_state_t S_tmp;
      evaluate_ow_value (alpha_g_fixed, beta_trial, S_tmp);
      if (S_out)
        *S_out = S_tmp;
      return S_tmp.holdups.oil * S_tmp.vels.oil - oil_superficial_velocity_input;
    };

  auto find_first_root_bracket = [&] (const std::function<double(double)> &f,
                                      double lo,
                                      double hi,
                                      int nscan,
                                      double &x_left,
                                      double &x_right,
                                      double &x_best,
                                      double &f_best) -> bool
    {
      x_left = lo;
      x_right = hi;
      x_best = lo;
      f_best = f (lo);
      double best_abs = fabs (f_best);

      double x_prev = lo;
      double f_prev = f_best;
      for (int is = 1; is <= nscan; ++is)
        {
          const double t = static_cast<double> (is) / static_cast<double> (nscan);
          const double x_now = lo + (hi - lo) * t;
          const double f_now = f (x_now);
          if (fabs (f_now) < best_abs)
            {
              best_abs = fabs (f_now);
              x_best = x_now;
              f_best = f_now;
            }
          if (std::isfinite (f_prev) && std::isfinite (f_now) && f_prev * f_now <= 0.0)
            {
              x_left = x_prev;
              x_right = x_now;
              return true;
            }
          x_prev = x_now;
          f_prev = f_now;
        }
      return false;
    };
  (void) find_first_root_bracket;

  struct df_root_bracket_t
    {
      double xl = 0.0;
      double xr = 0.0;
      double fl = 0.0;
      double fr = 0.0;
    };

  auto collect_root_brackets = [&] (const std::function<double(double)> &f,
                                    double lo,
                                    double hi,
                                    int nscan,
                                    std::vector<df_root_bracket_t> &brackets,
                                    double &x_best,
                                    double &f_best)
    {
      brackets.clear ();
      x_best = lo;
      f_best = f (lo);
      double best_abs = fabs (f_best);

      double x_prev = lo;
      double f_prev = f_best;
      for (int is = 1; is <= nscan; ++is)
        {
          const double t = static_cast<double> (is) / static_cast<double> (nscan);
          const double x_now = lo + (hi - lo) * t;
          const double f_now = f (x_now);
          if (fabs (f_now) < best_abs)
            {
              best_abs = fabs (f_now);
              x_best = x_now;
              f_best = f_now;
            }
          if (std::isfinite (f_prev) && std::isfinite (f_now) && f_prev * f_now <= 0.0)
            {
              df_root_bracket_t B;
              B.xl = x_prev;
              B.xr = x_now;
              B.fl = f_prev;
              B.fr = f_now;
              brackets.push_back (B);
            }
          x_prev = x_now;
          f_prev = f_now;
        }
    };

  auto select_nearest_root_bracket = [&] (const std::vector<df_root_bracket_t> &brackets,
                                          double x_target) -> int
    {
      if (brackets.empty ())
        return -1;
      int best_idx = 0;
      double best_dist = 1.0e300;
      for (size_t ib = 0; ib < brackets.size (); ++ib)
        {
          const double xl = brackets[ib].xl;
          const double xr = brackets[ib].xr;
          double dist = 0.0;
          if (x_target < xl)
            dist = xl - x_target;
          else if (x_target > xr)
            dist = x_target - xr;
          else
            dist = 0.0;
          if (dist < best_dist)
            {
              best_dist = dist;
              best_idx = static_cast<int> (ib);
            }
        }
      return best_idx;
    };

  auto solve_scalar_root = [&] (const std::function<double(double)> &f,
                                double x_left,
                                double x_right,
                                double x_seed,
                                double tol,
                                int max_it_local,
                                double &x_out,
                                double &f_out,
                                int &it_out) -> bool
    {
      double xl = x_left;
      double xr = x_right;
      double fl = f (xl);
      double fr = f (xr);
      if (!(std::isfinite (fl) && std::isfinite (fr)) || fl * fr > 0.0)
        {
          x_out = x_seed;
          f_out = f (x_out);
          it_out = 0;
          return false;
        }

      double x = std::max (xl, std::min (xr, x_seed));
      it_out = 0;
      for (int it_loc = 0; it_loc < max_it_local; ++it_loc)
        {
          ++it_out;
          const double fx = f (x);
          f_out = fx;
          if (fabs (fx) <= tol)
            {
              x_out = x;
              return true;
            }

          const double eps_x = std::max (1.0e-8, 1.0e-6 * std::max (1.0, fabs (x)));
          const double xp = std::min (xr, x + eps_x);
          const double xm = std::max (xl, x - eps_x);
          const double fp = f (xp);
          const double fm = f (xm);
          const double denom = std::max (xp - xm, 1.0e-14);
          const double dfdx = (fp - fm) / denom;

          double x_trial = x;
          if (std::isfinite (dfdx) && fabs (dfdx) > 1.0e-14)
            x_trial = x - fx / dfdx;
          if (!(x_trial > xl && x_trial < xr) || !std::isfinite (x_trial))
            x_trial = 0.5 * (xl + xr);

          const double f_trial = f (x_trial);
          if (fl * f_trial <= 0.0)
            {
              xr = x_trial;
              fr = f_trial;
            }
          else
            {
              xl = x_trial;
              fl = f_trial;
            }
          x = x_trial;
        }

      x_out = x;
      f_out = f (x);
      return (fabs (f_out) <= tol);
    };

  auto apply_df_flow_sign_to_state = [&] (df_local_value_state_t &S)
    {
      S.vels.gas *= df_flow_sign;
      S.vels.liquid *= df_flow_sign;
      S.vels.oil *= df_flow_sign;
      S.vels.water *= df_flow_sign;
      S.drift_velocity *= df_flow_sign;
      S.drift_velocity_OW *= df_flow_sign;
    };

  double alpha_g_gl = std::max (simplex_eps, std::min (1.0 - simplex_eps, alpha_g_seed_raw));
  double beta_ow = clamp01 (beta_input);
  double gl_best = alpha_g_gl;
  double gl_best_res = 0.0;
  double gl_left = simplex_eps;
  double gl_right = 1.0 - simplex_eps;
  int gl_it = 0;
  int ow_it = 0;

  std::vector<df_root_bracket_t> gl_brackets;
  collect_root_brackets (
      [&] (double ag) { return compute_rg_only (ag, nullptr); },
      simplex_eps, 1.0 - simplex_eps, 64,
      gl_brackets, gl_best, gl_best_res);
  const bool gl_root_exists = !gl_brackets.empty ();

  double Rg_stage = 0.0;
  if (gl_root_exists)
    {
      const int ibest_gl = select_nearest_root_bracket (gl_brackets, alpha_g_gl);
      gl_left = gl_brackets[ibest_gl].xl;
      gl_right = gl_brackets[ibest_gl].xr;
      const double alpha_g_seed_local = std::max (gl_left, std::min (gl_right, alpha_g_gl));
      solve_scalar_root ([&] (double ag) { return compute_rg_only (ag, nullptr); },
                         gl_left, gl_right, alpha_g_seed_local, 1.0e-12, 32,
                         alpha_g_gl, Rg_stage, gl_it);
    }
  else
    {
      alpha_g_gl = gl_best;
      Rg_stage = compute_rg_only (alpha_g_gl, nullptr);
    }

  const double alpha_l_gl = std::max (0.0, 1.0 - alpha_g_gl);
  if (alpha_l_gl > 1.0e-10 && fabs (liquid_superficial_velocity_input) > 1.0e-14)
    {
      const double beta_target = clamp01 (beta_input);
      const double Ro_target = compute_ro_only (alpha_g_gl, beta_target, nullptr);

      double beta_best = beta_target;
      double beta_best_res = 0.0;
      std::vector<df_root_bracket_t> ow_brackets;
      collect_root_brackets (
          [&] (double b) { return compute_ro_only (alpha_g_gl, b, nullptr); },
          beta_min, beta_max, 64,
          ow_brackets, beta_best, beta_best_res);

      double Ro_stage = 0.0;
      if (std::isfinite (Ro_target) && fabs (Ro_target) <= 1.0e-12)
        {
          beta_ow = beta_target;
          Ro_stage = Ro_target;
        }
      else if (!ow_brackets.empty ())
        {
          const int ibest = select_nearest_root_bracket (ow_brackets, beta_target);
          const double beta_left = ow_brackets[ibest].xl;
          const double beta_right = ow_brackets[ibest].xr;
          const double beta_seed_local = std::max (beta_left, std::min (beta_right, beta_target));
          solve_scalar_root ([&] (double b) { return compute_ro_only (alpha_g_gl, b, nullptr); },
                             beta_left, beta_right, beta_seed_local, 1.0e-12, 32,
                             beta_ow, Ro_stage, ow_it);
        }
      else
        {
          beta_ow = beta_best;
          Ro_stage = compute_ro_only (alpha_g_gl, beta_ow, nullptr);
        }
      (void) Ro_stage;
    }
  else
    {
      beta_ow = clamp01 (beta_input);
    }

  df_local_value_state_t state_now;
  evaluate_ow_value (alpha_g_gl, beta_ow, state_now);

  double error = 0.0;
  {
    double Rg = 0.0, Ro = 0.0, Rw = 0.0, res_norm = 0.0;
    compute_residuals_alphao (state_now, Rg, Ro, Rw, res_norm);
    error = res_norm;
  }
  int it = gl_it + ow_it;
  dbg_inner_it = it;

  apply_df_flow_sign_to_state (state_now);
  phase_holdups_DF prev_holdups (0.0, 0.0, 0.0, 0.0);
  phase_vel_DF prev_vels (0.0, 0.0, 0.0, 0.0);
  phase_holdups_DF new_holdups (0.0, 0.0, 0.0, 0.0);
  phase_vel_DF new_vels (0.0, 0.0, 0.0, 0.0);
  prev_holdups.copy_operator (&state_now.holdups);
  prev_vels.copy_operator (&state_now.vels);
  new_holdups.copy_operator (&state_now.holdups);
  new_vels.copy_operator (&state_now.vels);

  evaluate_df_value_alphao (prev_holdups.gas, prev_holdups.oil, state_now);
  apply_df_flow_sign_to_state (state_now);
  new_holdups.copy_operator (&state_now.holdups);
  new_vels.copy_operator (&state_now.vels);

  const double alpha_g = new_holdups.gas;
  const double alpha_l = new_holdups.liquid;
  const double alpha_o = new_holdups.oil;
  const double alpha_w = new_holdups.water;
  const double beta_o = (alpha_l > df_den_eps) ? (alpha_o / alpha_l) : beta_input;

  wsncs->phase_S[PHASE_GAS] = alpha_g;
  wsncs->phase_S[PHASE_OIL] = alpha_o;
  wsncs->phase_S[PHASE_WATER] = alpha_w;

  wsncs->wsn_C_0 = state_now.C0;
  wsncs->wsn_drift_velocity = state_now.drift_velocity;
  wsncs->wsn_C_0_OW = state_now.C0_OW;
  wsncs->wsn_drift_velocity_OW = state_now.drift_velocity_OW;

  dbg_alpha_g = alpha_g;
  dbg_alpha_o = alpha_o;
  dbg_alpha_w = alpha_w;
  dbg_beta_o = beta_o;

  gas_inter_tension = state_now.gas_liq_interfacial_tension;
  bubble_rise_velocity_buf = state_now.bubble_rise_velocity;
  liq_density_buf = state_now.liq_density;
  diametr_dimless_buf = state_now.diametr_dimless;
  Kut_number_buf = state_now.Kut_number;
  flooding_velocity_buf = state_now.flooding_velocity;
  ksi_buf = state_now.ksi;
  eta_buf = state_now.eta;
  K_g_buf = state_now.K_g;
  sigma_o = state_now.sigma_o;
  sigma_w = state_now.sigma_w;
  oil_inter_tension_buf = state_now.oil_water_interfacial_tension;
  bubble_rise_velocity_OW_buf = state_now.bubble_rise_velocity_OW;
  C_OW_buf = state_now.C0_OW;
  V_d_OW_buf = state_now.drift_velocity_OW;
  phase_gas_velocity_buf = state_now.vels.gas;
  phase_liquid_velocity_buf = state_now.vels.liquid;
  oil_velocity_buf = state_now.vels.oil;
  water_velocity_buf = state_now.vels.water;

  dbg_R_g = alpha_g * state_now.vels.gas - gas_superficial_velocity_input_raw;
  dbg_R_o = alpha_o * state_now.vels.oil - oil_superficial_velocity_input_raw;

  dbg_active_flags = 0U;
  if (alpha_g < 0.2)
    dbg_active_flags |= (1u << 7);
  else if (alpha_g > 0.4)
    dbg_active_flags |= (1u << 8);
  else
    dbg_active_flags |= (1u << 9);

  const double B1_OW = 0.4;
  const double B2_OW = 0.7;
  if (beta_o < B1_OW)
    dbg_active_flags |= (1u << 10);
  else if (beta_o > B2_OW)
    dbg_active_flags |= (1u << 11);
  else
    dbg_active_flags |= (1u << 12);

  if (state_now.eta <= 0.0 + 1.0e-14)
    dbg_active_flags |= (1u << 13);
  else if (state_now.eta >= 1.0 - 1.0e-14)
    dbg_active_flags |= (1u << 14);
  else
    dbg_active_flags |= (1u << 15);

  const double ksi_second =
      tnav_div (1.0 * alpha_g * fabs (mixture_superficial_velocity),
                df_safe_nonzero (state_now.flooding_velocity));
  const unsigned int ksi_flag = (alpha_g > ksi_second) ? 1U : 2U;
  dbg_active_flags |= (ksi_flag << 16);

  const double rho_avg_df =
      element_status->phase_rho[PHASE_GAS] * alpha_g
      + element_status->phase_rho[PHASE_OIL] * alpha_o
      + element_status->phase_rho[PHASE_WATER] * alpha_w;
  wsncs->rho_avg_DF = rho_avg_df;

  for (auto ic = mp.nc0; ic < mp.nc; ++ic)
    wsncs->wsn_component_rate[ic] = 0.0;

  for (unsigned int ip = 0; ip < mp.np; ++ip)
    {
      double phase_velocity = 0.0;
      double phase_holdup = 0.0;
      if (ip == PHASE_GAS)
        {
          phase_velocity = state_now.vels.gas;
          phase_holdup = alpha_g;
        }
      else if (ip == PHASE_OIL)
        {
          phase_velocity = state_now.vels.oil;
          phase_holdup = alpha_o;
        }
      else
        {
          phase_velocity = state_now.vels.water;
          phase_holdup = alpha_w;
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

  wsncs->wsn_mixture_mass_rate = 0.0;
  //wsncs->wsn_mmw = 0.0;
  for (auto ic = mp.nc0; ic < mp.nc; ++ic)
    {
      wsncs->wsn_mixture_mass_rate +=
          wsncs->wsn_component_rate[ic] * component_molar_weights[ic];
      //wsncs->wsn_mmw += component_z_for_flow[ic] * component_molar_weights[ic];
    }
}
