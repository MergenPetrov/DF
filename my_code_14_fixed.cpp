if (element_status)
        {
          const double q_tot_buf = wsncs->wsn_mixture_molar_rate;
          const double surf_mult = internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT ();
          const unsigned int nseg_vars_fd = 1U + mp.nc + 1U;
          // 0              -> pressure
          // 1 .. mp.nc     -> component_N[id - 1]
          // nseg_vars_fd-1 -> q_tot
          const unsigned int fd_var_id = nseg_vars_fd - nseg_vars_fd + 0;

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

          // Previous DF-state holdup guess captured BEFORE flash values overwrite wsncs.
          const double prev_alpha_g_state_raw = std::isfinite (wsncs->phase_S[PHASE_GAS])   ? wsncs->phase_S[PHASE_GAS]   : 0.0;
          const double prev_alpha_o_state_raw = std::isfinite (wsncs->phase_S[PHASE_OIL])   ? wsncs->phase_S[PHASE_OIL]   : 0.0;
          const double prev_alpha_w_state_raw = std::isfinite (wsncs->phase_S[PHASE_WATER]) ? wsncs->phase_S[PHASE_WATER] : 0.0;

          bool use_previous_df_holdup_seed = false;
          double prev_alpha_g_state = 0.0;
          double prev_alpha_o_state = 0.0;
          double prev_alpha_w_state = 0.0;
          {
            const double prev_sum_raw = prev_alpha_g_state_raw + prev_alpha_o_state_raw + prev_alpha_w_state_raw;
            if (std::isfinite (prev_sum_raw) && prev_sum_raw > 1.0e-12 &&
                prev_alpha_g_state_raw >= 0.0 && prev_alpha_o_state_raw >= 0.0 && prev_alpha_w_state_raw >= 0.0)
              {
                use_previous_df_holdup_seed = true;
                prev_alpha_g_state = prev_alpha_g_state_raw / prev_sum_raw;
                prev_alpha_o_state = prev_alpha_o_state_raw / prev_sum_raw;
                prev_alpha_w_state = prev_alpha_w_state_raw / prev_sum_raw;
              }
          }

          // prev outputs
          double prev_average_mixture_velocity = 0.;
          double prev_gas_liq_interfacial_tension = 0.;
          double prev_liquid_hp = 0.;
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
          double next_liquid_hp = 0.;
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
          double beta_o_num_derivative  = (next_beta_o  - prev_beta_o ) / (2.0 * fd_eps);
          //double alpha_o_num_derivative = (next_alpha_o - prev_alpha_o) / (2.0 * fd_eps);
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
          double liquid_hp_num = (next_liquid_hp - prev_liquid_hp) / (2.0 * fd_eps);
          double rho_avg_numerical = (next_rho_avg_dbg - prev_rho_avg_dbg) / (2.0 * fd_eps);
          double oil_water_interfacial_tension_num = (next_oil_water_interfacial_tension - prev_oil_water_interfacial_tension) / (2.0 * fd_eps);
          double bubble_rise_velocity_OW_num = (next_bubble_rise_velocity_OW - prev_bubble_rise_velocity_OW) / (2.0 * fd_eps);
          double C_OW_num = (next_C_OW_buf - prev_C_OW_buf) / (2.0 * fd_eps);
          double V_d_OW_num = (next_V_d_OW_buf - prev_V_d_OW_buf) / (2.0 * fd_eps);
          double oil_velocity_num = (next_oil_velocity_buf - prev_oil_velocity_buf) / (2.0 * fd_eps);
          double water_velocity_num = (next_water_velocity_buf - prev_water_velocity_buf) / (2.0 * fd_eps);

                  copy_segment_params_to_element_status (seg, element_status);

                  if (wsncs->wsn_flow_dir == segment_flow_direction_t::from_parent_to_child)
                    {
                      // in this case derivatives are calcluated wrt q_tot and pressure from curr seg and z_c and enth from parent segment
                      //densities, rates, holdups, viscosities are calculated for curr segment based on parent properties
                      set_flow_direction_dependent_segment_params_to_element_status (seg, element_status, new_status);
                     }

                 if (element_status->component_N_tot > tnm::min_compare)
                   {
                     if (auto err = run_flash <true> (rep, i_meshblock, element_status, current_therm_comp_input_props, itd); err != segments_solver_err_t::none)
                       return;
                   }

                  //if (!(seg.wsn->wsn_index == TOP_SEG_INDEX))

                    fill_wsncs_from_element_status (wsncs, element_status, mp);
                    wsncs->wsn_mixture_molar_rate = q_tot_buf;

                  const unsigned int nseg_vars = 1U + mp.nc + 1U;

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

                          double average_volumetric_mixture_velocity = tnav_div (wsncs->wsn_mixture_molar_rate, seg.wsn->pipe_props.area * element_status->avg_xi);

                              for (unsigned int id = 0; id < 1U + mp.nc + 1U; id++)
                                {
                                  double water = element_status->component_N[0];
                                  double oil = element_status->component_N[1];
                                  double gas = element_status->component_N[2];
                                  (void) water;
                                  (void) oil;
                                  (void) gas;

                                  if (id < 1U + mp.nc) //p^j, z^c_j
                                    wsncs->D_average_volumetric_mixture_velocity_D_seg_vars[id] = -tnav_div (wsncs->wsn_mixture_molar_rate, seg.wsn->pipe_props.area * element_status->avg_xi * element_status->avg_xi)
                                                                                                * element_status->avg_D_xi[id];
                                                                                                //* D_avg_xi_numerical[id];
                                  else
                                    wsncs->D_average_volumetric_mixture_velocity_D_seg_vars[id] = tnav_div (1., seg.wsn->pipe_props.area * element_status->avg_xi);
                                }

                          (void) error;
                          (void) it;
                          (void) max_it;

                          const double mixture_superficial_velocity =
                              average_volumetric_mixture_velocity; /// internal_const::DAYS_TO_SEC (); // [m/s]

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

                          // FIX #B: seed для holdups — equal-split (1/3, 1/3, 1/3), не phase_S.
                          // phase_S — это флэш-сатурация, она завышает долю газа для типичных режимов.
                          const double alpha_g_equal_seed = 1.0 / 3.0;
                          const double alpha_o_equal_seed = 1.0 / 3.0;
                          const double alpha_w_equal_seed = 1.0 / 3.0;

                          const double alpha_g_seed = use_previous_df_holdup_seed ? prev_alpha_g_state : alpha_g_equal_seed;
                          const double alpha_o_seed = use_previous_df_holdup_seed ? prev_alpha_o_state : alpha_o_equal_seed;
                          const double alpha_w_seed = use_previous_df_holdup_seed ? prev_alpha_w_state : alpha_w_equal_seed;
                          const double alpha_l_init = alpha_o_seed + alpha_w_seed;
                          const double alpha_g_init = alpha_g_seed;

                          double beta_o_init = 0.0;
                          if (fabs (alpha_l_init) > df_den_eps)
                            beta_o_init = alpha_o_seed / alpha_l_init;

                          phase_holdups_DF prev_holdups (alpha_g_init,
                                                         alpha_l_init,
                                                         alpha_o_seed,
                                                         alpha_w_seed);
                          phase_holdups_DF temp_holdups (0., 0., 0., 0.);
                          phase_holdups_DF new_holdups (0., 0., 0., 0.);

                          std::vector<double> D_alpha_g_iter (nseg_vars, 0.0);
                          std::vector<double> D_beta_o_iter (nseg_vars, 0.0);
                          std::vector<double> D_alpha_g_next (nseg_vars, 0.0);
                          std::vector<double> D_beta_o_next (nseg_vars, 0.0);

                          // FIX #B: equal-split seed не зависит от seg_vars, поэтому
                          // D_alpha_g_iter и D_beta_o_iter стартуют нулями. В ветви
                          // use_previous_df_holdup_seed persisted seed также берётся как
                          // константа предыдущей Newton-итерации (не функция текущих seg_vars),
                          // так что его производные равны нулю на входе цикла.
                          (void) alpha_g_equal_seed;
                          (void) alpha_o_equal_seed;
                          (void) alpha_w_equal_seed;

                          double avg_xi_prev_seed = 0.0;
                          if (use_previous_df_holdup_seed)
                            {
                              for (unsigned int ip = 0; ip < mp.np; ++ip)
                                avg_xi_prev_seed += wsncs_backup.phase_S[ip] * wsncs_backup.phase_xi[ip];
                            }

                          double mixture_superficial_velocity_seed = mixture_superficial_velocity;
                          if (use_previous_df_holdup_seed && fabs (avg_xi_prev_seed) > df_den_eps)
                            {
                              mixture_superficial_velocity_seed =
                                  wsncs_backup.wsn_mixture_molar_rate / (seg.wsn->pipe_props.area * avg_xi_prev_seed);
                            }

                          double gas_phase_velocity_seed = mixture_superficial_velocity_seed;
                          double liquid_phase_velocity_seed = mixture_superficial_velocity_seed;
                          if (use_previous_df_holdup_seed)
                            {
                              gas_phase_velocity_seed = wsncs_backup.wsn_C_0 * mixture_superficial_velocity_seed + wsncs_backup.wsn_drift_velocity;
                              if (fabs (1.0 - alpha_g_seed) > df_den_eps)
                                {
                                  liquid_phase_velocity_seed =
                                      ((1.0 - alpha_g_seed * wsncs_backup.wsn_C_0) / (1.0 - alpha_g_seed)) * mixture_superficial_velocity_seed
                                      - (alpha_g_seed / (1.0 - alpha_g_seed)) * wsncs_backup.wsn_drift_velocity;
                                }
                            }

                          double oil_phase_velocity_seed = liquid_phase_velocity_seed;
                          double water_phase_velocity_seed = liquid_phase_velocity_seed;
                          if (use_previous_df_holdup_seed)
                            {
                              oil_phase_velocity_seed =
                                  wsncs_backup.wsn_C_0_OW * liquid_phase_velocity_seed + wsncs_backup.wsn_drift_velocity_OW;
                              if (fabs (1.0 - beta_o_init) > df_den_eps)
                                {
                                  water_phase_velocity_seed =
                                      ((1.0 - beta_o_init * wsncs_backup.wsn_C_0_OW) / (1.0 - beta_o_init)) * liquid_phase_velocity_seed
                                      - (beta_o_init / (1.0 - beta_o_init)) * wsncs_backup.wsn_drift_velocity_OW;
                                }
                              else
                                {
                                  water_phase_velocity_seed = liquid_phase_velocity_seed;
                                }
                            }

                          // FIX #B: equal-split seed 1/3 в non-persisted ветке вместо phase_S.
                          const double gas_superficial_velocity_input =
                              use_previous_df_holdup_seed ? (alpha_g_seed * gas_phase_velocity_seed)
                                                          : (alpha_g_seed * mixture_superficial_velocity);
                          const double oil_superficial_velocity_input =
                              use_previous_df_holdup_seed ? (alpha_o_seed * oil_phase_velocity_seed)
                                                          : (alpha_o_seed * mixture_superficial_velocity);
                          const double water_superficial_velocity_input =
                              use_previous_df_holdup_seed ? (alpha_w_seed * water_phase_velocity_seed)
                                                          : (alpha_w_seed * mixture_superficial_velocity);

                          phase_vel_DF prev_vels (gas_phase_velocity_seed,
                                                  liquid_phase_velocity_seed,
                                                  oil_phase_velocity_seed,
                                                  water_phase_velocity_seed);
                          phase_vel_DF new_vels (0., 0., 0., 0.);

                          std::vector<double> D_gas_superficial_velocity_input_D_seg_vars   (nseg_vars, 0.);
                          std::vector<double> D_oil_superficial_velocity_input_D_seg_vars   (nseg_vars, 0.);
                          std::vector<double> D_water_superficial_velocity_input_D_seg_vars (nseg_vars, 0.);

                          // FIX #B: equal-split seed — константа, D_alpha_seed = 0.
                          // Non-persisted: D_vs_input = alpha_seed * D_mixture_superficial_velocity.
                          // Persisted: seed из prev Newton — не функция текущих seg_vars, D = 0.
                          if (!use_previous_df_holdup_seed)
                            {
                              for (unsigned int id = 0; id < nseg_vars; ++id)
                                {
                                  D_gas_superficial_velocity_input_D_seg_vars[id] =
                                      alpha_g_seed * D_mixture_superficial_velocity_D_seg_vars[id];
                                  D_oil_superficial_velocity_input_D_seg_vars[id] =
                                      alpha_o_seed * D_mixture_superficial_velocity_D_seg_vars[id];
                                  D_water_superficial_velocity_input_D_seg_vars[id] =
                                      alpha_w_seed * D_mixture_superficial_velocity_D_seg_vars[id];
                                }
                            }
                          else
                            {
                              for (unsigned int id = 0; id < nseg_vars; ++id)
                                {
                                  D_gas_superficial_velocity_input_D_seg_vars[id] = 0.0;
                                  D_oil_superficial_velocity_input_D_seg_vars[id] = 0.0;
                                  D_water_superficial_velocity_input_D_seg_vars[id] = 0.0;
                                }
                            }

                          const double liquid_superficial_velocity_input =
                              oil_superficial_velocity_input + water_superficial_velocity_input;

                          double oil_flow_weight_input = 0.5;
                          double water_flow_weight_input = 0.5;
                          std::vector<double> D_oil_flow_weight_input_D_seg_vars (nseg_vars, 0.0);
                          std::vector<double> D_water_flow_weight_input_D_seg_vars (nseg_vars, 0.0);
                          (void) D_water_flow_weight_input_D_seg_vars;
                          if (fabs (liquid_superficial_velocity_input) > df_den_eps)
                            {
                              oil_flow_weight_input = oil_superficial_velocity_input / liquid_superficial_velocity_input;
                              water_flow_weight_input = water_superficial_velocity_input / liquid_superficial_velocity_input;
                              for (unsigned int id = 0; id < nseg_vars; ++id)
                                {
                                  const double D_jso = D_oil_superficial_velocity_input_D_seg_vars[id];
                                  const double D_jsw = D_water_superficial_velocity_input_D_seg_vars[id];
                                  const double D_jsl = D_jso + D_jsw;
                                  D_oil_flow_weight_input_D_seg_vars[id] =
                                      (D_jso * liquid_superficial_velocity_input - oil_superficial_velocity_input * D_jsl)
                                      / (liquid_superficial_velocity_input * liquid_superficial_velocity_input);
                                  D_water_flow_weight_input_D_seg_vars[id] = -D_oil_flow_weight_input_D_seg_vars[id];
                                }
                            }
                          else
                            {
                              const double alpha_l_seed_local = alpha_o_seed + alpha_w_seed;
                              if (fabs (alpha_l_seed_local) > df_den_eps)
                                {
                                  oil_flow_weight_input = alpha_o_seed / alpha_l_seed_local;
                                  water_flow_weight_input = alpha_w_seed / alpha_l_seed_local;
                                }
                            }

                          std::vector<double> D_C0_partial_D_seg_vars                 (nseg_vars, 0.);
                          std::vector<double> D_drift_velocity_partial_D_seg_vars     (nseg_vars, 0.);
                          std::vector<double> D_gas_phase_velocity_partial_D_seg_vars (nseg_vars, 0.);
                          std::vector<double> D_liquid_phase_velocity_partial_D_seg_vars (nseg_vars, 0.);

                          std::vector<double> D_C0_OW_partial_D_seg_vars                 (nseg_vars, 0.);
                          std::vector<double> D_drift_velocity_OW_partial_D_seg_vars     (nseg_vars, 0.);
                          std::vector<double> D_oil_phase_velocity_partial_D_seg_vars    (nseg_vars, 0.);
                          std::vector<double> D_water_phase_velocity_partial_D_seg_vars  (nseg_vars, 0.);

                          std::vector<double> D_alpha_g_D_seg_vars   (nseg_vars, 0.); // d(alpha_g)/d(seg_var)
                          std::vector<double> D_beta_o_D_seg_vars    (nseg_vars, 0.); // d(beta_o)/d(seg_var), beta_o = alpha_o / (alpha_o + alpha_w)
                          std::vector<double> D_alpha_o_D_seg_vars   (nseg_vars, 0.);
                          std::vector<double> D_alpha_w_D_seg_vars   (nseg_vars, 0.);
                          std::vector<double> D_alpha_l_D_seg_vars   (nseg_vars, 0.);

                          std::vector<double> D_phase_molar_rate_D_seg_var ((1U + mp.nc + 1U) * mp.np, 0.);

                          while (error > 1.e-4 && it < max_it)
                            {
                              temp_holdups.copy_operator (&prev_holdups);
                              new_vels.copy_operator (&prev_vels);

                              // FIX #A: superficial velocities пересчитываются внутри цикла (my_code_12-style).
                              // vs_phase = v_phase · α_phase (лаговая Picard-итерация).
                              // j_iter — актуальная смесевая скорость на этой итерации.
                              const double vsg_iter = prev_vels.gas   * temp_holdups.gas;
                              const double vso_iter = prev_vels.oil   * temp_holdups.oil;
                              const double vsw_iter = prev_vels.water * temp_holdups.water;
                              const double j_iter   = vsg_iter + vso_iter + vsw_iter;

                              std::vector <double> D_gas_liq_interfacial_tension_D_seg_vars = {};
                              D_gas_liq_interfacial_tension_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_gas_liq_interfacial_tension_D_seg_vars.begin(), D_gas_liq_interfacial_tension_D_seg_vars.end(), 0.);

                              std::vector <double> D_liq_density_D_seg_vars = {};
                              D_liq_density_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_liq_density_D_seg_vars.begin(), D_liq_density_D_seg_vars.end(), 0.);

                              std::vector <double> D_bubble_rise_velocity_D_seg_vars = {};
                              D_bubble_rise_velocity_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_bubble_rise_velocity_D_seg_vars.begin(), D_bubble_rise_velocity_D_seg_vars.end(), 0.);

                              std::vector <double> D_diametr_dimless_D_seg_vars = {};
                              D_diametr_dimless_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_diametr_dimless_D_seg_vars.begin(), D_diametr_dimless_D_seg_vars.end(), 0.);

                              std::vector <double> D_Kut_number_D_seg_vars = {};
                              D_Kut_number_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_Kut_number_D_seg_vars.begin(), D_Kut_number_D_seg_vars.end(), 0.);

                              std::vector <double> D_flooding_velocity_D_seg_vars = {};
                              D_flooding_velocity_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_flooding_velocity_D_seg_vars.begin(), D_flooding_velocity_D_seg_vars.end(), 0.);

                              std::vector <double> D_ksi_D_seg_vars = {};
                              D_ksi_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_ksi_D_seg_vars.begin(), D_ksi_D_seg_vars.end(), 0.);

                              std::vector <double> D_eta_D_seg_vars = {};
                              D_eta_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_eta_D_seg_vars.begin(), D_eta_D_seg_vars.end(), 0.);

                              std::vector <double> D_C0_D_seg_vars = {};
                              D_C0_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_C0_D_seg_vars.begin(), D_C0_D_seg_vars.end(), 0.);

                              std::vector <double> D_K_g_D_seg_vars = {};
                              D_K_g_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_K_g_D_seg_vars.begin(), D_K_g_D_seg_vars.end(), 0.);

                              std::vector <double> D_drift_velocity_D_seg_vars = {};
                              D_drift_velocity_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_drift_velocity_D_seg_vars.begin(), D_drift_velocity_D_seg_vars.end(), 0.);

                              std::vector <double> D_gas_phase_velocity_D_seg_vars = {};
                              D_gas_phase_velocity_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_gas_phase_velocity_D_seg_vars.begin(), D_gas_phase_velocity_D_seg_vars.end(), 0.);

                              std::vector <double> D_liquid_phase_velocity_D_seg_vars = {};
                              D_liquid_phase_velocity_D_seg_vars.resize (1U + mp.nc + 1U);
                              std::fill (D_liquid_phase_velocity_D_seg_vars.begin(), D_liquid_phase_velocity_D_seg_vars.end(), 0.);

                              std::vector <double> D_phase_molar_rate_D_seg_var = {};
                              D_phase_molar_rate_D_seg_var.resize ((1U + mp.nc + 1U) * mp.np);
                              std::fill (D_phase_molar_rate_D_seg_var.begin(), D_phase_molar_rate_D_seg_var.end(), 0.);

                              double sigma_o = 0.;
                              double sigma_w = 0.;

                              double D_sigma_o_D_p = 0.;
                              double D_sigma_w_D_p = 0.;

                              double oil_api1 = internal_const::API_to_input_density_mult (units_system_t::metric) / (element_status->phase_rho[PHASE_OIL]) - internal_const::API_to_input_density_add ();
                              (void) oil_api1;

                              const double sigma_gl_unused =
                                  surf_mult * pipe_gas_liq_interfacial_tension_holdup_weightening (
                                      element_status->p * converter_metric_to_field.pressure_mult (),
                                      160 /* Fahrenheit. See ecl technical description */,
                                      45.5,
                                      temp_holdups.oil,
                                      temp_holdups.water,
                                      element_status,
                                      D_gas_liq_interfacial_tension_D_seg_vars,
                                      sigma_o,
                                      sigma_w,
                                      D_sigma_o_D_p,
                                      D_sigma_w_D_p);
                              (void) sigma_gl_unused;

                              // Shi gas/liquid stage: use liquid properties averaged with flow(superficial)-weights,
                              // not with current holdups.
                              const double sigma_o_SI = surf_mult * sigma_o;
                              const double sigma_w_SI = surf_mult * sigma_w;
                              double gas_liq_interfacial_tension =
                                  oil_flow_weight_input * sigma_o_SI + water_flow_weight_input * sigma_w_SI;

                              for (unsigned int id = 0; id < 1U + mp.nc; id++)
                                {
                                  const double D_sigma_o_x = (id == 0)
                                      ? (surf_mult * converter_metric_to_field.pressure_mult () * D_sigma_o_D_p)
                                      : 0.0;
                                  const double D_sigma_w_x = (id == 0)
                                      ? (surf_mult * converter_metric_to_field.pressure_mult () * D_sigma_w_D_p)
                                      : 0.0;
                                  D_gas_liq_interfacial_tension_D_seg_vars[id] =
                                      oil_flow_weight_input * D_sigma_o_x
                                      + water_flow_weight_input * D_sigma_w_x
                                      + (sigma_o_SI - sigma_w_SI) * D_oil_flow_weight_input_D_seg_vars[id];
                                }

                              double liq_density =
                                  oil_flow_weight_input * element_status->phase_rho[PHASE_OIL]
                                  + water_flow_weight_input * element_status->phase_rho[PHASE_WATER];
                              double gas_density = element_status->phase_rho[PHASE_GAS];

                              for (unsigned int id = 0; id < 1U + mp.nc; id++)
                                {
                                  D_liq_density_D_seg_vars[id] =
                                      oil_flow_weight_input * element_status->phase_D_rho[id * mp.np + PHASE_OIL]
                                      + water_flow_weight_input * element_status->phase_D_rho[id * mp.np + PHASE_WATER]
                                      + (element_status->phase_rho[PHASE_OIL] - element_status->phase_rho[PHASE_WATER])
                                            * D_oil_flow_weight_input_D_seg_vars[id];
                                }

                               double bubble_rise_velocity = tnav_pow (tnav_div (gas_liq_interfacial_tension * internal_const::grav_metric () * fabs (liq_density - gas_density), liq_density * liq_density), 0.25);

                               for (unsigned int id = 0; id < 1U + mp.nc; id++)
                                 {
                                   D_bubble_rise_velocity_D_seg_vars[id] = 0.25 * tnav_div (1., tnav_pow (bubble_rise_velocity, 3)) *
                                                                            tnav_div ((D_gas_liq_interfacial_tension_D_seg_vars[id] * internal_const::grav_metric () * fabs (liq_density - gas_density) +
                                                                              gas_liq_interfacial_tension * internal_const::grav_metric () * tnav_sgn (liq_density - gas_density) * (D_liq_density_D_seg_vars[id] - element_status->phase_D_rho[id * mp.np + PHASE_GAS])) * liq_density
                                                                              - 2 * gas_liq_interfacial_tension * internal_const::grav_metric () * fabs (liq_density - gas_density) * D_liq_density_D_seg_vars[id],
                                                                            tnav_pow (liq_density, 3));
                                 }

                               double diametr_dimless = 0;

                               if (liq_density - gas_density > 0)
                                 diametr_dimless = sqrt (tnav_div (internal_const::grav_metric () * (liq_density - gas_density), gas_liq_interfacial_tension)) * seg.wsn->pipe_props.diameter;

                               for (unsigned int id = 0; id < 1U+ mp.nc; id++)
                                 {
                                   D_diametr_dimless_D_seg_vars[id] = 0.5 * seg.wsn->pipe_props.diameter * sqrt (tnav_div (gas_liq_interfacial_tension, internal_const::grav_metric () * (liq_density - gas_density))) *
                                                                       tnav_div (internal_const::grav_metric () * tnav_sgn (liq_density - gas_density) * (D_liq_density_D_seg_vars[id] - element_status->phase_D_rho[id * mp.np + PHASE_GAS]) * gas_liq_interfacial_tension -
                                                                         internal_const::grav_metric () * fabs (liq_density - gas_density) * D_gas_liq_interfacial_tension_D_seg_vars[id], tnav_pow (gas_liq_interfacial_tension, 2));
                                 }

                               double linear_interpolation_derivative = 0.;
                               double Kut_number = compute_critical_Kutateladze_number_by_diametr (diametr_dimless, linear_interpolation_derivative);

                               for (unsigned int id = 0; id < 1U + mp.nc; id++)
                                 {
                                     D_Kut_number_D_seg_vars[id] = linear_interpolation_derivative * D_diametr_dimless_D_seg_vars[id];
                                 }

                               //const double flooding_velocity
                               //  = (gas_density > tnm::min_compare_well_limit) ? Kut_number * sqrt (liq_density / gas_density) * bubble_rise_velocity : .0;

                              double flooding_velocity = Kut_number * sqrt (liq_density / gas_density) * bubble_rise_velocity;

                               for (unsigned int id = 0; id < 1U + mp.nc; id++)
                                 {
                                   //if (gas_density > tnm::min_compare_well_limit)
                                   //  {
                                       double first = D_Kut_number_D_seg_vars[id] * sqrt (liq_density / gas_density) * bubble_rise_velocity;
                                       double second = Kut_number * 0.5 * tnav_div (1., tnav_pow (liq_density / gas_density, 0.5)) * bubble_rise_velocity *
                                                                             (D_liq_density_D_seg_vars[id] * gas_density -
                                                                              element_status->phase_D_rho[id * mp.np + PHASE_GAS] * liq_density) / (tnav_pow (gas_density, 2));
                                       double third = Kut_number * sqrt (liq_density / gas_density) * D_bubble_rise_velocity_D_seg_vars[id];
                                       D_flooding_velocity_D_seg_vars[id] = first + second + third;
                                       //D_flooding_velocity_D_seg_vars[id] *= 1000;
                                   //  }
                                   //else
                                   //  D_flooding_velocity_D_seg_vars[id] = 0.;
                                 }


                              double A = 1.2;
                              double B = 0.3;
                              double F_v = 1.;
                              double V_sgf = df_safe_nonzero (flooding_velocity);

                              // FIX #A: смесевая скорость — текущая сумма vs внутри итерации, не вход.
                              const double mixture_velocity = j_iter;

                              double ksi = std::max (temp_holdups.gas, tnav_div (F_v * temp_holdups.gas * fabs (mixture_velocity),
                                                                                                   V_sgf));

                             double eta = (ksi - B) / (1 - B);
                             bool eta_clamped_iter = false;
                             for (unsigned int id = 0; id < 1U + mp.nc + 1U; id++)
                               {
                                 D_eta_D_seg_vars[id] = D_ksi_D_seg_vars[id] / (1 - B);
                               }

                             if (!std::isfinite (eta))
                               {
                                 eta = (ksi > B) ? 1.0 : 0.0;
                                 eta_clamped_iter = true;
                                 std::fill (D_eta_D_seg_vars.begin (), D_eta_D_seg_vars.end (), 0.0);
                               }
                             else if (eta < 0.0)
                               {
                                 eta = 0.0;
                                 eta_clamped_iter = true;
                                 std::fill (D_eta_D_seg_vars.begin (), D_eta_D_seg_vars.end (), 0.0);
                               }
                             else if (eta > 1.0)
                               {
                                 eta = 1.0;
                                 eta_clamped_iter = true;
                                 std::fill (D_eta_D_seg_vars.begin (), D_eta_D_seg_vars.end (), 0.0);
                               }

                             double C_0 = A / (1 + (A - 1) * eta * eta);

                              wsncs->wsn_C_0 = C_0;

                             double K_g_low  = tnav_div (1.53, C_0);
                             double K_g_high = Kut_number;
                             double K_g;

                             if (temp_holdups.gas < 0.2)
                               {
                                 K_g = K_g_low;
                               }
                             else if (temp_holdups.gas > 0.4)
                               {
                                 K_g = K_g_high;
                               }
                             else
                               {
                                 K_g = interpolate_y_against_x (temp_holdups.gas, 0.2, 0.4, K_g_low, K_g_high);
                               }

                             // FIX #C: sqrt охватывает весь знаменатель (Eclipse 8.78, Shi2005 eq.14-15).
                             const double vd_denom_inner =
                                 temp_holdups.gas * C_0 * tnav_div (gas_density, liq_density)
                                 + 1.0 - temp_holdups.gas * C_0;
                             double drift_velocity = tnav_div (
                                 (1.0 - temp_holdups.gas * C_0) * C_0 * K_g * bubble_rise_velocity,
                                 sqrt (std::max (vd_denom_inner, tnm::min_compare)));

                             drift_velocity *= drift_incl_mult;
                             wsncs->wsn_drift_velocity = drift_velocity;

                              double gas_phase_velocity = C_0 * mixture_velocity + drift_velocity;

                              const double one_minus_alpha_iter = df_safe_nonzero (1.0 - temp_holdups.gas);
                              double liquid_phase_velocity = ((1. - temp_holdups.gas * C_0) / one_minus_alpha_iter) * mixture_velocity -
                                                             (temp_holdups.gas / one_minus_alpha_iter) * drift_velocity;

                              // Total derivatives of the CURRENT iteration map state.
                              std::vector<double> D_gas_phase_velocity_total_iter (nseg_vars, 0.0);
                              std::vector<double> D_oil_phase_velocity_total_iter (nseg_vars, 0.0);

                              const double beta_iter = (fabs (temp_holdups.liquid) > df_den_eps)
                                                       ? (temp_holdups.oil / temp_holdups.liquid)
                                                       : 0.0;

                              std::vector<double> D_liquid_phase_velocity_total_iter (nseg_vars, 0.0);
                              for (unsigned int id = 0; id < nseg_vars; ++id)
                                {
                                  const double D_alpha_iter = D_alpha_g_iter[id];
                                  const double D_beta_iter  = D_beta_o_iter[id];
                                  const double D_vm = D_mixture_superficial_velocity_D_seg_vars[id];

                                  const double D_sigma_o_x = (id == 0) ? surf_mult * D_sigma_o_D_p : 0.0;
                                  const double D_sigma_w_x = (id == 0) ? surf_mult * D_sigma_w_D_p : 0.0;
                                  const double D_sigma_gl_total =
                                      beta_iter * D_sigma_o_x
                                      + (1.0 - beta_iter) * D_sigma_w_x
                                      + (sigma_o * surf_mult - sigma_w * surf_mult) * D_beta_iter;

                                  const double D_rho_o_x = (id < 1U + mp.nc) ? element_status->phase_D_rho[id * mp.np + PHASE_OIL] : 0.0;
                                  const double D_rho_w_x = (id < 1U + mp.nc) ? element_status->phase_D_rho[id * mp.np + PHASE_WATER] : 0.0;
                                  const double D_rho_g_x = (id < 1U + mp.nc) ? element_status->phase_D_rho[id * mp.np + PHASE_GAS] : 0.0;

                                  const double D_liq_density_total =
                                      beta_iter * D_rho_o_x
                                      + (1.0 - beta_iter) * D_rho_w_x
                                      + (element_status->phase_rho[PHASE_OIL] - element_status->phase_rho[PHASE_WATER]) * D_beta_iter;

                                  const double delta_rho = liq_density - gas_density;
                                  const double D_delta_rho_total = D_liq_density_total - D_rho_g_x;

                                  double D_bubble_rise_velocity_total = 0.0;
                                  if (bubble_rise_velocity > df_den_eps)
                                    {
                                      const double num = gas_liq_interfacial_tension * internal_const::grav_metric () * fabs (delta_rho);
                                      const double D_num = internal_const::grav_metric ()
                                                          * (D_sigma_gl_total * fabs (delta_rho)
                                                             + gas_liq_interfacial_tension * tnav_sgn (delta_rho) * D_delta_rho_total);
                                      const double den = liq_density * liq_density;
                                      const double D_den = 2.0 * liq_density * D_liq_density_total;
                                      const double D_arg = (D_num * den - num * D_den) / (den * den);
                                      D_bubble_rise_velocity_total = 0.25 * D_arg / tnav_pow (bubble_rise_velocity, 3);
                                    }

                                  double D_diametr_dimless_total = 0.0;
                                  if (diametr_dimless > 0.0 && gas_liq_interfacial_tension > df_den_eps)
                                    {
                                      const double safe_delta = std::max (liq_density - gas_density, df_den_eps);
                                      D_diametr_dimless_total = 0.5 * diametr_dimless
                                          * (D_delta_rho_total / safe_delta - D_sigma_gl_total / gas_liq_interfacial_tension);
                                    }

                                  const double D_Kut_number_total = linear_interpolation_derivative * D_diametr_dimless_total;
                                  const double sqrt_rl_rg = sqrt (tnav_div (liq_density, gas_density));
                                  const double D_sqrt_rl_rg =
                                      0.5 / df_safe_nonzero (sqrt_rl_rg)
                                      * (D_liq_density_total * gas_density - D_rho_g_x * liq_density)
                                      / (gas_density * gas_density);
                                  const double D_flooding_velocity_total =
                                      D_Kut_number_total * sqrt_rl_rg * bubble_rise_velocity
                                      + Kut_number * D_sqrt_rl_rg * bubble_rise_velocity
                                      + Kut_number * sqrt_rl_rg * D_bubble_rise_velocity_total;

                                  const double ksi_alt = F_v * temp_holdups.gas * fabs (mixture_velocity) / df_safe_nonzero (V_sgf);
                                  double D_ksi_total = 0.0;
                                  if (temp_holdups.gas >= ksi_alt)
                                    D_ksi_total = D_alpha_iter;
                                  else
                                    D_ksi_total = F_v * (
                                        D_alpha_iter * fabs (mixture_velocity) / df_safe_nonzero (V_sgf)
                                        + temp_holdups.gas * tnav_sgn (mixture_velocity) * D_vm / df_safe_nonzero (V_sgf)
                                        - temp_holdups.gas * fabs (mixture_velocity) * D_flooding_velocity_total / (df_safe_nonzero (V_sgf) * df_safe_nonzero (V_sgf)));

                                  const double D_eta_total = eta_clamped_iter ? 0.0 : (D_ksi_total / (1.0 - B));
                                  const double D_C0_total =
                                      -A * (A - 1.0) * 2.0 * eta * D_eta_total / tnav_pow (1.0 + (A - 1.0) * eta * eta, 2);

                                  const double D_K_g_low_total = -1.53 * D_C0_total / (C_0 * C_0);
                                  double D_K_g_total = 0.0;
                                  if (temp_holdups.gas < 0.2)
                                    D_K_g_total = D_K_g_low_total;
                                  else if (temp_holdups.gas > 0.4)
                                    D_K_g_total = D_Kut_number_total;
                                  else
                                    {
                                      const double t_kg = (temp_holdups.gas - 0.2) / 0.2;
                                      D_K_g_total = (1.0 - t_kg) * D_K_g_low_total
                                                    + t_kg * D_Kut_number_total
                                                    + (K_g_high - K_g_low) * D_alpha_iter / 0.2;
                                    }

                                  // FIX #C: vd = num_vd / sqrt(S), где S = αg·C0·(ρg/ρl) + 1 − αg·C0.
                                  const double rg_rl = tnav_div (gas_density, liq_density);
                                  const double D_rg_rl =
                                      (D_rho_g_x * liq_density - gas_density * D_liq_density_total)
                                      * tnav_div (1.0, liq_density * liq_density);
                                  const double num_vd = (1.0 - temp_holdups.gas * C_0) * C_0 * K_g * bubble_rise_velocity;
                                  const double D_num_vd =
                                      (-D_alpha_iter * C_0 - temp_holdups.gas * D_C0_total) * C_0 * K_g * bubble_rise_velocity
                                      + (1.0 - temp_holdups.gas * C_0) * D_C0_total * K_g * bubble_rise_velocity
                                      + (1.0 - temp_holdups.gas * C_0) * C_0 * D_K_g_total * bubble_rise_velocity
                                      + (1.0 - temp_holdups.gas * C_0) * C_0 * K_g * D_bubble_rise_velocity_total;
                                  const double S_vd = std::max (temp_holdups.gas * C_0 * rg_rl + 1.0 - temp_holdups.gas * C_0, tnm::min_compare);
                                  const double D_S_vd =
                                      (D_alpha_iter * C_0 + temp_holdups.gas * D_C0_total) * (rg_rl - 1.0)
                                      + temp_holdups.gas * C_0 * D_rg_rl;
                                  const double sqrt_S_vd = sqrt (S_vd);
                                  const double D_drift_velocity_total =
                                      drift_incl_mult
                                      * (D_num_vd / sqrt_S_vd - 0.5 * num_vd * D_S_vd / (S_vd * sqrt_S_vd));

                                  D_gas_phase_velocity_total_iter[id] = D_C0_total * mixture_velocity + C_0 * D_vm + D_drift_velocity_total;

                                  const double f_gl = (1.0 - temp_holdups.gas * C_0) / one_minus_alpha_iter;
                                  const double g_gl = temp_holdups.gas / one_minus_alpha_iter;
                                  const double D_f_gl_total =
                                      ((-D_alpha_iter * C_0 - temp_holdups.gas * D_C0_total) * one_minus_alpha_iter
                                       + (1.0 - temp_holdups.gas * C_0) * D_alpha_iter)
                                      / (one_minus_alpha_iter * one_minus_alpha_iter);
                                  const double D_g_gl_total = D_alpha_iter / (one_minus_alpha_iter * one_minus_alpha_iter);

                                  D_liquid_phase_velocity_total_iter[id] =
                                      D_f_gl_total * mixture_velocity + f_gl * D_vm
                                      - D_g_gl_total * drift_velocity - g_gl * D_drift_velocity_total;
                                }

                              new_vels.gas = gas_phase_velocity;
                              new_vels.liquid = liquid_phase_velocity;

                              // FIX #A: α_g обновляется из текущего vs_g (пересчитанного внутри цикла).
                              const double raw_alpha_g_new = vsg_iter / df_safe_nonzero (new_vels.gas);
                              new_holdups.gas = raw_alpha_g_new;
                              new_holdups.liquid = 1.0 - new_holdups.gas;

                              double oil_frac_in_liquid = (fabs (temp_holdups.liquid) > df_den_eps)
                                                          ? (temp_holdups.oil / temp_holdups.liquid)
                                                          : 0.0;
                              double water_frac_in_liquid = 1.0 - oil_frac_in_liquid;

                                   double wsn_C0_OW = 0.;
                                   double B1_OW = 0.4;
                                   double B2_OW = 0.7;
                                   double A_OW = 1.2;

                                       if (oil_frac_in_liquid < B1_OW)
                                         {
                                           wsn_C0_OW = A_OW;
                                         }
                                       else if (oil_frac_in_liquid > B2_OW)
                                         {
                                           wsn_C0_OW = 1.;
                                         }
                                       else
                                         {
                                           wsn_C0_OW = interpolate_y_against_x (oil_frac_in_liquid, B1_OW, B2_OW, A_OW, 1.);
                                         }

                                       double oil_api = internal_const::API_to_input_density_mult (units_system_t::metric) / (element_status->phase_rho[PHASE_OIL]) - internal_const::API_to_input_density_add ();
                                       (void) oil_api;

                                       double D_gas_oil_interfacial_tension_D_p = 0.;
                                       double gas_oil_interfacial_tension
                                           = internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT () *
                                             pipe_gas_oil_interfacial_tension_and_deriv (45.5 , element_status->p * converter_metric_to_field.pressure_mult (), 160., D_gas_oil_interfacial_tension_D_p);

                                       double D_gas_wat_interfacial_tension_D_p = 0.;
                                       double gas_wat_interfacial_tension
                                           = internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT () * pipe_gas_wat_interfacial_tension_and_deriv (element_status->p * converter_metric_to_field.pressure_mult (), 160., D_gas_wat_interfacial_tension_D_p);

                                       // FIX #D: σow = |σgo − σgw| без взвешивания по фракциям.
                                       double wat_oil_interfacial_tension = fabs (gas_oil_interfacial_tension - gas_wat_interfacial_tension);

                                       double bubble_rise_velocity_OW = tnav_pow (wat_oil_interfacial_tension * internal_const::grav_metric () *
                                                                                   (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL]) /
                                                                                       (element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER]), 0.25);

                                       double drift_velocity_OW = 1.53 * bubble_rise_velocity_OW * tnav_pow (1 - oil_frac_in_liquid, 2);
                                   drift_velocity_OW *= drift_incl_mult;

                                  wsncs->wsn_C_0_OW = wsn_C0_OW;
                                  wsncs->wsn_drift_velocity_OW = drift_velocity_OW;

                                  double oil_phase_velocity = wsn_C0_OW * liquid_phase_velocity + drift_velocity_OW;

                                  double water_phase_velocity = tnav_div (1. - oil_frac_in_liquid * wsn_C0_OW, 1. - oil_frac_in_liquid) * liquid_phase_velocity -
                                                                tnav_div (oil_frac_in_liquid, 1. - oil_frac_in_liquid) * drift_velocity_OW;

                              new_vels.oil = oil_phase_velocity;
                              new_vels.water = water_phase_velocity;

                              for (unsigned int id = 0; id < nseg_vars; ++id)
                                {
                                  const double D_beta_iter = D_beta_o_iter[id];
                                  const double D_sigma_go_x = (id == 0) ? surf_mult * D_gas_oil_interfacial_tension_D_p : 0.0;
                                  const double D_sigma_gw_x = (id == 0) ? surf_mult * D_gas_wat_interfacial_tension_D_p : 0.0;
                                  double D_C0_OW_total = 0.0;
                                  if (oil_frac_in_liquid > B1_OW && oil_frac_in_liquid < B2_OW)
                                    D_C0_OW_total = -(A_OW - 1.0) * D_beta_iter / (B2_OW - B1_OW);

                                  // FIX #D: производная от |σgo − σgw|.
                                  const double arg_ow = gas_oil_interfacial_tension - gas_wat_interfacial_tension;
                                  const double D_sigma_ow_total =
                                      tnav_sgn (arg_ow) * (D_sigma_go_x - D_sigma_gw_x);

                                  double D_bubble_rise_velocity_OW_total = 0.0;
                                  if (bubble_rise_velocity_OW > df_den_eps)
                                    {
                                      const double delta_ow = element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL];
                                      const double D_rho_w_x = (id < 1U + mp.nc) ? element_status->phase_D_rho[id * mp.np + PHASE_WATER] : 0.0;
                                      const double D_rho_o_x = (id < 1U + mp.nc) ? element_status->phase_D_rho[id * mp.np + PHASE_OIL] : 0.0;
                                      const double D_arg_vcow = internal_const::grav_metric () * (
                                          D_sigma_ow_total * delta_ow / (element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER])
                                          + wat_oil_interfacial_tension * ((D_rho_w_x - D_rho_o_x)
                                              / (element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER])
                                              - delta_ow * 2.0 * D_rho_w_x / tnav_pow (element_status->phase_rho[PHASE_WATER], 3)));
                                      D_bubble_rise_velocity_OW_total = 0.25 * D_arg_vcow / tnav_pow (bubble_rise_velocity_OW, 3);
                                    }

                                  const double D_drift_velocity_OW_total =
                                      drift_incl_mult * (1.53 * D_bubble_rise_velocity_OW_total * tnav_pow (1.0 - oil_frac_in_liquid, 2)
                                      - 1.53 * bubble_rise_velocity_OW * 2.0 * (1.0 - oil_frac_in_liquid) * D_beta_iter);

                                  D_oil_phase_velocity_total_iter[id] =
                                      D_C0_OW_total * liquid_phase_velocity
                                      + wsn_C0_OW * D_liquid_phase_velocity_total_iter[id]
                                      + D_drift_velocity_OW_total;
                                }

                              // FIX #A: β_o обновляется из текущего vs_o.
                              const double raw_beta_o = vso_iter
                                                        / df_safe_nonzero (new_holdups.liquid * new_vels.oil);
                              double new_beta_o = raw_beta_o;

                              if (new_beta_o > 1.)
                                new_beta_o = 1.;
                              else if (new_beta_o < 0.)
                                new_beta_o = 0.;

                              new_holdups.oil   = new_beta_o * new_holdups.liquid;
                              new_holdups.water = (1.0 - new_beta_o) * new_holdups.liquid;

                              for (unsigned int id = 0; id < nseg_vars; ++id)
                                {
                                  const double D_sg_in = D_gas_superficial_velocity_input_D_seg_vars[id];
                                  const double D_so_in = D_oil_superficial_velocity_input_D_seg_vars[id];

                                  D_alpha_g_next[id] =
                                      (D_sg_in * new_vels.gas - gas_superficial_velocity_input * D_gas_phase_velocity_total_iter[id])
                                      / (new_vels.gas * new_vels.gas);

                                  const double D_alpha_l_next = -D_alpha_g_next[id];
                                  const double beta_den = df_safe_nonzero (new_holdups.liquid * new_vels.oil);
                                  const double D_beta_den = D_alpha_l_next * new_vels.oil + new_holdups.liquid * D_oil_phase_velocity_total_iter[id];

                                  D_beta_o_next[id] =
                                      (D_so_in * beta_den - oil_superficial_velocity_input * D_beta_den)
                                      / (beta_den * beta_den);
                                }

                              // FIX #A: при vs = prev·temp невязка α·v − vs_input тождественно мала,
                              // поэтому критерий сходимости — только изменение holdups между итерациями.
                              const double iter_delta =
                                  sqrt (tnav_sqr (prev_holdups.oil - new_holdups.oil)
                                        + tnav_sqr (prev_holdups.water - new_holdups.water)
                                        + tnav_sqr (prev_holdups.gas - new_holdups.gas));

                              error = iter_delta;

                              prev_holdups.copy_operator (&new_holdups);
                              prev_vels.copy_operator (&new_vels);
                              D_alpha_g_iter.swap (D_alpha_g_next);
                              D_beta_o_iter.swap (D_beta_o_next);
                              it++;

                          PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS, "MERGEN: Well {}: it = {}, Seg_idx = {}, error = {}, average_volumetric_mixture_velocity = {}, element_status->avg_xi = {}, "
                                                                           "seg.wsn->pipe_props.area = {}, wsncs->wsn_mixture_molar_rate = {}, wsncs->wsn_C_0 = {}, wsncs->wsn_drift_velocity = {}, "
                                                                           "gas_liq_interfacial_tension = {}, gas_density = {}, liquid_density = {}, diametr_dimless = {}, bubble_rise_velocity = {}, ksi = {}, "
                                                                           "Kut_number = {}, flooding_velocity = {}, K_g = {}, element_status->phase_S[PHASE_GAS] = {}, "
                                                                           "sqrt (tnav_div (internal_const::grav_metric () * fabs (liq_density - gas_density), gas_liq_interfacial_tension)) = {}, "
                                                                           "seg.wsn->pipe_props.diameter = {}, phase_S_WAT = {}, phase_S_OIL = {}, phase_S_GAS = {}, "
                                                                           "wsncs->wsn_C_0_OW = {}, wsncs->wsn_drift_velocity_OW = {}, "
                                                                           "new_holdups.gas = {}, new_holdups.liquid = {}, new_holdups.oil = {}, new_holdups.water = {} \n",
                                                 wcb_wis->get_well_name (), it, seg.wsn->wsn_index, error, average_volumetric_mixture_velocity, element_status->avg_xi,
                                                 seg.wsn->pipe_props.area, wsncs->wsn_mixture_molar_rate, wsncs->wsn_C_0, wsncs->wsn_drift_velocity,
                                                 gas_liq_interfacial_tension, gas_density, liq_density, diametr_dimless, bubble_rise_velocity, ksi,
                                                 Kut_number, flooding_velocity, K_g, element_status->phase_S[PHASE_GAS],
                                                 sqrt (tnav_div (internal_const::grav_metric () * fabs (liq_density - gas_density), gas_liq_interfacial_tension)),
                                                 seg.wsn->pipe_props.diameter, element_status->phase_S[PHASE_WATER], element_status->phase_S[PHASE_OIL], element_status->phase_S[PHASE_GAS],
                                                 wsncs->wsn_C_0_OW, wsncs->wsn_drift_velocity_OW, new_holdups.gas, new_holdups.liquid, new_holdups.oil, new_holdups.water);
                          const double dbg_Rg_iter = prev_holdups.gas * prev_vels.gas - gas_superficial_velocity_input;
                          const double dbg_Ro_iter = prev_holdups.oil * prev_vels.oil - oil_superficial_velocity_input;
                          const double dbg_Rw_iter = prev_holdups.water * prev_vels.water - water_superficial_velocity_input;
                          PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                               "MERGEN: Well {}: Seg_idx = {}, local residuals: Rg = {}, Ro = {}, Rw = {}\n",
                               wcb_wis->get_well_name (), seg.wsn->wsn_index, dbg_Rg_iter, dbg_Ro_iter, dbg_Rw_iter);
                            }

                        for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                          {
                            wsncs->wsn_component_rate[ic] = 0.0;
                            for (unsigned int id = 0; id < nseg_vars; ++id)
                              wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id] = 0.0;
                          }

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

                      // Persist actual DF holdups as segment-state seed for the next outer iteration.
                      wsncs->phase_S[PHASE_GAS] = alpha_g;
                      wsncs->phase_S[PHASE_OIL] = alpha_o;
                      wsncs->phase_S[PHASE_WATER] = alpha_w;

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
                                 "Seg {} var {}: alpha_l ana={}, num={}, relerr={}\n",
                                 seg.wsn->wsn_index, fd_var_id,
                                 alpha_l, liquid_hp_num, fabs (alpha_l - liquid_hp_num));

                        // Convert to N/m.
                        double sigma_o = surf_tension_mult * sigma_o_raw;
                        double sigma_w = surf_tension_mult * sigma_w_raw;

                        // Gas/liquid stage in Shi is built on a *combined liquid phase* with
                        // flow-weighted average liquid properties. These weights are defined by
                        // the fixed superficial liquid inputs used in the local DF solve, not by
                        // the solved oil fraction in liquid beta_o.
                        double gas_liq_interfacial_tension =
                            oil_flow_weight_input * sigma_o + water_flow_weight_input * sigma_w;

                        std::fill (D_gas_liq_interfacial_tension_partial_D_seg_vars.begin (),
                                   D_gas_liq_interfacial_tension_partial_D_seg_vars.end (), 0.0);
                        for (unsigned int id = 0; id < 1U + mp.nc; ++id)
                          {
                            const double D_sigma_o_x = (id == 0) ? (surf_tension_mult * D_sigma_o_D_p_psi) : 0.0;
                            const double D_sigma_w_x = (id == 0) ? (surf_tension_mult * D_sigma_w_D_p_psi) : 0.0;
                            D_gas_liq_interfacial_tension_partial_D_seg_vars[id] =
                                oil_flow_weight_input * D_sigma_o_x
                                + water_flow_weight_input * D_sigma_w_x
                                + (sigma_o - sigma_w) * D_oil_flow_weight_input_D_seg_vars[id];
                          }
                        PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                             "Seg {} var {}: D_gas_liq_interfacial_tension_partial_D_seg_vars[0] ana={}, num={}, relerr={}\n",
                             seg.wsn->wsn_index, fd_var_id,
                             D_gas_liq_interfacial_tension_partial_D_seg_vars[0],
                             gas_liq_interfacial_tension_numerical,
                             fabs (D_gas_liq_interfacial_tension_partial_D_seg_vars[0] - gas_liq_interfacial_tension_numerical));

                        // At the gas/liquid stage the liquid properties are fixed by the
                        // superficial oil/water inputs, so there is no dependence on beta_o.
                        const double D_gas_liq_interfacial_tension_D_beta_o = 0.0;

                        // Combined liquid density for gas/liquid stage: flow-weighted, not
                        // holdup-weighted.
                        const double liq_density =
                            oil_flow_weight_input * element_status->phase_rho[PHASE_OIL]
                            + water_flow_weight_input * element_status->phase_rho[PHASE_WATER];

                        const double gas_density = element_status->phase_rho[PHASE_GAS];

                        for (unsigned int id = 0; id < 1U + mp.nc; ++id)
                          {
                            D_liq_density_partial_D_seg_vars[id] =
                                oil_flow_weight_input * element_status->phase_D_rho[id * mp.np + PHASE_OIL]
                                + water_flow_weight_input * element_status->phase_D_rho[id * mp.np + PHASE_WATER]
                                + (element_status->phase_rho[PHASE_OIL] - element_status->phase_rho[PHASE_WATER])
                                      * D_oil_flow_weight_input_D_seg_vars[id];
                          }

                        const double D_liq_density_D_beta_o = 0.0;

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

                      wsncs->wsn_C_0 = C_0;

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
                      wsncs->wsn_drift_velocity = drift_velocity;

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

                      wsncs->wsn_C_0_OW = wsn_C0_OW;
                      wsncs->wsn_drift_velocity_OW = drift_velocity_OW;

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

                      wsncs->wsn_C_0_OW = wsn_C0_OW;
                      wsncs->wsn_drift_velocity_OW = drift_velocity_OW;

                      // -----------------------------
                      // Derivatives of holdups from the IMPLICIT local DF system.
                      // Use the same 2x2 IFT system that defines the converged local solve:
                      //   Rg(alpha_g, beta_o, x) = alpha_g * Vg - Vsg_in = 0
                      //   Ro(alpha_g, beta_o, x) = alpha_l * beta_o * Vo - Vso_in = 0
                      // This is more consistent for the outer Newton than derivatives of the
                      // fixed-point map, especially when the inner simple iteration is only
                      // approximately converged.
                      // -----------------------------
                      const double J11 =
                          gas_phase_velocity + alpha_g * D_gas_phase_velocity_D_alpha_g;

                      const double J12 =
                          alpha_g * D_gas_phase_velocity_D_beta_o;

                      const double J21 =
                          -beta_o * oil_phase_velocity
                          + alpha_l * beta_o * D_oil_phase_velocity_D_alpha_g;

                      const double J22 =
                          alpha_l * (oil_phase_velocity + beta_o * D_oil_phase_velocity_D_beta_o);

                      const double detJ = J11 * J22 - J12 * J21;

                      if (fabs (detJ) < tnm::min_compare)
                        {
                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              D_alpha_g_D_seg_vars[id] = 0.0;
                              D_beta_o_D_seg_vars[id] = 0.0;
                            }
                        }
                      else
                        {
                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              const double Rg_x =
                                  alpha_g * D_gas_phase_velocity_partial_D_seg_vars[id]
                                  - D_gas_superficial_velocity_input_D_seg_vars[id];

                              const double Ro_x =
                                  alpha_l * beta_o * D_oil_phase_velocity_partial_D_seg_vars[id]
                                  - D_oil_superficial_velocity_input_D_seg_vars[id];

                              D_alpha_g_D_seg_vars[id] =
                                  tnav_div (-J22 * Rg_x + J12 * Ro_x, detJ);

                              D_beta_o_D_seg_vars[id] =
                                  tnav_div ( J21 * Rg_x - J11 * Ro_x, detJ);
                            }
                        }

                            // total holdup derivatives
                            for (unsigned int id = 0; id < nseg_vars; ++id)
                              {
                                D_alpha_l_D_seg_vars[id] = -D_alpha_g_D_seg_vars[id];
                                D_alpha_o_D_seg_vars[id] = -beta_o * D_alpha_g_D_seg_vars[id] + alpha_l * D_beta_o_D_seg_vars[id];
                                D_alpha_w_D_seg_vars[id] = -(1. - beta_o) * D_alpha_g_D_seg_vars[id] - alpha_l * D_beta_o_D_seg_vars[id];
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
                        }

                      // ============================================================================
                      // COMPONENT RATES MUST BE COMPUTED ONCE, AFTER FIXED-POINT
                      // ============================================================================

                          std::vector<double> D_rho_avg_D_seg_vars (nseg_vars, 0.0);
                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            {
                              const double D_rho_g = (id < 1U + mp.nc) ? element_status->phase_D_rho[id * mp.np + PHASE_GAS]   : 0.0;
                              const double D_rho_o = (id < 1U + mp.nc) ? element_status->phase_D_rho[id * mp.np + PHASE_OIL]   : 0.0;
                              const double D_rho_w = (id < 1U + mp.nc) ? element_status->phase_D_rho[id * mp.np + PHASE_WATER] : 0.0;

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

                          PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                                 "Seg {} var {}: alpha_water = {}, alpha_oil = {}, alpha_gas = {}, alpha_liq = {}, \n"
                                 "water_velo = {}, oil_velo = {}, gas_velo = {}, liquid_velo = {}, beta_o = {}, \n"
                                 "D_alpha_w_D_seg_vars[0] = {}, D_alpha_w_D_seg_vars[1] = {}, D_alpha_w_D_seg_vars[2] = {},"
                                 "D_alpha_w_D_seg_vars[3] = {}, D_alpha_w_D_seg_vars[4] = {}, inner_error = {}, rho_avg = {} \n",
                                 seg.wsn->wsn_index, fd_var_id, alpha_w, alpha_o, alpha_g, alpha_l,
                                 water_phase_velocity, oil_phase_velocity, gas_phase_velocity, liquid_phase_velocity, beta_o,
                                 D_alpha_w_D_seg_vars[0], D_alpha_w_D_seg_vars[1], D_alpha_w_D_seg_vars[2], D_alpha_w_D_seg_vars[3],
                                 D_alpha_w_D_seg_vars[4], error, wsncs->rho_avg_DF);

                      for (auto ic = mp.nc0; ic < mp.nc; ++ic)
                        {
                          wsncs->wsn_component_rate[ic] = 0.;
                          for (unsigned int id = 0; id < nseg_vars; ++id)
                            wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id] = 0.;
                        }

                      const double area_day = seg.wsn->pipe_props.area; // same units as phase velocities in this block
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

        const bool same_active_set = (prev_active_flags == next_active_flags);
        const bool inner_converged =
            (fabs (prev_R_g) < 1.e-4 && fabs (prev_R_o) < 1.e-4 &&
             fabs (next_R_g) < 1.e-4 && fabs (next_R_o) < 1.e-4);

        if ((same_active_set && inner_converged && seg.wsn->wsn_index != TOP_SEG_INDEX) || true)
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

            // -------------------- IFT defect check
            double Rg_x_dbg =
                alpha_g * D_gas_phase_velocity_partial_D_seg_vars[fd_var_id]
                - D_gas_superficial_velocity_input_D_seg_vars[fd_var_id];

            double Ro_x_dbg =
                alpha_l * beta_o * D_oil_phase_velocity_partial_D_seg_vars[fd_var_id]
                - D_oil_superficial_velocity_input_D_seg_vars[fd_var_id];

            const double J11_dbg =
                gas_phase_velocity + alpha_g * D_gas_phase_velocity_D_alpha_g;

            const double J12_dbg =
                alpha_g * D_gas_phase_velocity_D_beta_o;

            const double J21_dbg =
                -beta_o * oil_phase_velocity
                + alpha_l * beta_o * D_oil_phase_velocity_D_alpha_g;

            const double J22_dbg =
                alpha_l * (oil_phase_velocity + beta_o * D_oil_phase_velocity_D_beta_o);

            double defect_Rg_ana =
                J11_dbg * D_alpha_g_D_seg_vars[fd_var_id]
                + J12_dbg * D_beta_o_D_seg_vars[fd_var_id]
                + Rg_x_dbg;

            double defect_Ro_ana =
                J21_dbg * D_alpha_g_D_seg_vars[fd_var_id]
                + J22_dbg * D_beta_o_D_seg_vars[fd_var_id]
                + Ro_x_dbg;

            double defect_Rg_num =
                J11_dbg * alpha_g_num_derivative
                + J12_dbg * beta_o_num_derivative
                + Rg_x_dbg;

            double defect_Ro_num =
                J21_dbg * alpha_g_num_derivative
                + J22_dbg * beta_o_num_derivative
                + Ro_x_dbg;

            // -------------------- prints
            PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                 "Seg {} var {}: alpha_g ana={}, num={}, relerr={}, beta_o ana={}, num={}, relerr={}\n",
                 seg.wsn->wsn_index, fd_var_id,
                 D_alpha_g_D_seg_vars[fd_var_id], alpha_g_num_derivative,
                 rel_err_dbg (D_alpha_g_D_seg_vars[fd_var_id], alpha_g_num_derivative),
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
                  return std::string ("z[") + std::to_string (vid - 1) + "]";
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
                wsncs_dbg_final_state.restore (wsncs, mp);
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

                fully_implicit_element_status es_next_all (*element_status);
                fully_implicit_element_status *es_next_all_ptr = &es_next_all;
                copy_segment_params_to_element_status (seg, es_next_all_ptr);
                wsncs_dbg_final_state.restore (wsncs, mp);
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

                wsncs_dbg_final_state.restore (wsncs, mp);
                wsncs->wsn_mixture_molar_rate = q_tot_buf;

                const double d_alpha_num = (na_ag - pa_ag) / (2.0 * eps_dbg);
                const double d_beta_num = (na_bo - pa_bo) / (2.0 * eps_dbg);
                const double d_rho_num = (rho_next_all - rho_prev_all) / (2.0 * eps_dbg);
                const double d_c0_num = (wsncs->wsn_C_0 - wsncs->wsn_C_0) ;
                (void) d_c0_num;

                PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
                     "Seg {} ALLVARS {}: alpha_g a={} n={} r={} | beta_o a={} n={} r={} | rho_avg a={} n={} r={}\n",
                     seg.wsn->wsn_index, fd_var_name_dbg (dbg_vid),
                     D_alpha_g_D_seg_vars[dbg_vid], d_alpha_num, rel_err_dbg (D_alpha_g_D_seg_vars[dbg_vid], d_alpha_num),
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
  // ---------------------------------------------------------------------------
  // VALUE-ONLY helper used for numerical derivative checks.
  // This function must follow the SAME value path as the main DF calculation:
  //   flash -> local DF simple iteration -> final post-processing values.
  // No analytical derivatives are assembled here.
  // ---------------------------------------------------------------------------

  // -------------------- initialize scalar outputs
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
  sigma_o = 0.0; // raw dynes/cm
  sigma_w = 0.0; // raw dynes/cm
  oil_inter_tension_buf = 0.0; // N/m
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

  // -------------------- same pre-processing as in the main calculation
  if (wsncs->wsn_flow_dir == segment_flow_direction_t::from_parent_to_child)
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

  const double q_tot_local = wsncs->wsn_mixture_molar_rate;

  fill_wsncs_from_element_status (wsncs, element_status, mp);
  wsncs->wsn_mixture_molar_rate = q_tot_local;

  const double area = seg.wsn->pipe_props.area;
  const double diameter = seg.wsn->pipe_props.diameter;
  const double surf_mult = internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT ();
  //const double p_to_psi_mult = converter_metric_to_field.pressure_mult ();

  average_volumetric_velocity = tnav_div (wsncs->wsn_mixture_molar_rate, area * element_status->avg_xi);
  avg_xi = element_status->avg_xi;
  xi = element_status->phase_xi[PHASE_GAS];

  const double mixture_superficial_velocity =
      average_volumetric_velocity; /// internal_const::DAYS_TO_SEC (); // [m/s]

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

  // -------------------- initialize component rates/mass rate every call
  wsncs->wsn_mmw = 0.0;
  wsncs->wsn_mixture_mass_rate = 0.0;
  for (auto ic = mp.nc0; ic < mp.nc; ++ic)
    wsncs->wsn_component_rate[ic] = 0.0;

  // -------------------- TOP segment: no DF
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

  // -------------------- superficial inputs used in local DF iteration
  const double prev_alpha_g_state_raw = std::isfinite (wsncs->phase_S[PHASE_GAS])   ? wsncs->phase_S[PHASE_GAS]   : 0.0;
  const double prev_alpha_o_state_raw = std::isfinite (wsncs->phase_S[PHASE_OIL])   ? wsncs->phase_S[PHASE_OIL]   : 0.0;
  const double prev_alpha_w_state_raw = std::isfinite (wsncs->phase_S[PHASE_WATER]) ? wsncs->phase_S[PHASE_WATER] : 0.0;

  bool use_previous_df_holdup_seed = false;
  double prev_alpha_g_state = 0.0;
  double prev_alpha_o_state = 0.0;
  double prev_alpha_w_state = 0.0;
  {
    const double prev_sum_raw = prev_alpha_g_state_raw + prev_alpha_o_state_raw + prev_alpha_w_state_raw;
    if (std::isfinite (prev_sum_raw) && prev_sum_raw > 1.0e-12 &&
        prev_alpha_g_state_raw >= 0.0 && prev_alpha_o_state_raw >= 0.0 && prev_alpha_w_state_raw >= 0.0)
      {
        use_previous_df_holdup_seed = true;
        prev_alpha_g_state = prev_alpha_g_state_raw / prev_sum_raw;
        prev_alpha_o_state = prev_alpha_o_state_raw / prev_sum_raw;
        prev_alpha_w_state = prev_alpha_w_state_raw / prev_sum_raw;
      }
  }

  // FIX #B: equal-split seed вместо phase_S.
  const double alpha_g_init = use_previous_df_holdup_seed ? prev_alpha_g_state : (1.0 / 3.0);
  const double alpha_o_init = use_previous_df_holdup_seed ? prev_alpha_o_state : (1.0 / 3.0);
  const double alpha_w_init = use_previous_df_holdup_seed ? prev_alpha_w_state : (1.0 / 3.0);
  const double alpha_l_init = alpha_o_init + alpha_w_init;

  double beta_o_init = 0.0;
  if (fabs (alpha_l_init) > df_den_eps)
    beta_o_init = alpha_o_init / alpha_l_init;

  double avg_xi_seed_input = 0.0;
  if (use_previous_df_holdup_seed)
    {
      for (unsigned int ip = 0; ip < mp.np; ++ip)
        avg_xi_seed_input += wsncs->phase_S[ip] * wsncs->phase_xi[ip];
    }

  double mixture_superficial_velocity_seed = mixture_superficial_velocity;
  if (use_previous_df_holdup_seed && fabs (avg_xi_seed_input) > df_den_eps)
    mixture_superficial_velocity_seed = wsncs->wsn_mixture_molar_rate / (area * avg_xi_seed_input);

  double gas_phase_velocity_seed = mixture_superficial_velocity_seed;
  double liquid_phase_velocity_seed = mixture_superficial_velocity_seed;
  if (use_previous_df_holdup_seed)
    {
      gas_phase_velocity_seed = wsncs->wsn_C_0 * mixture_superficial_velocity_seed + wsncs->wsn_drift_velocity;
      if (fabs (1.0 - alpha_g_init) > df_den_eps)
        {
          liquid_phase_velocity_seed =
              ((1.0 - alpha_g_init * wsncs->wsn_C_0) / (1.0 - alpha_g_init)) * mixture_superficial_velocity_seed
              - (alpha_g_init / (1.0 - alpha_g_init)) * wsncs->wsn_drift_velocity;
        }
    }

  double oil_phase_velocity_seed = liquid_phase_velocity_seed;
  double water_phase_velocity_seed = liquid_phase_velocity_seed;
  if (use_previous_df_holdup_seed)
    {
      oil_phase_velocity_seed = wsncs->wsn_C_0_OW * liquid_phase_velocity_seed + wsncs->wsn_drift_velocity_OW;
      if (fabs (1.0 - beta_o_init) > df_den_eps)
        {
          water_phase_velocity_seed =
              ((1.0 - beta_o_init * wsncs->wsn_C_0_OW) / (1.0 - beta_o_init)) * liquid_phase_velocity_seed
              - (beta_o_init / (1.0 - beta_o_init)) * wsncs->wsn_drift_velocity_OW;
        }
      else
        {
          water_phase_velocity_seed = liquid_phase_velocity_seed;
        }
    }

  // FIX #A: vs = prev_vels * temp_holdups теперь пересчитывается внутри цикла.
  // Начальный oil_flow_weight берём из seed holdups (обновится внутри цикла).
  double oil_flow_weight_input = 0.5;
  double water_flow_weight_input = 0.5;
  if (fabs (alpha_l_init) > df_den_eps)
    {
      oil_flow_weight_input = alpha_o_init / alpha_l_init;
      water_flow_weight_input = alpha_w_init / alpha_l_init;
    }

  // ---------------------------------------------------------------------------
  // Local simple iteration for holdups (same stopping policy as main value-path)
  // ---------------------------------------------------------------------------
  double error = 100.0;
  int it = 0;
  const int max_it = 10;

  phase_holdups_DF prev_holdups (alpha_g_init,
                                 alpha_l_init,
                                 alpha_o_init,
                                 alpha_w_init);
  phase_holdups_DF temp_holdups (0., 0., 0., 0.);
  phase_holdups_DF new_holdups (0., 0., 0., 0.);

  phase_vel_DF prev_vels (gas_phase_velocity_seed,
                          liquid_phase_velocity_seed,
                          oil_phase_velocity_seed,
                          water_phase_velocity_seed);
  phase_vel_DF new_vels (0., 0., 0., 0.);

  while (error > 1.e-4 && it < max_it)
    {
      temp_holdups.copy_operator (&prev_holdups);
      new_vels.copy_operator (&prev_vels);

      // FIX #A: vs = prev_vels * temp_holdups (Picard, my_code_12 style).
      const double vsg_iter = prev_vels.gas   * temp_holdups.gas;
      const double vso_iter = prev_vels.oil   * temp_holdups.oil;
      const double vsw_iter = prev_vels.water * temp_holdups.water;
      const double j_iter   = vsg_iter + vso_iter + vsw_iter;

      // Пересчёт oil/water flow weights внутри цикла.
      const double vsl_iter = vso_iter + vsw_iter;
      if (fabs (vsl_iter) > df_den_eps)
        {
          oil_flow_weight_input   = vso_iter / vsl_iter;
          water_flow_weight_input = vsw_iter / vsl_iter;
        }
      else if (temp_holdups.liquid > df_den_eps)
        {
          oil_flow_weight_input   = temp_holdups.oil   / temp_holdups.liquid;
          water_flow_weight_input = temp_holdups.water / temp_holdups.liquid;
        }

      // -------------------- gas/liquid stage on current holdups
      std::vector<double> dummy_D_sigma_gl (1U + mp.nc + 1U, 0.0);
      double sigma_o_raw_iter = 0.0;
      double sigma_w_raw_iter = 0.0;
      double D_sigma_o_D_p_dummy = 0.0;
      double D_sigma_w_D_p_dummy = 0.0;

      double gas_liq_interfacial_tension_iter =
          surf_mult *
          pipe_gas_liq_interfacial_tension_holdup_weightening (
              element_status->p * converter_metric_to_field.pressure_mult (),
              160.0,
              45.5,
              temp_holdups.oil,
              temp_holdups.water,
              element_status,
              dummy_D_sigma_gl,
              sigma_o_raw_iter,
              sigma_w_raw_iter,
              D_sigma_o_D_p_dummy,
              D_sigma_w_D_p_dummy);
      gas_liq_interfacial_tension_iter =
          oil_flow_weight_input * surf_mult * sigma_o_raw_iter
          + water_flow_weight_input * surf_mult * sigma_w_raw_iter;

      double liq_density_iter =
          oil_flow_weight_input * element_status->phase_rho[PHASE_OIL]
          + water_flow_weight_input * element_status->phase_rho[PHASE_WATER];

      const double gas_density_iter = std::max (element_status->phase_rho[PHASE_GAS], tnm::min_compare);

      double bubble_rise_velocity_iter = 0.0;
      if (gas_liq_interfacial_tension_iter > tnm::min_compare && liq_density_iter > tnm::min_compare)
        {
          bubble_rise_velocity_iter =
              tnav_pow (
                  tnav_div (gas_liq_interfacial_tension_iter * internal_const::grav_metric ()
                            * fabs (liq_density_iter - gas_density_iter),
                            liq_density_iter * liq_density_iter),
                  0.25);
        }

      double diametr_dimless_iter = 0.0;
      if (liq_density_iter - gas_density_iter > 0.0 && gas_liq_interfacial_tension_iter > tnm::min_compare)
        {
          diametr_dimless_iter =
              sqrt (tnav_div (internal_const::grav_metric () * (liq_density_iter - gas_density_iter),
                              gas_liq_interfacial_tension_iter))
              * diameter;
        }

      double linear_interpolation_derivative_dummy = 0.0;
      double Kut_number_iter =
          compute_critical_Kutateladze_number_by_diametr (
              diametr_dimless_iter,
              linear_interpolation_derivative_dummy);

      double flooding_velocity_iter =
          Kut_number_iter * sqrt (liq_density_iter / gas_density_iter) * bubble_rise_velocity_iter;


      const double A = 1.2;
      const double B = 0.3;
      const double F_v = 1.0;
      const double V_sgf_iter = df_safe_nonzero (flooding_velocity_iter);

      // FIX #A: mixture velocity из внутрицикловых vs.
      const double mixture_velocity = j_iter;

      const double ksi_second_iter =
          tnav_div (F_v * temp_holdups.gas * fabs (mixture_velocity), V_sgf_iter);

      const double ksi_iter = std::max (temp_holdups.gas, ksi_second_iter);

      double eta_iter = (ksi_iter - B) / (1.0 - B);
      if (!std::isfinite (eta_iter))
        eta_iter = (ksi_iter > B) ? 1.0 : 0.0;
      else if (eta_iter < 0.0)
        eta_iter = 0.0;
      else if (eta_iter > 1.0)
        eta_iter = 1.0;

      const double C0_iter = A / (1.0 + (A - 1.0) * eta_iter * eta_iter);

      const double K_g_low_iter = 1.53 / C0_iter;
      const double K_g_high_iter = Kut_number_iter;
      double K_g_iter = 0.0;
      if (temp_holdups.gas < 0.2)
        K_g_iter = K_g_low_iter;
      else if (temp_holdups.gas > 0.4)
        K_g_iter = K_g_high_iter;
      else
        K_g_iter = interpolate_y_against_x (temp_holdups.gas, 0.2, 0.4, K_g_low_iter, K_g_high_iter);

      double drift_velocity_iter = 0.0;
      {
        // FIX #C: sqrt над всем знаменателем.
        const double drift_den_inner_iter =
            temp_holdups.gas * C0_iter * tnav_div (gas_density_iter, liq_density_iter)
            + 1.0 - temp_holdups.gas * C0_iter;
        const double drift_den_iter = sqrt (std::max (drift_den_inner_iter, tnm::min_compare));

        if (drift_den_iter > tnm::min_division)
          {
            drift_velocity_iter =
                ((1.0 - temp_holdups.gas * C0_iter) * C0_iter * K_g_iter * bubble_rise_velocity_iter)
                / drift_den_iter;
            drift_velocity_iter *= drift_incl_mult;
          }
      }

      const double gas_phase_velocity_iter =
          C0_iter * mixture_velocity + drift_velocity_iter;

      double liquid_phase_velocity_iter = 0.0;
      if (1.0 - temp_holdups.gas > tnm::min_division)
        {
          liquid_phase_velocity_iter =
              tnav_div (1.0 - temp_holdups.gas * C0_iter, 1.0 - temp_holdups.gas) * mixture_velocity
              - tnav_div (temp_holdups.gas, 1.0 - temp_holdups.gas) * drift_velocity_iter;
        }

      new_vels.gas = gas_phase_velocity_iter;

      // FIX #A: holdup update через vsg_iter вместо зафиксированного vs-input.
      double raw_alpha_g_new = 0.5;
      if (fabs (new_vels.gas) > tnm::min_compare)
        raw_alpha_g_new = vsg_iter / new_vels.gas;
      else
        raw_alpha_g_new = prev_holdups.gas;

      new_holdups.gas = raw_alpha_g_new;
      new_holdups.liquid = 1.0 - new_holdups.gas;
      new_vels.liquid = liquid_phase_velocity_iter;

      double oil_frac_in_liquid_iter = 0.5;
      double water_frac_in_liquid_iter = 0.5;
      if (temp_holdups.liquid > tnm::min_division)
        {
          oil_frac_in_liquid_iter = temp_holdups.oil / df_safe_nonzero (temp_holdups.liquid);
          water_frac_in_liquid_iter = 1.0 - oil_frac_in_liquid_iter;
        }

      // -------------------- oil/water stage on current liquid split
      const double B1_OW = 0.4;
      const double B2_OW = 0.7;
      const double A_OW = 1.2;

      double C0_OW_iter = 0.0;
      if (oil_frac_in_liquid_iter < B1_OW)
        C0_OW_iter = A_OW;
      else if (oil_frac_in_liquid_iter > B2_OW)
        C0_OW_iter = 1.0;
      else
        C0_OW_iter = interpolate_y_against_x (oil_frac_in_liquid_iter, B1_OW, B2_OW, A_OW, 1.0);

      double D_go_dummy = 0.0;
      double D_gw_dummy = 0.0;

      const double gas_oil_interfacial_tension_iter =
          surf_mult *
          pipe_gas_oil_interfacial_tension_and_deriv (
              45.5,
              element_status->p * converter_metric_to_field.pressure_mult (),
              160.0,
              D_go_dummy);

      const double gas_wat_interfacial_tension_iter =
          surf_mult *
          pipe_gas_wat_interfacial_tension_and_deriv (
              element_status->p * converter_metric_to_field.pressure_mult (),
              160.0,
              D_gw_dummy);

      // FIX #D: σow = |σgo − σgw|.
      const double wat_oil_interfacial_tension_iter =
          fabs (gas_oil_interfacial_tension_iter - gas_wat_interfacial_tension_iter);

      double bubble_rise_velocity_OW_iter = 0.0;
      if (wat_oil_interfacial_tension_iter > tnm::min_compare
          && element_status->phase_rho[PHASE_WATER] > element_status->phase_rho[PHASE_OIL]
          && element_status->phase_rho[PHASE_WATER] > tnm::min_compare)
        {
          bubble_rise_velocity_OW_iter =
              tnav_pow (
                  wat_oil_interfacial_tension_iter * internal_const::grav_metric ()
                  * (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL])
                  / (element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER]),
                  0.25);
        }

      const double drift_velocity_OW_iter =
          drift_incl_mult * 1.53 * bubble_rise_velocity_OW_iter * tnav_pow (1.0 - oil_frac_in_liquid_iter, 2);

      const double oil_phase_velocity_iter =
          C0_OW_iter * liquid_phase_velocity_iter + drift_velocity_OW_iter;

      double water_phase_velocity_iter = 0.0;
      if (1.0 - oil_frac_in_liquid_iter > tnm::min_division)
        {
          water_phase_velocity_iter =
              tnav_div (1.0 - oil_frac_in_liquid_iter * C0_OW_iter, 1.0 - oil_frac_in_liquid_iter) * liquid_phase_velocity_iter
              - tnav_div (oil_frac_in_liquid_iter, 1.0 - oil_frac_in_liquid_iter) * drift_velocity_OW_iter;
        }

      new_vels.oil = oil_phase_velocity_iter;
      new_vels.water = water_phase_velocity_iter;

      // FIX #A: beta_o update через vso_iter вместо зафиксированного vs-input.
      double new_beta_o = 0.5;
      if (new_holdups.liquid > tnm::min_division && fabs (new_vels.oil) > tnm::min_compare)
        {
          const double raw_beta_o =
              vso_iter / (new_holdups.liquid * new_vels.oil);
          new_beta_o = raw_beta_o;
        }
      else
        {
          new_beta_o = (fabs (prev_holdups.liquid) > df_den_eps)
                       ? (prev_holdups.oil / prev_holdups.liquid)
                       : 0.0;
        }

      new_holdups.oil = new_beta_o * new_holdups.liquid;
      new_holdups.water = (1.0 - new_beta_o) * new_holdups.liquid;

      // FIX #A: при vs = prev*temp Picard-residuals тавтологически ≈0,
      // критерий сходимости — изменение holdups между итерациями.
      const double iter_delta =
          sqrt (tnav_sqr (prev_holdups.oil - new_holdups.oil)
                + tnav_sqr (prev_holdups.water - new_holdups.water)
                + tnav_sqr (prev_holdups.gas - new_holdups.gas));

      error = iter_delta;

      prev_holdups.copy_operator (&new_holdups);
      prev_vels.copy_operator (&new_vels);
      ++it;
    }

  dbg_inner_it = it;

  // ---------------------------------------------------------------------------
  // Final post-processing at the converged holdups.
  // These are the values y(x) for finite-difference derivative checks.
  // ---------------------------------------------------------------------------
  const double alpha_g = prev_holdups.gas;
  const double alpha_l = prev_holdups.liquid;

  double beta_o = 0.0;
  if (fabs (alpha_l) > df_den_eps)
    beta_o = prev_holdups.oil / alpha_l;

  const double alpha_o = prev_holdups.oil;
  const double alpha_w = prev_holdups.water;

  dbg_alpha_g = alpha_g;
  dbg_beta_o = beta_o;
  dbg_alpha_o = alpha_o;
  dbg_alpha_w = alpha_w;

  // -------------------- gas/liquid stage at final holdups
  double D_sigma_o_D_p_psi = 0.0;
  double D_sigma_w_D_p_psi = 0.0;

  sigma_o = pipe_gas_oil_interfacial_tension_and_deriv (
      45.5,
      element_status->p * converter_metric_to_field.pressure_mult (),
      160.0,
      D_sigma_o_D_p_psi); // raw dynes/cm

  sigma_w = pipe_gas_wat_interfacial_tension_and_deriv (
      element_status->p * converter_metric_to_field.pressure_mult (),
      160.0,
      D_sigma_w_D_p_psi); // raw dynes/cm

  const double sigma_o_SI = surf_mult * sigma_o;
  const double sigma_w_SI = surf_mult * sigma_w;

  double gas_liq_interfacial_tension =
      oil_flow_weight_input * sigma_o_SI + water_flow_weight_input * sigma_w_SI;

  gas_inter_tension = gas_liq_interfacial_tension;

  double liq_density =
      oil_flow_weight_input * element_status->phase_rho[PHASE_OIL]
      + water_flow_weight_input * element_status->phase_rho[PHASE_WATER];

  const double gas_density = std::max (element_status->phase_rho[PHASE_GAS], tnm::min_compare);
  liq_density_buf = liq_density;

  double bubble_rise_velocity = 0.0;
  if (gas_liq_interfacial_tension > tnm::min_compare && liq_density > tnm::min_compare)
    {
      bubble_rise_velocity =
          tnav_pow (
              tnav_div (gas_liq_interfacial_tension * internal_const::grav_metric ()
                        * fabs (liq_density - gas_density),
                        liq_density * liq_density),
              0.25);
    }
  bubble_rise_velocity_buf = bubble_rise_velocity;

  double diametr_dimless = 0.0;
  if (liq_density - gas_density > 0.0 && gas_liq_interfacial_tension > tnm::min_compare)
    {
      diametr_dimless =
          sqrt (tnav_div (internal_const::grav_metric () * (liq_density - gas_density),
                          gas_liq_interfacial_tension))
          * diameter;
    }
  diametr_dimless_buf = diametr_dimless;

  double linear_interpolation_derivative_dummy = 0.0;
  double Kut_number =
      compute_critical_Kutateladze_number_by_diametr (
          diametr_dimless,
          linear_interpolation_derivative_dummy);
  Kut_number_buf = Kut_number;

  double flooding_velocity = Kut_number * sqrt (liq_density / gas_density) * bubble_rise_velocity;
  flooding_velocity_buf = flooding_velocity;

  const double A = 1.2;
  const double B = 0.3;
  const double F_v = 1.0;
  const double V_sgf = df_safe_nonzero (flooding_velocity);

  const double mixture_velocity = mixture_superficial_velocity;

  const double ksi_second =
      tnav_div (F_v * alpha_g * fabs (mixture_velocity), V_sgf);

  double ksi_local = std::max (alpha_g, ksi_second);
  ksi_buf = ksi_local;
  unsigned int ksi_flag = (alpha_g > ksi_second) ? 1U : 2U;

  double eta_local = (ksi_local - B) / (1.0 - B);
  bool eta_clamped_low_local = false;
  bool eta_clamped_high_local = false;
  if (!std::isfinite (eta_local))
    {
      eta_local = (ksi_local > B) ? 1.0 : 0.0;
      eta_clamped_low_local = !(ksi_local > B);
      eta_clamped_high_local = (ksi_local > B);
    }
  else if (eta_local < 0.0)
    {
      eta_local = 0.0;
      eta_clamped_low_local = true;
    }
  else if (eta_local > 1.0)
    {
      eta_local = 1.0;
      eta_clamped_high_local = true;
    }
  eta_buf = eta_local;

  const double C0_local = A / (1.0 + (A - 1.0) * eta_local * eta_local);
  wsncs->wsn_C_0 = C0_local;

  const double K_g_low = 1.53 / C0_local;
  const double K_g_high = Kut_number;
  double K_g_local = 0.0;
  if (alpha_g < 0.2)
    K_g_local = K_g_low;
  else if (alpha_g > 0.4)
    K_g_local = K_g_high;
  else
    K_g_local = interpolate_y_against_x (alpha_g, 0.2, 0.4, K_g_low, K_g_high);
  K_g_buf = K_g_local;

  double drift_velocity = 0.0;
  {
    const double drift_den =
        1.0 + alpha_g * C0_local * (sqrt (tnav_div (gas_density, liq_density)) - 1.0);

    if (fabs (drift_den) > tnm::min_division)
      {
        drift_velocity =
            tnav_div ((1.0 - alpha_g * C0_local) * C0_local * K_g_local * bubble_rise_velocity,
                      drift_den);
        drift_velocity *= drift_incl_mult;
      }
  }
  wsncs->wsn_drift_velocity = drift_velocity;

  const double gas_phase_velocity = C0_local * mixture_velocity + drift_velocity;
  phase_gas_velocity_buf = gas_phase_velocity;

  double liquid_phase_velocity = 0.0;
  if (1.0 - alpha_g > tnm::min_division)
    {
      liquid_phase_velocity =
          tnav_div (1.0 - alpha_g * C0_local, 1.0 - alpha_g) * mixture_velocity
          - tnav_div (alpha_g, 1.0 - alpha_g) * drift_velocity;
    }
  phase_liquid_velocity_buf = liquid_phase_velocity;

  // -------------------- oil/water stage at final beta_o
  const double B1_OW = 0.4;
  const double B2_OW = 0.7;
  const double A_OW = 1.2;

  double C0_OW = 0.0;
  if (beta_o < B1_OW)
    C0_OW = A_OW;
  else if (beta_o > B2_OW)
    C0_OW = 1.0;
  else
    C0_OW = interpolate_y_against_x (beta_o, B1_OW, B2_OW, A_OW, 1.0);

  C_OW_buf = C0_OW;
  wsncs->wsn_C_0_OW = C0_OW;

  double D_go_dummy = 0.0;
  double D_gw_dummy = 0.0;

  const double gas_oil_interfacial_tension =
      surf_mult *
      pipe_gas_oil_interfacial_tension_and_deriv (
          45.5,
          element_status->p * converter_metric_to_field.pressure_mult (),
          160.0,
          D_go_dummy);

  const double gas_wat_interfacial_tension =
      surf_mult *
      pipe_gas_wat_interfacial_tension_and_deriv (
          element_status->p * converter_metric_to_field.pressure_mult (),
          160.0,
          D_gw_dummy);

  const double wat_oil_interfacial_tension =
      fabs (gas_oil_interfacial_tension * beta_o - gas_wat_interfacial_tension * (1.0 - beta_o));

  oil_inter_tension_buf = wat_oil_interfacial_tension;

  double bubble_rise_velocity_OW = 0.0;
  if (wat_oil_interfacial_tension > tnm::min_compare
      && element_status->phase_rho[PHASE_WATER] > element_status->phase_rho[PHASE_OIL]
      && element_status->phase_rho[PHASE_WATER] > tnm::min_compare)
    {
      bubble_rise_velocity_OW =
          tnav_pow (
              wat_oil_interfacial_tension * internal_const::grav_metric ()
              * (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL])
              / (element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER]),
              0.25);
    }
  bubble_rise_velocity_OW_buf = bubble_rise_velocity_OW;

  const double drift_velocity_OW =
      drift_incl_mult * 1.53 * bubble_rise_velocity_OW * tnav_pow (1.0 - beta_o, 2);

  V_d_OW_buf = drift_velocity_OW;
  wsncs->wsn_drift_velocity_OW = drift_velocity_OW;

  const double oil_phase_velocity = C0_OW * liquid_phase_velocity + drift_velocity_OW;
  oil_velocity_buf = oil_phase_velocity;

  double water_phase_velocity = liquid_phase_velocity;
  if (1.0 - beta_o > tnm::min_division)
    {
      water_phase_velocity =
          tnav_div (1.0 - beta_o * C0_OW, 1.0 - beta_o) * liquid_phase_velocity
          - tnav_div (beta_o, 1.0 - beta_o) * drift_velocity_OW;
    }
  water_velocity_buf = water_phase_velocity;

  // FIX #A: при Picard vs = prev*temp residuals нулевые по построению.
  dbg_R_g = 0.0;
  dbg_R_o = 0.0;

  const double rho_avg_df =
      element_status->phase_rho[PHASE_GAS]   * alpha_g
    + element_status->phase_rho[PHASE_OIL]   * alpha_o
    + element_status->phase_rho[PHASE_WATER] * alpha_w;
  wsncs->rho_avg_DF = rho_avg_df;

  // Active-set / branch flags for finite-difference validity checks.
  dbg_active_flags = 0U;
  if (alpha_g < 0.2)
    dbg_active_flags |= (1u << 7);
  else if (alpha_g > 0.4)
    dbg_active_flags |= (1u << 8);
  else
    dbg_active_flags |= (1u << 9);

  if (beta_o < B1_OW)
    dbg_active_flags |= (1u << 10);
  else if (beta_o > B2_OW)
    dbg_active_flags |= (1u << 11);
  else
    dbg_active_flags |= (1u << 12);

  if (eta_clamped_low_local)
    dbg_active_flags |= (1u << 13);
  else if (eta_clamped_high_local)
    dbg_active_flags |= (1u << 14);
  else
    dbg_active_flags |= (1u << 15);

  dbg_active_flags |= (ksi_flag << 16);

  // ---------------------------------------------------------------------------
  // VALUE-ONLY component rates for numerical derivative checks of q_c.
  // This is the crucial part that was missing in the old helper.
  // ---------------------------------------------------------------------------
  for (auto ic = mp.nc0; ic < mp.nc; ++ic)
    wsncs->wsn_component_rate[ic] = 0.0;

  for (unsigned int ip = 0; ip < mp.np; ++ip)
    {
      double phase_velocity = 0.0;
      double phase_holdup = 0.0;

      if (ip == PHASE_GAS)
        {
          phase_velocity = gas_phase_velocity;
          phase_holdup = alpha_g;
        }
      else if (ip == PHASE_OIL)
        {
          phase_velocity = oil_phase_velocity;
          phase_holdup = alpha_o;
        }
      else // PHASE_WATER
        {
          phase_velocity = water_phase_velocity;
          phase_holdup = alpha_w;
        }

      const double phase_superficial_flux =
          phase_velocity * phase_holdup * area;

      for (auto ic = mp.nc0; ic < mp.nc; ++ic)
        {
          wsncs->wsn_component_rate[ic] +=
              phase_superficial_flux
              * element_status->component_phase_x[ic * mp.np + ip]
              * element_status->phase_xi[ip];
        }
    }

  // For completeness update mixture mass rate and MMW in the same value-only way.
  //wsncs->wsn_mixture_mass_rate = 0.0;
  //wsncs->wsn_mmw = 0.0;

  /*for (auto ic = mp.nc0; ic < mp.nc; ++ic)
    {
      wsncs->wsn_mixture_mass_rate +=
          wsncs->wsn_component_rate[ic] * component_molar_weights[ic];
      wsncs->wsn_mmw += component_z_for_flow[ic] * component_molar_weights[ic];
    }
    */
}
