// ============================================================================
// DRIFT FLUX — clean implementation (Eclipse-compatible)
//
// Drop-in fragment for the body of a wells_compute_base member function.
// Assumes the same in-scope names as my_code_15.cpp:
//   wsncs, element_status, mp, seg, rep, i_meshblock,
//   current_therm_comp_input_props, itd, new_status,
//   converter_metric_to_field, component_molar_weights, component_z_for_flow.
//
// Model: Shi 2005 / Eclipse 8.66-8.91 default parameter set.
//   Gas-liquid:    A=1.2, B=0.3, F_v=1.0, a1=0.2, a2=0.4.
//   Oil-water:     A'=1.2, B1=0.4, B2=0.7, n'=2.
//
// Key correctness notes vs. prior versions:
//   (1) sqrt in V_d covers the ENTIRE denominator    a_g*C0*rho_g/rho_l + 1 - a_g*C0.
//   (2) beta* clamp keeps the ratio in [0,1] without fabs.
//   (3) Superficial inputs (vs_g, vs_o, vs_w) are computed ONCE before the
//       Picard loop and held constant; only (alpha_g, beta_o) iterate.
//   (4) sigma_ow = |sigma_go - sigma_wg|  with NO holdup weighting.
//   (5) Jacobian z-columns receive INDEPENDENT flash contributions
//       per component via phase_D_rho[id*np+ip], phase_D_xi[id*np+ip],
//       component_phase_D_x[(id*nc+ic)*np+ip] and phase_D_S[id*np+ip] —
//       so the Newton matrix is not rank-deficient on z.
//
// Layout of seg_vars (id):
//   id = 0              -> pressure
//   id = 1 .. mp.nc     -> component_N[id-1]
//   id = mp.nc + 1      -> q_tot (total molar rate)
// ============================================================================

if (element_status)
  {
    const unsigned int nseg_vars = 1U + mp.nc + 1U;
    const unsigned int id_qtot   = mp.nc + 1U;
    const double surf_mult = internal_const::SURFACE_TENSION_DYNES_CM_TO_NEWTON_M_MULT ();
    const double g_acc     = internal_const::grav_metric ();
    const double area      = seg.wsn->pipe_props.area;
    const double diameter  = seg.wsn->pipe_props.diameter;
    const double p_to_psi  = converter_metric_to_field.pressure_mult ();
    const double eps_den   = tnm::min_compare;

    auto safe_nonzero =
      [&] (double v) -> double
      {
        if (!std::isfinite (v)) return (v < 0.0) ? -eps_den : eps_den;
        if (fabs (v) < eps_den) return (v < 0.0) ? -eps_den : eps_den;
        return v;
      };

    auto clamp01 =
      [] (double x) -> double
      {
        if (!std::isfinite (x)) return 0.0;
        if (x < 0.0) return 0.0;
        if (x > 1.0) return 1.0;
        return x;
      };

    // -----------------------------------------------------------------------
    // Preserve q_tot across flash, then refresh wsncs from element_status.
    // -----------------------------------------------------------------------
    const double q_tot = wsncs->wsn_mixture_molar_rate;

    if (wsncs->wsn_flow_dir == segment_flow_direction_t::from_parent_to_child)
      set_flow_direction_dependent_segment_params_to_element_status (seg, element_status, new_status);

    if (element_status->component_N_tot > tnm::min_compare)
      {
        if (auto err = run_flash <true> (rep, i_meshblock, element_status,
                                         current_therm_comp_input_props, itd);
            err != segments_solver_err_t::none)
          return;
      }

    // Snapshot previous DF state BEFORE fill_wsncs overwrites phase_S with flash.
    const double prev_df_ag = std::isfinite (wsncs->phase_S[PHASE_GAS])   ? wsncs->phase_S[PHASE_GAS]   : 0.0;
    const double prev_df_ao = std::isfinite (wsncs->phase_S[PHASE_OIL])   ? wsncs->phase_S[PHASE_OIL]   : 0.0;
    const double prev_df_aw = std::isfinite (wsncs->phase_S[PHASE_WATER]) ? wsncs->phase_S[PHASE_WATER] : 0.0;
    const double prev_df_sum = prev_df_ag + prev_df_ao + prev_df_aw;
    const bool   use_prev_seed =
        std::isfinite (prev_df_sum) && prev_df_sum > 1.0e-9 &&
        prev_df_ag >= 0.0 && prev_df_ao >= 0.0 && prev_df_aw >= 0.0;

    fill_wsncs_from_element_status (wsncs, element_status, mp);
    wsncs->wsn_mixture_molar_rate = q_tot;

    wsncs->wsn_mmw = 0.0;
    wsncs->wsn_mixture_mass_rate = 0.0;
    for (auto ic = mp.nc0; ic < mp.nc; ++ic)
      wsncs->wsn_component_rate[ic] = 0.0;
    for (unsigned int id = 0; id < nseg_vars; ++id)
      {
        wsncs->D_mixture_mass_rate_D_seg_vars[id] = 0.0;
        wsncs->D_C0_D_seg_vars[id]                = 0.0;
        wsncs->D_drift_velocity_D_seg_vars[id]    = 0.0;
        wsncs->D_C0_OW_D_seg_vars[id]             = 0.0;
        wsncs->D_drift_velocity_OW_D_seg_vars[id] = 0.0;
        wsncs->D_rho_avg_D_seg_vars[id]           = 0.0;
        for (auto ic = mp.nc0; ic < mp.nc; ++ic)
          wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id] = 0.0;
      }

    // -----------------------------------------------------------------------
    // TOP segment: no DF, no slip. Rates from mixture composition.
    // -----------------------------------------------------------------------
    if (seg.wsn->wsn_index == TOP_SEG_INDEX)
      {
        wsncs->wsn_C_0              = 1.0;
        wsncs->wsn_drift_velocity   = 0.0;
        wsncs->wsn_C_0_OW           = 1.0;
        wsncs->wsn_drift_velocity_OW = 0.0;

        for (auto ic = mp.nc0; ic < mp.nc; ++ic)
          {
            wsncs->wsn_component_rate[ic] = q_tot * component_z_for_flow[ic];
            wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id_qtot] = component_z_for_flow[ic];
            wsncs->wsn_mixture_mass_rate += wsncs->wsn_component_rate[ic] * component_molar_weights[ic];
            wsncs->wsn_mmw               += component_z_for_flow[ic] * component_molar_weights[ic];
          }
        wsncs->D_mixture_mass_rate_D_seg_vars[id_qtot] = wsncs->wsn_mmw;

        wsncs->rho_avg_DF = 0.0;
        for (unsigned int ip = 0; ip < mp.np; ++ip)
          wsncs->rho_avg_DF += element_status->phase_S[ip] * element_status->phase_rho[ip];
        // fall through: this is terminal for TOP; caller handles the rest.
      }
    else
      {
        // ===================================================================
        //  Non-top segment: full DF.
        // ===================================================================

        // -----------------------------------------------------------------
        // Inclination multiplier (Eclipse 8.91). Depends only on geometry.
        // -----------------------------------------------------------------
        double drift_incl_mult = 1.0;
        {
          const double L = fabs (seg.wsn->pipe_props.length);
          const double H = fabs (seg.wsn->pipe_props.depth_change);
          if (L > tnm::min_compare)
            {
              double cos_t = H / L;
              if (cos_t < 0.0) cos_t = 0.0;
              else if (cos_t > 1.0) cos_t = 1.0;
              if (cos_t >= 1.0 - tnm::min_compare)
                drift_incl_mult = 1.0;
              else
                {
                  const double sin_t = sqrt (std::max (0.0, 1.0 - cos_t * cos_t));
                  drift_incl_mult = sqrt (cos_t) * tnav_pow (1.0 + sin_t, 2);
                }
            }
        }

        // -----------------------------------------------------------------
        // Mixture volumetric superficial velocity j_m and its derivatives.
        // j_m = q_tot / (area * avg_xi).
        // For id < 1+nc : d(j_m)/d(x_id) = -q_tot/(area*avg_xi^2) * avg_D_xi[id].
        // For id = id_qtot:                  d(j_m)/d(q_tot)     =  1/(area*avg_xi).
        // -----------------------------------------------------------------
        const double avg_xi = safe_nonzero (element_status->avg_xi);
        const double j_m    = q_tot / (area * avg_xi);

        std::vector<double> D_jm (nseg_vars, 0.0);
        for (unsigned int id = 0; id < nseg_vars; ++id)
          {
            if (id < 1U + mp.nc)
              D_jm[id] = -q_tot / (area * avg_xi * avg_xi) * element_status->avg_D_xi[id];
            else
              D_jm[id] =  1.0  / (area * avg_xi);
            wsncs->D_average_volumetric_mixture_velocity_D_seg_vars[id] = D_jm[id];
          }

        // -----------------------------------------------------------------
        // Frozen superficial inputs vs_{g,o,w}_in = phase_S[ip] * j_m.
        // CRITICAL: these stay constant throughout the Picard loop.
        // Their derivatives carry TWO independent z-paths:
        //   - phase_D_S[id*np+ip]  * j_m                           (flash path)
        //   - phase_S[ip]          * D_jm[id]                      (mixture path)
        // Using phase_D_S directly (not avg_D_xi) is what breaks the
        // z-column collinearity seen in the prior code.
        // -----------------------------------------------------------------
        const double S_g = element_status->phase_S[PHASE_GAS];
        const double S_o = element_status->phase_S[PHASE_OIL];
        const double S_w = element_status->phase_S[PHASE_WATER];
        const double vs_g_in = S_g * j_m;
        const double vs_o_in = S_o * j_m;
        const double vs_w_in = S_w * j_m;
        const double vs_l_in = vs_o_in + vs_w_in;

        auto D_phase_S_id = [&] (unsigned int id, unsigned int ip) -> double
          {
            if (id < 1U + mp.nc) return element_status->phase_D_S[id * mp.np + ip];
            return 0.0;
          };

        std::vector<double> D_vsg_in (nseg_vars, 0.0);
        std::vector<double> D_vso_in (nseg_vars, 0.0);
        std::vector<double> D_vsw_in (nseg_vars, 0.0);
        for (unsigned int id = 0; id < nseg_vars; ++id)
          {
            D_vsg_in[id] = D_phase_S_id (id, PHASE_GAS)   * j_m + S_g * D_jm[id];
            D_vso_in[id] = D_phase_S_id (id, PHASE_OIL)   * j_m + S_o * D_jm[id];
            D_vsw_in[id] = D_phase_S_id (id, PHASE_WATER) * j_m + S_w * D_jm[id];
          }

        // Oil/water mixing weights INSIDE the liquid input (used for sigma_gl
        // and liquid density). Flow-weighted by superficial inputs, NOT by the
        // solved holdup beta_o — this is what keeps the sigma_gl evaluation
        // stable under the Picard iteration on beta_o.
        double w_o = 0.5, w_w = 0.5;
        if (fabs (vs_l_in) > eps_den)
          {
            w_o = vs_o_in / vs_l_in;
            w_w = vs_w_in / vs_l_in;
          }

        std::vector<double> D_w_o (nseg_vars, 0.0);
        std::vector<double> D_w_w (nseg_vars, 0.0);
        if (fabs (vs_l_in) > eps_den)
          for (unsigned int id = 0; id < nseg_vars; ++id)
            {
              const double D_vsl = D_vso_in[id] + D_vsw_in[id];
              D_w_o[id] = (D_vso_in[id] * vs_l_in - vs_o_in * D_vsl) / (vs_l_in * vs_l_in);
              D_w_w[id] = (D_vsw_in[id] * vs_l_in - vs_w_in * D_vsl) / (vs_l_in * vs_l_in);
            }

        // -----------------------------------------------------------------
        // Initial seed for (alpha_g, beta_o).
        //   Step 1   : from flash (phase_S of current composition).
        //   Subsequent: previous converged DF state (continuation).
        // -----------------------------------------------------------------
        double alpha_g_seed, alpha_o_seed, alpha_w_seed;
        if (use_prev_seed)
          {
            alpha_g_seed = prev_df_ag / prev_df_sum;
            alpha_o_seed = prev_df_ao / prev_df_sum;
            alpha_w_seed = prev_df_aw / prev_df_sum;
          }
        else
          {
            alpha_g_seed = S_g;
            alpha_o_seed = S_o;
            alpha_w_seed = S_w;
          }
        const double alpha_l_seed = alpha_o_seed + alpha_w_seed;
        double beta_o_seed = 0.5;
        if (alpha_l_seed > eps_den)
          beta_o_seed = alpha_o_seed / alpha_l_seed;

        // -----------------------------------------------------------------
        // Picard iteration (value-only).
        //   1. Evaluate sigma_gl, rho_l, v_c, Dhat, Ku, v_sgf at current a_g.
        //   2. Compute C_0, V_d, v_g.
        //   3. Update a_g_new = vs_g_in / v_g.
        //   4. On (a_g) converged: compute O/W closures and beta_o update.
        //   5. Repeat until |d(a_g)| and |d(beta_o)| small.
        //
        // Freezes: vs_g_in, vs_o_in, vs_w_in, w_o, w_w throughout.
        // -----------------------------------------------------------------
        double alpha_g = alpha_g_seed;
        double beta_o  = beta_o_seed;

        const double A_gl  = 1.2;
        const double B_gl  = 0.3;
        const double F_v   = 1.0;
        const double a1_gl = 0.2;
        const double a2_gl = 0.4;

        const double A_ow  = 1.2;
        const double B1_ow = 0.4;
        const double B2_ow = 0.7;
        const int    n_ow  = 2;

        // Locals that the value loop stores for later analytical partial
        // derivative reconstruction in the converged block.
        double sigma_o_raw  = 0.0;
        double sigma_w_raw  = 0.0;
        double sigma_gl     = 0.0;
        double sigma_ow     = 0.0;
        double rho_g        = std::max (element_status->phase_rho[PHASE_GAS],   eps_den);
        double rho_o        = std::max (element_status->phase_rho[PHASE_OIL],   eps_den);
        double rho_w        = std::max (element_status->phase_rho[PHASE_WATER], eps_den);
        double rho_l        = w_o * rho_o + w_w * rho_w;
        double v_c          = 0.0;
        double Dhat         = 0.0;
        double Ku           = 0.0;
        double v_sgf        = 0.0;
        double xi_sh        = 0.0;   // "ksi" in Shi
        double eta_sh       = 0.0;
        double C0           = A_gl;
        double K_gl         = 1.53 / A_gl;
        double V_d          = 0.0;
        double v_g          = j_m;
        double v_l          = j_m;
        double v_c_ow       = 0.0;
        double C0_ow        = 1.0;
        double V_d_ow       = 0.0;
        double v_o          = j_m;
        double v_w          = j_m;
        double eta_ow       = 0.0;

        std::vector<double> D_sigma_gl_dummy (nseg_vars, 0.0);
        double D_sigma_o_raw_D_p = 0.0;
        double D_sigma_w_raw_D_p = 0.0;

        const int    max_it = 50;
        const double tol    = 1.0e-6;
        int it = 0;
        double err_g = 1.0, err_b = 1.0;

        while ((err_g > tol || err_b > tol) && it < max_it)
          {
            const double alpha_g_prev = alpha_g;
            const double beta_o_prev  = beta_o;

            // --- surface tensions via the user's helper; holdup args only
            // control the internal mixing the helper may apply. We overwrite
            // with flow-weighted values using w_o/w_w right after.
            sigma_gl = surf_mult *
                pipe_gas_liq_interfacial_tension_holdup_weightening (
                    element_status->p * p_to_psi,
                    160.0, 45.5,
                    std::max (eps_den, (1.0 - alpha_g) * beta_o),
                    std::max (eps_den, (1.0 - alpha_g) * (1.0 - beta_o)),
                    element_status,
                    D_sigma_gl_dummy, sigma_o_raw, sigma_w_raw,
                    D_sigma_o_raw_D_p, D_sigma_w_raw_D_p);
            sigma_gl = w_o * surf_mult * sigma_o_raw + w_w * surf_mult * sigma_w_raw;
            sigma_ow = fabs (surf_mult * sigma_o_raw - surf_mult * sigma_w_raw);

            rho_g = std::max (element_status->phase_rho[PHASE_GAS],   eps_den);
            rho_o = std::max (element_status->phase_rho[PHASE_OIL],   eps_den);
            rho_w = std::max (element_status->phase_rho[PHASE_WATER], eps_den);
            rho_l = w_o * rho_o + w_w * rho_w;

            const double drho_gl  = std::max (rho_l - rho_g, eps_den);
            v_c = tnav_pow (sigma_gl * g_acc * drho_gl / (rho_l * rho_l), 0.25);

            Dhat = sqrt (g_acc * drho_gl / std::max (sigma_gl, eps_den)) * diameter;
            double lin_deriv_dummy = 0.0;
            Ku = compute_critical_Kutateladze_number_by_diametr (Dhat, lin_deriv_dummy);
            v_sgf = Ku * sqrt (rho_l / rho_g) * v_c;

            // Shi "xi" — Eclipse 8.70 with F_v.
            const double xi2 = F_v * alpha_g * fabs (j_m) / safe_nonzero (v_sgf);
            xi_sh = std::max (alpha_g, xi2);
            eta_sh = clamp01 ((xi_sh - B_gl) / (1.0 - B_gl));
            C0 = A_gl / (1.0 + (A_gl - 1.0) * eta_sh * eta_sh);

            // K_gl: piecewise in alpha_g on [a1, a2].
            const double K_low  = 1.53 / C0;
            const double K_high = Ku;
            if (alpha_g <= a1_gl)       K_gl = K_low;
            else if (alpha_g >= a2_gl)  K_gl = K_high;
            else                        K_gl = interpolate_y_against_x (alpha_g, a1_gl, a2_gl, K_low, K_high);

            // V_d (Eclipse 8.88) — sqrt over ENTIRE denominator.
            const double aC      = alpha_g * C0;
            const double num_gd  = (1.0 - aC);
            const double den_arg = aC * (rho_g / rho_l) + 1.0 - aC;
            const double den_gd  = sqrt (std::max (den_arg, eps_den));
            V_d = drift_incl_mult * K_gl * v_c * num_gd / den_gd;

            v_g = C0 * j_m + V_d;
            if (fabs (1.0 - alpha_g) > eps_den)
              {
                const double f_gl = (1.0 - aC)     / (1.0 - alpha_g);
                const double g_gl = alpha_g        / (1.0 - alpha_g);
                v_l = f_gl * j_m - g_gl * V_d;
              }
            else
              v_l = j_m;

            // Alpha_g update from vs_g_in = alpha_g * v_g.
            const double alpha_g_new = vs_g_in / safe_nonzero (v_g);
            alpha_g = clamp01 (alpha_g_new);

            // Oil-water stage.
            const double drho_ow = std::max (rho_w - rho_o, eps_den);
            v_c_ow = tnav_pow (sigma_ow * g_acc * drho_ow / (rho_w * rho_w), 0.25);

            // C_0_ow via double-cutoff in beta_o (Eclipse analog of gas-liquid profile).
            const double t_ow = clamp01 ((beta_o - B1_ow) / (B2_ow - B1_ow));
            double t_pow = 1.0;
            for (int k = 0; k < n_ow; ++k) t_pow *= t_ow;
            eta_ow = t_pow;
            C0_ow = 1.0 + (A_ow - 1.0) * eta_ow;

            const double bC      = beta_o * C0_ow;
            const double num_ow  = (1.0 - bC);
            const double den_arg_ow = bC * (rho_o / rho_w) + 1.0 - bC;
            const double den_ow  = sqrt (std::max (den_arg_ow, eps_den));
            V_d_ow = drift_incl_mult * 1.53 * v_c_ow * num_ow / den_ow;

            v_o = C0_ow * v_l + V_d_ow;
            if (fabs (1.0 - beta_o) > eps_den)
              {
                const double f_ow = (1.0 - bC)     / (1.0 - beta_o);
                const double g_ow = beta_o         / (1.0 - beta_o);
                v_w = f_ow * v_l - g_ow * V_d_ow;
              }
            else
              v_w = v_l;

            const double alpha_l = 1.0 - alpha_g;
            if (alpha_l > eps_den)
              {
                const double beta_o_new = vs_o_in / safe_nonzero (alpha_l * v_o);
                beta_o = clamp01 (beta_o_new);
              }

            err_g = fabs (alpha_g - alpha_g_prev);
            err_b = fabs (beta_o - beta_o_prev);
            ++it;
          }

        // Final holdups after Picard.
        const double alpha_l = std::max (eps_den, 1.0 - alpha_g);
        const double alpha_o = alpha_l * beta_o;
        const double alpha_w = alpha_l * (1.0 - beta_o);

        // ===================================================================
        // Analytical derivatives at the converged fixed point.
        //
        // We need, for each id in [0, nseg_vars):
        //   d(alpha_g)/d(x_id), d(beta_o)/d(x_id)
        //   d(v_g)/d(x_id), d(v_l)/d(x_id), d(v_o)/d(x_id), d(v_w)/d(x_id)
        //
        // The holdup derivatives come from the 2x2 IFT on the residuals:
        //   R_g(a_g, b_o, x) = a_g * v_g(a_g, b_o, x) - vs_g_in(x) = 0
        //   R_o(a_g, b_o, x) = (1 - a_g) * b_o * v_o(a_g, b_o, x) - vs_o_in(x) = 0
        //
        // We decompose each intermediate derivative into:
        //   (partial) : holding (a_g, b_o) fixed, varying only x_id
        //   (D_ag)    : derivative w.r.t. a_g at fixed (b_o, x)
        //   (D_bo)    : derivative w.r.t. b_o at fixed (a_g, x)
        // then chain for the total derivative.
        // ===================================================================

        // -----------------------------------------------------------------
        // Phase densities and their partial derivatives.
        // phase_D_rho[id*np+ip]  — ip-per-id flash derivative (in scope).
        // -----------------------------------------------------------------
        auto D_phase_rho_id = [&] (unsigned int id, unsigned int ip) -> double
          {
            if (id < 1U + mp.nc) return element_status->phase_D_rho[id * mp.np + ip];
            return 0.0;
          };
        auto D_phase_xi_id = [&] (unsigned int id, unsigned int ip) -> double
          {
            if (id < 1U + mp.nc) return element_status->phase_D_xi[id * mp.np + ip];
            return 0.0;
          };

        // rho_l = w_o * rho_o + w_w * rho_w.
        std::vector<double> D_rho_l_partial (nseg_vars, 0.0);
        for (unsigned int id = 0; id < nseg_vars; ++id)
          {
            D_rho_l_partial[id] =
                D_w_o[id] * rho_o + w_o * D_phase_rho_id (id, PHASE_OIL)
              + D_w_w[id] * rho_w + w_w * D_phase_rho_id (id, PHASE_WATER);
          }

        // -----------------------------------------------------------------
        // Surface tensions — partials w.r.t. seg_vars.
        // We use the weighted combination:
        //   sigma_gl = w_o * surf_mult * sigma_o_raw + w_w * surf_mult * sigma_w_raw.
        // With sigma_{o,w}_raw depending on pressure (derivative from helper),
        // and w_{o,w} depending on z,q via D_vs{o,w}_in.
        // Off-pressure composition dependence of sigma_raw is neglected here
        // (helper does not return z-derivatives); this is acceptable since
        // sigma varies weakly with composition at given (p, T).
        // -----------------------------------------------------------------
        const double sigma_o_nm = surf_mult * sigma_o_raw;
        const double sigma_w_nm = surf_mult * sigma_w_raw;
        const double D_sigma_o_D_p = surf_mult * D_sigma_o_raw_D_p * p_to_psi;
        const double D_sigma_w_D_p = surf_mult * D_sigma_w_raw_D_p * p_to_psi;

        std::vector<double> D_sigma_gl_partial (nseg_vars, 0.0);
        std::vector<double> D_sigma_ow_partial (nseg_vars, 0.0);
        for (unsigned int id = 0; id < nseg_vars; ++id)
          {
            const double dSo = (id == 0) ? D_sigma_o_D_p : 0.0;
            const double dSw = (id == 0) ? D_sigma_w_D_p : 0.0;
            D_sigma_gl_partial[id] = D_w_o[id] * sigma_o_nm + w_o * dSo
                                    + D_w_w[id] * sigma_w_nm + w_w * dSw;
            const double sign_ow = (sigma_o_nm - sigma_w_nm >= 0.0) ? 1.0 : -1.0;
            D_sigma_ow_partial[id] = sign_ow * (dSo - dSw);
          }

        // -----------------------------------------------------------------
        // Characteristic bubble velocity v_c = (sigma_gl*g*drho/rho_l^2)^0.25.
        // d(v_c)/d(x) = 0.25 * v_c * d(log argument)/d(x)
        //            = 0.25 * v_c * [D_sigma/sigma + D_drho/drho - 2*D_rho_l/rho_l]
        // where drho = rho_l - rho_g.
        // -----------------------------------------------------------------
        const double drho_gl = std::max (rho_l - rho_g, eps_den);
        std::vector<double> D_v_c_partial (nseg_vars, 0.0);
        if (v_c > eps_den)
          for (unsigned int id = 0; id < nseg_vars; ++id)
            {
              const double dSigma = D_sigma_gl_partial[id];
              const double dRhoL  = D_rho_l_partial[id];
              const double dRhoG  = D_phase_rho_id (id, PHASE_GAS);
              const double dDrho  = dRhoL - dRhoG;
              D_v_c_partial[id] = 0.25 * v_c *
                  (dSigma / safe_nonzero (sigma_gl)
                   + dDrho / safe_nonzero (drho_gl)
                   - 2.0 * dRhoL / safe_nonzero (rho_l));
            }

        // Dhat = diameter * sqrt(g * drho / sigma_gl).
        // d(Dhat)/d(x) = 0.5 * Dhat * [D_drho/drho - D_sigma/sigma].
        std::vector<double> D_Dhat_partial (nseg_vars, 0.0);
        if (Dhat > eps_den)
          for (unsigned int id = 0; id < nseg_vars; ++id)
            {
              const double dRhoL  = D_rho_l_partial[id];
              const double dRhoG  = D_phase_rho_id (id, PHASE_GAS);
              const double dDrho  = dRhoL - dRhoG;
              const double dSigma = D_sigma_gl_partial[id];
              D_Dhat_partial[id] = 0.5 * Dhat *
                  (dDrho / safe_nonzero (drho_gl) - dSigma / safe_nonzero (sigma_gl));
            }

        // Ku = f(Dhat). Linear derivative from helper call.
        double Ku_lin_deriv = 0.0;
        (void) compute_critical_Kutateladze_number_by_diametr (Dhat, Ku_lin_deriv);
        std::vector<double> D_Ku_partial (nseg_vars, 0.0);
        for (unsigned int id = 0; id < nseg_vars; ++id)
          D_Ku_partial[id] = Ku_lin_deriv * D_Dhat_partial[id];

        // v_sgf = Ku * sqrt(rho_l/rho_g) * v_c.
        // d(v_sgf)/d(x) = D_Ku*sqrt(...)*v_c + Ku*d(sqrt)/d(x)*v_c + Ku*sqrt(...)*D_vc,
        //   d(sqrt(rho_l/rho_g))/d(x) = 0.5*sqrt(rho_l/rho_g)*(D_rho_l/rho_l - D_rho_g/rho_g).
        const double sqrt_rr = sqrt (std::max (rho_l / std::max (rho_g, eps_den), eps_den));
        std::vector<double> D_v_sgf_partial (nseg_vars, 0.0);
        for (unsigned int id = 0; id < nseg_vars; ++id)
          {
            const double dRhoL  = D_rho_l_partial[id];
            const double dRhoG  = D_phase_rho_id (id, PHASE_GAS);
            const double dSqrt  = 0.5 * sqrt_rr * (dRhoL / safe_nonzero (rho_l) - dRhoG / safe_nonzero (rho_g));
            D_v_sgf_partial[id] = D_Ku_partial[id] * sqrt_rr * v_c
                                + Ku * dSqrt * v_c
                                + Ku * sqrt_rr * D_v_c_partial[id];
          }

        // -----------------------------------------------------------------
        // xi (Shi): xi = max(alpha_g, F_v * alpha_g * |j_m| / v_sgf).
        // Compute D_xi split into partial + d(xi)/d(alpha_g).
        // -----------------------------------------------------------------
        const double xi2_val = F_v * alpha_g * fabs (j_m) / safe_nonzero (v_sgf);
        const bool   xi_is_alpha = (alpha_g > xi2_val);

        std::vector<double> D_xi_partial (nseg_vars, 0.0);
        double D_xi_D_alpha_g = 0.0;
        if (xi_is_alpha)
          {
            D_xi_D_alpha_g = 1.0;
          }
        else
          {
            D_xi_D_alpha_g = F_v * fabs (j_m) / safe_nonzero (v_sgf);
            const double sign_jm = (j_m >= 0.0) ? 1.0 : -1.0;
            for (unsigned int id = 0; id < nseg_vars; ++id)
              {
                // xi2 = F_v * alpha_g * sign_jm * j_m / v_sgf
                const double d_jm_abs = sign_jm * D_jm[id];
                const double d_num    = F_v * alpha_g * d_jm_abs;
                const double d_den    = D_v_sgf_partial[id];
                D_xi_partial[id] = (d_num * v_sgf - F_v * alpha_g * fabs (j_m) * d_den)
                                   / (v_sgf * v_sgf);
              }
          }

        // eta = clamp01((xi - B)/(1 - B)).
        const double eta_raw = (xi_sh - B_gl) / (1.0 - B_gl);
        const bool   eta_interior = (eta_raw > 0.0 && eta_raw < 1.0);
        std::vector<double> D_eta_partial (nseg_vars, 0.0);
        double D_eta_D_alpha_g = 0.0;
        if (eta_interior)
          {
            const double inv = 1.0 / (1.0 - B_gl);
            for (unsigned int id = 0; id < nseg_vars; ++id)
              D_eta_partial[id] = D_xi_partial[id] * inv;
            D_eta_D_alpha_g = D_xi_D_alpha_g * inv;
          }

        // C0 = A / (1 + (A - 1)*eta^2).
        // dC0/d? = -A*(A-1)*2*eta / (1 + (A-1)*eta^2)^2 * d(eta).
        std::vector<double> D_C0_partial (nseg_vars, 0.0);
        double D_C0_D_alpha_g = 0.0;
        {
          const double denom = 1.0 + (A_gl - 1.0) * eta_sh * eta_sh;
          const double coef  = -A_gl * (A_gl - 1.0) * 2.0 * eta_sh / (denom * denom);
          for (unsigned int id = 0; id < nseg_vars; ++id)
            D_C0_partial[id] = coef * D_eta_partial[id];
          D_C0_D_alpha_g = coef * D_eta_D_alpha_g;
        }

        // K_gl piecewise in alpha_g.
        const double K_low  = 1.53 / C0;
        const double K_high = Ku;
        std::vector<double> D_K_gl_partial (nseg_vars, 0.0);
        double D_K_gl_D_alpha_g = 0.0;
        if (alpha_g <= a1_gl)
          {
            // K_gl = 1.53 / C0
            for (unsigned int id = 0; id < nseg_vars; ++id)
              D_K_gl_partial[id] = -1.53 / (C0 * C0) * D_C0_partial[id];
            D_K_gl_D_alpha_g = -1.53 / (C0 * C0) * D_C0_D_alpha_g;
          }
        else if (alpha_g >= a2_gl)
          {
            // K_gl = Ku
            for (unsigned int id = 0; id < nseg_vars; ++id)
              D_K_gl_partial[id] = D_Ku_partial[id];
          }
        else
          {
            // K_gl = K_low + (K_high - K_low) * (alpha_g - a1)/(a2 - a1)
            const double t = (alpha_g - a1_gl) / (a2_gl - a1_gl);
            for (unsigned int id = 0; id < nseg_vars; ++id)
              {
                const double dKlow  = -1.53 / (C0 * C0) * D_C0_partial[id];
                const double dKhigh = D_Ku_partial[id];
                D_K_gl_partial[id] = dKlow + (dKhigh - dKlow) * t;
              }
            D_K_gl_D_alpha_g = (K_high - K_low) / (a2_gl - a1_gl)
                             + (- 1.53 / (C0 * C0) * D_C0_D_alpha_g) * (1.0 - t);
          }

        // V_d derivative — decompose numerator (1 - aC) and denominator sqrt(...).
        const double aC      = alpha_g * C0;
        const double num_gd  = (1.0 - aC);
        const double den_arg = aC * (rho_g / rho_l) + 1.0 - aC;
        const double den_gd  = sqrt (std::max (den_arg, eps_den));
        const double rg_over_rl = rho_g / safe_nonzero (rho_l);

        std::vector<double> D_V_d_partial (nseg_vars, 0.0);
        double D_V_d_D_alpha_g = 0.0;
        {
          const double prefac = drift_incl_mult; // constant
          // Vd = prefac * K * v_c * num / den.
          for (unsigned int id = 0; id < nseg_vars; ++id)
            {
              const double dK    = D_K_gl_partial[id];
              const double dVc   = D_v_c_partial[id];
              // d(aC) = alpha_g * D_C0 (no partial w.r.t. a_g here).
              const double dAC   = alpha_g * D_C0_partial[id];
              const double dNum  = -dAC;
              // d(den_arg) = d(aC)*rg/rl + aC*[d(rg)/rl - rg*d(rl)/rl^2] - d(aC).
              const double dRhoL = D_rho_l_partial[id];
              const double dRhoG = D_phase_rho_id (id, PHASE_GAS);
              const double dRR   = (dRhoG * rho_l - rho_g * dRhoL) / (rho_l * rho_l);
              const double d_den_arg = dAC * rg_over_rl + aC * dRR - dAC;
              const double dDen  = 0.5 / safe_nonzero (den_gd) * d_den_arg;

              D_V_d_partial[id] = prefac *
                  ( dK   * v_c  * num_gd / safe_nonzero (den_gd)
                  + K_gl * dVc  * num_gd / safe_nonzero (den_gd)
                  + K_gl * v_c  * dNum   / safe_nonzero (den_gd)
                  - K_gl * v_c  * num_gd * dDen / (den_gd * den_gd) );
            }
          // d(V_d)/d(alpha_g) — through K_gl(a_g), C0(a_g), aC = a_g*C0 explicit path.
          const double dAC_ag = C0 + alpha_g * D_C0_D_alpha_g;
          const double dNum_ag = -dAC_ag;
          const double d_den_arg_ag = dAC_ag * rg_over_rl - dAC_ag;
          const double dDen_ag = 0.5 / safe_nonzero (den_gd) * d_den_arg_ag;
          D_V_d_D_alpha_g = prefac *
              ( D_K_gl_D_alpha_g * v_c * num_gd / safe_nonzero (den_gd)
              + K_gl * v_c * dNum_ag / safe_nonzero (den_gd)
              - K_gl * v_c * num_gd * dDen_ag / (den_gd * den_gd) );
        }

        // v_g = C0 * j_m + V_d.
        std::vector<double> D_v_g_partial (nseg_vars, 0.0);
        double D_v_g_D_alpha_g = D_C0_D_alpha_g * j_m + D_V_d_D_alpha_g;
        double D_v_g_D_beta_o  = 0.0;
        for (unsigned int id = 0; id < nseg_vars; ++id)
          D_v_g_partial[id] = D_C0_partial[id] * j_m + C0 * D_jm[id] + D_V_d_partial[id];

        // v_l = f_gl * j_m - g_gl * V_d, with f_gl = (1 - aC)/(1 - a_g), g_gl = a_g/(1 - a_g).
        const double one_minus_ag = std::max (eps_den, 1.0 - alpha_g);
        const double f_gl = (1.0 - aC) / one_minus_ag;
        const double g_gl = alpha_g   / one_minus_ag;

        std::vector<double> D_v_l_partial (nseg_vars, 0.0);
        double D_v_l_D_alpha_g = 0.0;
        double D_v_l_D_beta_o  = 0.0;
        {
          const double dF_gl_ag = ((-1.0)*(D_C0_D_alpha_g * alpha_g + C0) * one_minus_ag
                                    - (1.0 - aC) * (-1.0)) / (one_minus_ag * one_minus_ag);
          const double dG_gl_ag = 1.0 / (one_minus_ag * one_minus_ag);
          D_v_l_D_alpha_g = dF_gl_ag * j_m - dG_gl_ag * V_d - g_gl * D_V_d_D_alpha_g;

          for (unsigned int id = 0; id < nseg_vars; ++id)
            {
              const double dF_gl_x = (-alpha_g * D_C0_partial[id]) / one_minus_ag;
              D_v_l_partial[id] =
                  dF_gl_x * j_m + f_gl * D_jm[id]
                - g_gl * D_V_d_partial[id];
            }
        }

        // Oil-water stage derivatives.
        const double drho_ow = std::max (rho_w - rho_o, eps_den);

        // v_c_ow.
        std::vector<double> D_v_c_ow_partial (nseg_vars, 0.0);
        if (v_c_ow > eps_den)
          for (unsigned int id = 0; id < nseg_vars; ++id)
            {
              const double dSigma = D_sigma_ow_partial[id];
              const double dRhoO  = D_phase_rho_id (id, PHASE_OIL);
              const double dRhoW  = D_phase_rho_id (id, PHASE_WATER);
              const double dDrho  = dRhoW - dRhoO;
              D_v_c_ow_partial[id] = 0.25 * v_c_ow *
                  (dSigma / safe_nonzero (sigma_ow)
                   + dDrho / safe_nonzero (drho_ow)
                   - 2.0 * dRhoW / safe_nonzero (rho_w));
            }

        // t_ow = clamp01((beta_o - B1)/(B2 - B1)); eta_ow = t_ow^n_ow.
        const double t_ow = clamp01 ((beta_o - B1_ow) / (B2_ow - B1_ow));
        const bool   t_ow_interior = (t_ow > 0.0 && t_ow < 1.0);

        double D_eta_ow_D_beta_o = 0.0;
        if (t_ow_interior)
          {
            const double inv_ow = 1.0 / (B2_ow - B1_ow);
            // eta_ow = t_ow^n; d(eta_ow)/d(b_o) = n * t_ow^(n-1) * inv_ow.
            double t_pow_m1 = 1.0;
            for (int k = 0; k < n_ow - 1; ++k) t_pow_m1 *= t_ow;
            D_eta_ow_D_beta_o = static_cast<double> (n_ow) * t_pow_m1 * inv_ow;
          }

        // C0_ow = 1 + (A' - 1)*eta_ow.
        double D_C0_ow_D_beta_o = (A_ow - 1.0) * D_eta_ow_D_beta_o;

        // V_d_ow = drift_incl_mult * 1.53 * v_c_ow * (1 - bC) / sqrt(bC*rho_o/rho_w + 1 - bC).
        const double bC      = beta_o * C0_ow;
        const double num_ow  = (1.0 - bC);
        const double den_arg_ow = bC * (rho_o / safe_nonzero (rho_w)) + 1.0 - bC;
        const double den_ow  = sqrt (std::max (den_arg_ow, eps_den));
        const double ro_over_rw = rho_o / safe_nonzero (rho_w);

        std::vector<double> D_V_d_ow_partial (nseg_vars, 0.0);
        double D_V_d_ow_D_beta_o = 0.0;
        {
          const double prefac = drift_incl_mult * 1.53;
          for (unsigned int id = 0; id < nseg_vars; ++id)
            {
              const double dVc    = D_v_c_ow_partial[id];
              const double dRhoO  = D_phase_rho_id (id, PHASE_OIL);
              const double dRhoW  = D_phase_rho_id (id, PHASE_WATER);
              const double dRR    = (dRhoO * rho_w - rho_o * dRhoW) / (rho_w * rho_w);
              const double d_den_arg = bC * dRR;  // bC has no partial via x (only via b_o)
              const double dDen   = 0.5 / safe_nonzero (den_ow) * d_den_arg;
              D_V_d_ow_partial[id] = prefac *
                  ( dVc * num_ow / safe_nonzero (den_ow)
                  - v_c_ow * num_ow * dDen / (den_ow * den_ow) );
            }
          const double dBC_b   = C0_ow + beta_o * D_C0_ow_D_beta_o;
          const double dNum_b  = -dBC_b;
          const double d_den_arg_b = dBC_b * ro_over_rw - dBC_b;
          const double dDen_b  = 0.5 / safe_nonzero (den_ow) * d_den_arg_b;
          D_V_d_ow_D_beta_o = prefac *
              ( v_c_ow * dNum_b / safe_nonzero (den_ow)
              - v_c_ow * num_ow * dDen_b / (den_ow * den_ow) );
        }

        // v_o = C0_ow * v_l + V_d_ow.
        std::vector<double> D_v_o_partial (nseg_vars, 0.0);
        double D_v_o_D_alpha_g = C0_ow * D_v_l_D_alpha_g;
        double D_v_o_D_beta_o  = D_C0_ow_D_beta_o * v_l + D_V_d_ow_D_beta_o;
        for (unsigned int id = 0; id < nseg_vars; ++id)
          D_v_o_partial[id] = C0_ow * D_v_l_partial[id] + D_V_d_ow_partial[id];

        // v_w = f_ow * v_l - g_ow * V_d_ow,  f_ow = (1 - bC)/(1 - b_o), g_ow = b_o/(1 - b_o).
        const double one_minus_bo = std::max (eps_den, 1.0 - beta_o);
        const double f_ow = (1.0 - bC) / one_minus_bo;
        const double g_ow = beta_o     / one_minus_bo;

        std::vector<double> D_v_w_partial (nseg_vars, 0.0);
        double D_v_w_D_alpha_g = f_ow * D_v_l_D_alpha_g;
        double D_v_w_D_beta_o  = 0.0;
        {
          const double dBC_b    = C0_ow + beta_o * D_C0_ow_D_beta_o;
          const double dF_ow_b  = (-dBC_b * one_minus_bo - (1.0 - bC) * (-1.0))
                                  / (one_minus_bo * one_minus_bo);
          const double dG_ow_b  = 1.0 / (one_minus_bo * one_minus_bo);
          D_v_w_D_beta_o = dF_ow_b * v_l - dG_ow_b * V_d_ow - g_ow * D_V_d_ow_D_beta_o;

          for (unsigned int id = 0; id < nseg_vars; ++id)
            D_v_w_partial[id] = f_ow * D_v_l_partial[id] - g_ow * D_V_d_ow_partial[id];
        }

        // ===================================================================
        // IFT 2x2 for d(alpha_g)/d(x) and d(beta_o)/d(x).
        //
        //   R_g = alpha_g * v_g - vs_g_in
        //   R_o = (1 - alpha_g) * beta_o * v_o - vs_o_in
        //
        //   J11 = dR_g/d(a_g) = v_g + a_g * D_v_g_D_alpha_g
        //   J12 = dR_g/d(b_o) = a_g * D_v_g_D_beta_o  (0 here)
        //   J21 = dR_o/d(a_g) = -b_o*v_o + (1-a_g)*b_o*D_v_o_D_alpha_g
        //   J22 = dR_o/d(b_o) = (1-a_g)*(v_o + b_o*D_v_o_D_beta_o)
        //
        //   rhs_g(id) = a_g * D_v_g_partial - D_vsg_in[id]
        //   rhs_o(id) = (1-a_g)*b_o * D_v_o_partial - D_vso_in[id]
        //
        //   J * [da_g; db_o] = - [rhs_g; rhs_o]   -> da_g, db_o.
        // ===================================================================
        const double J11 = v_g + alpha_g * D_v_g_D_alpha_g;
        const double J12 = alpha_g * D_v_g_D_beta_o;
        const double J21 = -beta_o * v_o + (1.0 - alpha_g) * beta_o * D_v_o_D_alpha_g;
        const double J22 = (1.0 - alpha_g) * (v_o + beta_o * D_v_o_D_beta_o);
        const double detJ = J11 * J22 - J12 * J21;

        std::vector<double> D_alpha_g_D_seg_vars (nseg_vars, 0.0);
        std::vector<double> D_beta_o_D_seg_vars  (nseg_vars, 0.0);

        if (fabs (detJ) > eps_den)
          {
            for (unsigned int id = 0; id < nseg_vars; ++id)
              {
                const double rg = alpha_g * D_v_g_partial[id] - D_vsg_in[id];
                const double ro = (1.0 - alpha_g) * beta_o * D_v_o_partial[id] - D_vso_in[id];
                D_alpha_g_D_seg_vars[id] = (-J22 * rg + J12 * ro) / detJ;
                D_beta_o_D_seg_vars[id]  = ( J21 * rg - J11 * ro) / detJ;
              }
          }

        std::vector<double> D_alpha_l_D_seg_vars (nseg_vars, 0.0);
        std::vector<double> D_alpha_o_D_seg_vars (nseg_vars, 0.0);
        std::vector<double> D_alpha_w_D_seg_vars (nseg_vars, 0.0);
        for (unsigned int id = 0; id < nseg_vars; ++id)
          {
            D_alpha_l_D_seg_vars[id] = -D_alpha_g_D_seg_vars[id];
            D_alpha_o_D_seg_vars[id] = -beta_o * D_alpha_g_D_seg_vars[id]
                                       + (1.0 - alpha_g) * D_beta_o_D_seg_vars[id];
            D_alpha_w_D_seg_vars[id] = -(1.0 - beta_o) * D_alpha_g_D_seg_vars[id]
                                       - (1.0 - alpha_g) * D_beta_o_D_seg_vars[id];
          }

        // Total phase velocity derivatives — chain through a_g and b_o.
        std::vector<double> D_v_g_D_seg_vars (nseg_vars, 0.0);
        std::vector<double> D_v_l_D_seg_vars (nseg_vars, 0.0);
        std::vector<double> D_v_o_D_seg_vars (nseg_vars, 0.0);
        std::vector<double> D_v_w_D_seg_vars (nseg_vars, 0.0);
        for (unsigned int id = 0; id < nseg_vars; ++id)
          {
            const double dAg = D_alpha_g_D_seg_vars[id];
            const double dBo = D_beta_o_D_seg_vars[id];
            D_v_g_D_seg_vars[id] = D_v_g_partial[id] + D_v_g_D_alpha_g * dAg + D_v_g_D_beta_o * dBo;
            D_v_l_D_seg_vars[id] = D_v_l_partial[id] + D_v_l_D_alpha_g * dAg + D_v_l_D_beta_o * dBo;
            D_v_o_D_seg_vars[id] = D_v_o_partial[id] + D_v_o_D_alpha_g * dAg + D_v_o_D_beta_o * dBo;
            D_v_w_D_seg_vars[id] = D_v_w_partial[id] + D_v_w_D_alpha_g * dAg + D_v_w_D_beta_o * dBo;
          }

        // Store C_0, V_d and their totals (same chain).
        wsncs->wsn_C_0              = C0;
        wsncs->wsn_drift_velocity   = V_d;
        wsncs->wsn_C_0_OW           = C0_ow;
        wsncs->wsn_drift_velocity_OW = V_d_ow;
        for (unsigned int id = 0; id < nseg_vars; ++id)
          {
            const double dAg = D_alpha_g_D_seg_vars[id];
            const double dBo = D_beta_o_D_seg_vars[id];
            wsncs->D_C0_D_seg_vars[id]                = D_C0_partial[id] + D_C0_D_alpha_g * dAg;
            wsncs->D_drift_velocity_D_seg_vars[id]    = D_V_d_partial[id] + D_V_d_D_alpha_g * dAg;
            wsncs->D_C0_OW_D_seg_vars[id]             = D_C0_ow_D_beta_o * dBo;
            wsncs->D_drift_velocity_OW_D_seg_vars[id] = D_V_d_ow_partial[id] + D_V_d_ow_D_beta_o * dBo;
          }

        // ===================================================================
        // Mixture density (DF-weighted) and its derivative.
        // rho_avg = a_g*rho_g + a_o*rho_o + a_w*rho_w.
        // ===================================================================
        const double rho_avg = alpha_g * rho_g + alpha_o * rho_o + alpha_w * rho_w;
        wsncs->rho_avg_DF = rho_avg;
        for (unsigned int id = 0; id < nseg_vars; ++id)
          {
            const double dAg = D_alpha_g_D_seg_vars[id];
            const double dAo = D_alpha_o_D_seg_vars[id];
            const double dAw = D_alpha_w_D_seg_vars[id];
            const double dRG = D_phase_rho_id (id, PHASE_GAS);
            const double dRO = D_phase_rho_id (id, PHASE_OIL);
            const double dRW = D_phase_rho_id (id, PHASE_WATER);
            wsncs->D_rho_avg_D_seg_vars[id] =
                dAg * rho_g + alpha_g * dRG
              + dAo * rho_o + alpha_o * dRO
              + dAw * rho_w + alpha_w * dRW;
          }

        // Save converged DF state for next-step continuation.
        wsncs->phase_S[PHASE_GAS]   = alpha_g;
        wsncs->phase_S[PHASE_OIL]   = alpha_o;
        wsncs->phase_S[PHASE_WATER] = alpha_w;

        // ===================================================================
        // Component rates and their Jacobian.
        //
        //   q_c = sum_ip  alpha_ip * v_ip * area * x[ic,ip] * xi[ip]
        //
        // d(q_c)/d(x_id) = sum_ip [
        //     (d(alpha_ip)/d(x_id) * v_ip + alpha_ip * d(v_ip)/d(x_id))
        //        * area * x[ic,ip] * xi[ip]
        //   + alpha_ip * v_ip * area * (
        //        d(x[ic,ip])/d(x_id) * xi[ip]
        //      + x[ic,ip] * d(xi[ip])/d(x_id)
        //     )
        // ]
        //
        // CRITICAL: d(x[ic,ip])/d(x_id) uses per-ic, per-id flash info from
        // component_phase_D_x, and d(xi[ip])/d(x_id) uses phase_D_xi per id.
        // This is what prevents z-column collinearity in the Jacobian.
        // ===================================================================
        for (auto ic = mp.nc0; ic < mp.nc; ++ic)
          wsncs->wsn_component_rate[ic] = 0.0;

        struct phase_binding_t { double alpha; double v; const std::vector<double> *dAlpha; const std::vector<double> *dV; };
        phase_binding_t pb[3];
        pb[PHASE_GAS]   = { alpha_g, v_g, &D_alpha_g_D_seg_vars, &D_v_g_D_seg_vars };
        pb[PHASE_OIL]   = { alpha_o, v_o, &D_alpha_o_D_seg_vars, &D_v_o_D_seg_vars };
        pb[PHASE_WATER] = { alpha_w, v_w, &D_alpha_w_D_seg_vars, &D_v_w_D_seg_vars };

        for (unsigned int ip = 0; ip < mp.np; ++ip)
          {
            const double a_ip = pb[ip].alpha;
            const double v_ip = pb[ip].v;
            const double xi_ip = element_status->phase_xi[ip];
            const double flux_ip = a_ip * v_ip * area;

            for (auto ic = mp.nc0; ic < mp.nc; ++ic)
              {
                const double x_icip = element_status->component_phase_x[ic * mp.np + ip];
                wsncs->wsn_component_rate[ic] += flux_ip * x_icip * xi_ip;

                for (unsigned int id = 0; id < nseg_vars; ++id)
                  {
                    const double dAlpha = (*pb[ip].dAlpha)[id];
                    const double dV     = (*pb[ip].dV)[id];
                    const double dFlux  = (dAlpha * v_ip + a_ip * dV) * area;

                    double dX = 0.0;
                    double dXi = 0.0;
                    if (id < 1U + mp.nc)
                      {
                        dX  = element_status->component_phase_D_x[(id * mp.nc + ic) * mp.np + ip];
                        dXi = element_status->phase_D_xi[id * mp.np + ip];
                      }

                    wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id] +=
                        dFlux * x_icip * xi_ip
                      + flux_ip * (dX * xi_ip + x_icip * dXi);
                  }
              }
          }

        // ===================================================================
        // Mixture mass rate and its Jacobian.
        //   Mm = sum_ic  q_c * MW_c
        // ===================================================================
        wsncs->wsn_mixture_mass_rate = 0.0;
        wsncs->wsn_mmw = 0.0;
        for (auto ic = mp.nc0; ic < mp.nc; ++ic)
          {
            wsncs->wsn_mixture_mass_rate += wsncs->wsn_component_rate[ic] * component_molar_weights[ic];
            wsncs->wsn_mmw               += component_z_for_flow[ic]      * component_molar_weights[ic];
            for (unsigned int id = 0; id < nseg_vars; ++id)
              wsncs->D_mixture_mass_rate_D_seg_vars[id] +=
                  wsncs->D_q_c_D_seg_vars[(1U + mp.nc + 1U) * ic + id] * component_molar_weights[ic];
          }

        // ===================================================================
        // Diagnostic log (one line per converged segment).
        // ===================================================================
        PR1 (LOG_WELL_SECTION, LOG_MESSAGE, WELL_PARAMS,
             "MERGEN_DF16: seg={} it={} err_g={} err_b={} "
             "a_g={} a_o={} a_w={} b_o={} "
             "v_g={} v_l={} v_o={} v_w={} j_m={} "
             "C0={} V_d={} C0_ow={} V_d_ow={} "
             "sigma_gl={} sigma_ow={} rho_g={} rho_l={} rho_avg={} "
             "Ku={} Dhat={} v_sgf={} K_gl={} "
             "vs_g_in={} vs_o_in={} vs_w_in={} seed_prev={}\n",
             seg.wsn->wsn_index, it, err_g, err_b,
             alpha_g, alpha_o, alpha_w, beta_o,
             v_g, v_l, v_o, v_w, j_m,
             C0, V_d, C0_ow, V_d_ow,
             sigma_gl, sigma_ow, rho_g, rho_l, rho_avg,
             Ku, Dhat, v_sgf, K_gl,
             vs_g_in, vs_o_in, vs_w_in, (int) use_prev_seed);
      }  // non-top segment block
  }    // if (element_status)
