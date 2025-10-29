import csv
import numpy as np
import emerge as em
from ifalib import build_mifa,get_resonant_frequency
from optimize_lib import _fmt_params_singleline_raw

Z0 = 50.0

def s_to_zin(S11):
    return Z0 * (1.0 + S11) / (1.0 - S11)

def find_fr(freq, S11, f_hint):
    fr= get_resonant_frequency(S11=S11, freq_dense=freq)
    return float(fr)

def eval_impedance_features(parameters):
    # print(f"parameters: {_fmt_params_singleline_raw(parameters)}")
    model, S11, freq, _, _, _ = build_mifa(parameters,
        view_mesh=False, view_model=False, run_simulation=True,
        compute_farfield=False, loglevel="WARNING", solver=em.EMSolver.CUDSS)
    if S11 is None:
        return None
    f0 = parameters['f0']
    fr = find_fr(freq, S11, f0)
    # robust complex interpolation at f0
    S = np.interp(f0, freq, S11.real) + 1j*np.interp(f0, freq, S11.imag)
    Zin0 = s_to_zin(S)
    R0, X0 = float(np.real(Zin0)), float(np.imag(Zin0))
    return dict(fr=fr, R0=R0, X0=X0, freq=freq, S11=S11)

def finite_diff_sensitivities(params, keys, base_feat, rel_step=0.02, abs_mins=None):
    dfr, dR, dX = [], [], []
    base = dict(params)
    for k in keys:
        p0 = base[k]
        step = (rel_step*abs(p0) if p0 != 0 else rel_step)
        if abs_mins and k in abs_mins:
            step = max(step, abs_mins[k])
        pd = dict(base); pd[k] = p0 + step
        feat = eval_impedance_features(pd)
        if feat is None:
            pd[k] = p0 - step
            feat = eval_impedance_features(pd)
        if feat is None:
            dfr.append(0.0); dR.append(0.0); dX.append(0.0); continue
        dfr.append((feat['fr'] - base_feat['fr'])/step)
        dR.append((feat['R0'] - base_feat['R0'])/step)
        dX.append((feat['X0'] - base_feat['X0'])/step)
    return np.array(dfr), np.array(dR), np.array(dX)

def _score(feat, f0):
    # normalized squared errors (tweak weights as you like)
    s_fr = (feat['fr'] - f0)**2 / (0.02*f0)**2    # 2% band
    s_R  = ((feat['R0'] - Z0)/Z0)**2
    s_X  = (feat['X0']/Z0)**2
    return s_fr + 0.5*s_R + 0.7*s_X

def _fmt_table(headers, rows, colw=11):
    line = " ".join(h.ljust(colw) for h in headers)
    body = "\n".join(" ".join(str(v)[:colw].ljust(colw) for v in r) for r in rows)
    return line + "\n" + body

def _mini_sweep_report(params, center, df=15e6):
    """3-point sweep at f0±df to show R/X slope."""
    p = dict(params)
    p['f1'], p['f0'], p['f2'] = center - df, center, center + df
    p['freq_points'] = 3
    m, S11, f, *_ = build_mifa(p, view_mesh=False, view_model=False, run_simulation=True,
                               compute_farfield=False, loglevel="ERROR", solver=em.EMSolver.CUDSS)
    if S11 is None:
        print("  Mini-sweep: simulation failed.")
        return
    Zin = s_to_zin(S11)
    R = np.real(Zin); X = np.imag(Zin)
    print(f"  Mini-sweep @ f0±{df/1e6:.0f} MHz:  X[lo,0,hi]=[{X[0]:+.1f}, {X[1]:+.1f}, {X[2]:+.1f}] Ω;  "
          f"R[lo,0,hi]=[{R[0]:.1f}, {R[1]:.1f}, {R[2]:.1f}] Ω")

def physics_optimize_mifa(
    initial_params: dict,
    bounds: dict,
    keys_to_tune: list,
    max_iters=12,
    rel_probe=0.02,
    trust_frac=0.15,
    wR=0.8, wX=1.0,           # base weights (will be adapted by phase)
    ridge=1e-6,
    verbose=2,                # 0=silent, 1=brief, 2=detailed
    csv_log_path=None,        # e.g., "opt_trace.csv"
    # --- new knobs below ---
    per_param_ridge: dict = None,     # e.g., {'ifa_w1':5e-5, 'ifa_w2':5e-5, ...}
    trust_caps: dict = None,          # fraction of span per param, e.g., {'ifa_w1':0.05,...}
    x_phase_thresh_ohm: float = 5.0,  # switch to phase 2 when |X| <= this
    mini_sweep_df_hz: float = 15e6    # 3-point sweep offset
):
    """
    Verbose, physics-informed optimizer with:
    - per-parameter ridge
    - two-phase weighting (phase 1: kill X; phase 2: refine R)
    - per-parameter trust caps
    - mini 3-point sweep around f0 each iteration
    """
    params = dict(initial_params)

    # Ensure side freqs differ from f0 to stabilize fr detection if far off
    if params.get('f1', params['f0']) == params['f0']: params['f1'] = params['f0'] - 2.5e8
    if params.get('f2', params['f0']) == params['f0']: params['f2'] = params['f0'] + 2.5e8

    spans  = {k: (bounds[k][1] - bounds[k][0]) for k in keys_to_tune}
    trust  = {k: trust_frac*max(spans[k], 1e-12) for k in keys_to_tune}

    # # Apply per-parameter trust caps (fractions of span)
    # if trust_caps:
    #     for k, frac in trust_caps.items():
    #         if k in trust and spans[k] > 0:
    #             trust[k] = min(trust[k], float(frac) * spans[k])

    # CSV logger
    csv_writer = None
    if csv_log_path:
        fcsv = open(csv_log_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(["iter","fr_GHz","R_ohm","X_ohm","score",
                             *[f"p:{k}" for k in keys_to_tune],
                             *[f"trust:{k}" for k in keys_to_tune]])

    for it in range(1, max_iters+1):
        feat = eval_impedance_features(params)
        if feat is None:
            if verbose:
                print(f"[it {it}] simulation failed; shrinking trust and continuing")
            for k in trust: trust[k] *= 0.5
            continue

        f0, fr, R0, X0 = params['f0'], feat['fr'], feat['R0'], feat['X0']
        err_fr, err_R, err_X = (f0 - fr), (Z0 - R0), (-X0)

        # --- Phase selection: prioritize X first, then R ---
        if abs(X0) > x_phase_thresh_ohm:
            wR_eff, wX_eff = 0.3*wR, 1.4*wX   # kill reactance, keep R changes restrained
            phase = 1
        else:
            wR_eff, wX_eff = 1.2*wR, 0.8*wX   # bring R→50 without re-introducing big X
            phase = 2

        if verbose >= 1:
            print(f"[it {it}] fr={fr/1e9:.4f} GHz  @f0: R={R0:.2f}Ω X={X0:.2f}Ω  "
                  f"targets: Δfr={err_fr/1e6:.2f} MHz, ΔR={err_R:.2f}, ΔX={err_X:.2f}  (phase {phase})")

        # Sensitivities
        dfr, dR, dX = finite_diff_sensitivities(params, keys_to_tune, feat, rel_step=rel_probe)

        # Stack system  A Δp = b
        A = np.vstack([dfr, wR_eff*dR, wX_eff*dX])
        b = np.array([err_fr, wR_eff*err_R, wX_eff*err_X], dtype=float)

        # Regularized LS with per-parameter ridge
        AT = A.T
        if per_param_ridge:
            Lam = np.diag([float(per_param_ridge.get(k, ridge)) for k in keys_to_tune])
        else:
            Lam = ridge*np.eye(len(keys_to_tune))
        H = AT @ A + Lam
        g = AT @ b
        try:
            dp_unclipped = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dp_unclipped = np.zeros_like(g)

        # Trust & bounds clipping
        dp_trust = []
        dp_clipped = []
        new_params = dict(params)
        for k, dpk in zip(keys_to_tune, dp_unclipped):
            dlim = trust[k]
            # apply per-parameter trust caps again here (robustness)
            if trust_caps and k in trust_caps:
                dlim = min(dlim, trust_caps[k]*spans[k])
            dpt = float(np.clip(dpk, -dlim, dlim))
            dp_trust.append(dpt)
            cand = params[k] + dpt
            lo, hi = bounds[k]
            cand_c = float(np.clip(cand, lo, hi))
            dp_clipped.append(cand_c - params[k])
            new_params[k] = cand_c

        # Predictions & contributions
        dp_vec = np.array(dp_trust)
        pred = A @ dp_vec
        labels = ["Δfr_pred [Hz]", "wR·ΔR_pred [Ω]", "wX·ΔX_pred [Ω]"]
        print("\n  Predicted target changes (linear model):")
        for lab, val, tgt in zip(labels, pred, b):
            print(f"    {lab}: {val:+.3e}  vs  target {tgt:+.3e}  (residual {val - tgt:+.3e})")

        print("\n  Per-parameter contributions:")
        row_names = ["dfr", "wR·dR", "wX·dX"]
        for rname, row in zip(row_names, A):
            print(f"   {rname}:")
            for k, a_ik, dpk in zip(keys_to_tune, row, dp_vec):
                print(f"      {k:<30}  a={a_ik:+.3e} · Δp={dpk:+.3e}  → {a_ik*dpk:+.3e}")

        # Linear model residual norm (for intuition only)
        pred_err = A @ np.array(dp_trust) - b
        pred_norm = float(np.linalg.norm(pred_err))
        if verbose >= 2:
            # Print Jacobians
            print("\n  Jacobian (per parameter):  dfr/ dp,  wR*dR/dp,  wX*dX/dp")
            rows = []
            for i,k in enumerate(keys_to_tune):
                rows.append([
                    k,
                    f"{dfr[i]:+.3e}",
                    f"{(wR_eff*dR[i]):+.3e}",
                    f"{(wX_eff*dX[i]):+.3e}"
                ])
            print(_fmt_table(["param","dfr","wR*dR","wX*dX"], rows))

            # Print steps
            step_rows = []
            for k, u, t, c in zip(keys_to_tune, dp_unclipped, dp_trust, dp_clipped):
                step_rows.append([k,
                                  f"{u:+.3e}",  # raw LS
                                  f"{t:+.3e}",  # trust-clipped
                                  f"{c:+.3e}",  # trust+bounds applied
                                  f"{trust[k]:.3e}"]) # current trust
            print("\n  Proposed steps (Δp):")
            print(_fmt_table(["param","LS","trust","clipped","trust_rad"], step_rows))
            print(f"\n  Linear model residual norm (trust-clipped Δp): {pred_norm:.3e}")

        # Evaluate candidate
        feat_new = eval_impedance_features(new_params)
        if feat_new is None:
            for k in trust: trust[k] *= 0.6
            if verbose:
                print("  -> candidate failed; shrinking trust.\n")
            continue

        score_old = _score(feat, f0)
        score_new = _score(feat_new, f0)
        accepted  = score_new < score_old

        if verbose >= 1:
            print(f"  Score: old={score_old:.4f} new={score_new:.4f}  "
                  f"({'accepted' if accepted else 'rejected'})")

        if accepted:
            params = new_params
            # expand trust a bit on success
            for k in trust:
                trust[k] = min(trust[k]*1.4, spans[k])
            if verbose:
                print("  -> accepted; expanding trust.")
                #mini_sweep_report(params, f0, df=mini_sweep_df_hz)
                print()
        else:
            # shrink trust on no-improve
            for k in trust:
                trust[k] *= 0.6
            if verbose:
                print("  -> rejected; shrinking trust.\n")

        if csv_writer:
            csv_writer.writerow(
                [it, feat_new['fr']/1e9, feat_new['R0'], feat_new['X0'], score_new,
                 *[params[k] for k in keys_to_tune],
                 *[trust[k] for k in keys_to_tune]]
            )

    if csv_writer:
        fcsv.close()

    return params

# ---------------- Example run (yours) ----------------
parameters = { 'ifa_h': 0.006,
        'ifa_l': 0.027-0.00075,
        'ifa_w1': 0.0015,
        'ifa_w2': 0.0005,
        'ifa_wf': 0.0005,
        'ifa_fp': 0.0025,
        'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_te': 0.0005,
        'via_size': 0.0003, 'board_wsub': 0.014, 'board_hsub': 0.025, 'board_th': 0.0015,
        'mifa_meander': 0.0015, 'mifa_meander_edge_distance': 0.0005,
        'f1': 2.3e+09, 'f0': 2.45e+09, 'f2': 2.6e+09, 'freq_points': 5,
        'mesh_boundry_size_divisor': 0.33, 'mesh_wavelength_fraction': 0.2, 'lambda_scale': 1 }

# Choose knobs with clear L/C effects:
keys = [
    'ifa_l','ifa_fp','ifa_w1','ifa_w2','ifa_wf'
]
bounds = {k: (0.8*parameters[k], 1.2*parameters[k]) for k in keys}
for k,v in parameters.items():
    bounds.setdefault(k, (v, v))

# # Make widths "expensive" and trust-capped; steer X with C knobs first
# per_param_ridge = {
#     'ifa_w2': 5e-5, 'ifa_wf': 5e-5,          # widths move only if needed
#     'mifa_meander_edge_distance': 2e-5, 'ifa_w1': 2e-5,                  # mild preference not to overuse
#     'ifa_l': 1e-6, 'mifa_meander': 2e-6, 'ifa_fp': 2e-6      # cheap (primary steering knobs)
# }

# Make widths "expensive" and trust-capped; steer X with C knobs first
per_param_ridge = {
    'ifa_w2': 5e-5, 'ifa_wf': 5e-5,          # widths move only if needed
    'ifa_w1': 2e-5,                  # mild preference not to overuse
    'ifa_l': 1e-6, 'ifa_fp': 2e-6      # cheap (primary steering knobs)
}


best = physics_optimize_mifa(
    initial_params=parameters,
    bounds=bounds,
    keys_to_tune=keys,
    max_iters=1000,
    rel_probe=0.0005,
    trust_frac=0.005,
    wR=0.8, wX=1.0,
    verbose=1,
    csv_log_path="opt_trace.csv",
    x_phase_thresh_ohm=5.0,
    mini_sweep_df_hz=15e6
)

print("Best params:", _fmt_params_singleline_raw(best))
