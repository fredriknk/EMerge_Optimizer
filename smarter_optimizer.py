import numpy as np
import emerge as em
from ifalib import build_mifa
from optimize_lib import _fmt_params_singleline_raw

Z0 = 50.0

def s_to_zin(S11):
    return Z0 * (1.0 + S11) / (1.0 - S11)

def find_fr(freq, S11, f_hint):
    """Find resonance ~ root of Im(Zin) near f_hint. Uses local window + interpolation."""
    Zin = s_to_zin(S11)
    X = np.imag(Zin)
    # pick window around f_hint
    idx = np.argsort(np.abs(freq - f_hint))[:15]
    idx = np.sort(idx)
    f_loc, X_loc = freq[idx], X[idx]
    # Find sign change or minimum |X|
    k = np.argmin(np.abs(X_loc))
    # Try linear interpolation if neighbors change sign
    if 0 < k < len(f_loc)-1 and X_loc[k-1]*X_loc[k+1] <= 0:
        f1, x1 = f_loc[k-1], X_loc[k-1]
        f2, x2 = f_loc[k+1], X_loc[k+1]
        if x2 != x1:
            return float(f1 - x1*(f2-f1)/(x2-x1))
    return float(f_loc[k])

def eval_impedance_features(parameters):
    """Run sim, return fr, R0, X0, S11 arrays etc."""
    model, S11, freq, _, _, _ = build_mifa(parameters,
        view_mesh=False, view_model=False, run_simulation=True,
        compute_farfield=False, loglevel="ERROR", solver=em.EMSolver.CUDSS)
    if S11 is None:
        return None
    f0 = parameters['f0']
    # resonance close to f0
    fr = find_fr(freq, S11, f0)
    # values at f0 (interpolate complex S11 at f0)
    # (you had helpers; here is inline robust interp)
    S = np.interp(f0, freq, S11.real) + 1j*np.interp(f0, freq, S11.imag)
    Zin0 = s_to_zin(S)
    R0, X0 = float(np.real(Zin0)), float(np.imag(Zin0))
    return dict(fr=fr, R0=R0, X0=X0, freq=freq, S11=S11)

def finite_diff_sensitivities(params, keys, base_feat, rel_step=0.02, abs_mins=None):
    """Return Jacobians wrt keys: dfr/dp, dR0/dp, dX0/dp."""
    dfr, dR, dX = [], [], []
    base = dict(params)
    for k in keys:
        p0 = base[k]
        if abs_mins and k in abs_mins:
            step = max(rel_step*abs(p0), abs_mins[k])
        else:
            step = rel_step*abs(p0) if p0 != 0 else rel_step
        # One-sided step clipped to bounds handled outside (we assume valid here)
        pd = dict(base)
        pd[k] = p0 + step
        feat = eval_impedance_features(pd)
        if feat is None:
            # if sim failed, try negative step
            pd[k] = p0 - step
            feat = eval_impedance_features(pd)
        if feat is None:
            # if still failed, zero sensitivity for this key
            dfr.append(0.0); dR.append(0.0); dX.append(0.0)
            continue
        dfr.append((feat['fr'] - base_feat['fr'])/step)
        dR.append((feat['R0'] - base_feat['R0'])/step)
        dX.append((feat['X0'] - base_feat['X0'])/step)
    return np.array(dfr), np.array(dR), np.array(dX)

def physics_optimize_mifa(initial_params: dict,
                          bounds: dict,
                          keys_to_tune: list,
                          max_iters=12,
                          rel_probe=0.02,
                          trust_frac=0.15,
                          wR=0.8, wX=1.0,
                          ridge=1e-6,
                          verbose=True):
    """
    bounds: {k: (lo, hi)}; keys_to_tune is the subset we move.
    Solves a small least-squares system each iteration to move (fr->f0, R0->Z0, X0->0).
    """
    params = dict(initial_params)
    # make sure f1,f2 bracket f0 loosely (helps find fr if off)
    if params['f1'] == params['f0']: params['f1'] = params['f0'] - 2.5e8
    if params['f2'] == params['f0']: params['f2'] = params['f0'] + 2.5e8

    # trust radii per parameter (fraction of range)
    spans = {k: (bounds[k][1] - bounds[k][0]) for k in keys_to_tune}
    trust = {k: trust_frac*max(spans[k], 1e-9) for k in keys_to_tune}

    hist = []
    for it in range(1, max_iters+1):
        feat = eval_impedance_features(params)
        if feat is None:
            if verbose: print(f"[it {it}] simulation failed; shrinking trust and continuing")
            for k in trust: trust[k] *= 0.5
            continue

        f0 = params['f0']
        fr, R0, X0 = feat['fr'], feat['R0'], feat['X0']
        err_fr = f0 - fr
        err_R  = Z0 - R0
        err_X  = -X0

        if verbose:
            print(f"[it {it}] fr={fr/1e9:.4f} GHz  @f0: R={R0:.2f}Ω X={X0:.2f}Ω  "
                  f"targets: Δfr={err_fr/1e6:.1f} MHz, ΔR={err_R:.1f}, ΔX={err_X:.1f}")

        # Build sensitivities (Jacobian rows)
        dfr, dR, dX = finite_diff_sensitivities(params, keys_to_tune, feat, rel_step=rel_probe)

        # Stack system: [dfr; wR dR; wX dX] Δp = [err_fr; wR err_R; wX err_X]
        A = np.vstack([dfr, wR*dR, wX*dX])
        b = np.array([err_fr, wR*err_R, wX*err_X], dtype=float)

        # Tikhonov-regularized least squares (bias small steps; also lets us add priors later)
        # Solve min ||A Δp - b||^2 + ridge ||Δp||^2
        # (A^T A + λI) Δp = A^T b
        AT = A.T
        H = AT @ A + ridge*np.eye(len(keys_to_tune))
        g = AT @ b
        try:
            dp = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dp = np.zeros_like(g)

        # Clip to trust & bounds
        new_params = dict(params)
        improved = False
        for k, dpk in zip(keys_to_tune, dp):
            # trust clip
            dlim = trust[k]
            dpk = float(np.clip(dpk, -dlim, dlim))
            cand = params[k] + dpk
            lo, hi = bounds[k]
            cand = float(np.clip(cand, lo, hi))
            new_params[k] = cand

        # Evaluate candidate
        feat_new = eval_impedance_features(new_params)
        if feat_new is None:
            # shrink trust and retry next iter
            for k in trust: trust[k] *= 0.6
            hist.append((params, feat, False))
            if verbose: print(f"  -> candidate failed; shrinking trust.")
            continue

        # Decide improvement: combine weighted goals
        def score(feat):
            # squared errors with Hz→Ω scale harmonized
            s_fr = (feat['fr'] - f0)**2 / (0.02*f0)**2     # 2% of f0 scale
            s_R  = ((feat['R0'] - Z0)/Z0)**2
            s_X  = (feat['X0']/Z0)**2
            return s_fr + 0.5*s_R + 0.7*s_X

        if score(feat_new) < score(feat):
            params = new_params
            # expand trust a bit on success
            for k in trust:
                trust[k] = min(trust[k]*1.4, spans[k])
            improved = True
            if verbose:
                print("  -> accepted; expanding trust.")
        else:
            # shrink trust on no-improve
            for k in trust:
                trust[k] *= 0.6
            if verbose:
                print("  -> rejected; shrinking trust.")

        hist.append((params, feat_new if improved else feat, improved))

    return params, hist

parameters = { 'ifa_h': 0.006, 
        'ifa_l': 0.027-0.00075, 
        'ifa_w1': 0.0015, 
        'ifa_w2': 0.0005, 
        'ifa_wf': 0.0005, 
        'ifa_fp': 0.0025,
        'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_te': 0.0005, 
        'via_size': 0.0003, 'board_wsub': 0.014, 'board_hsub': 0.025, 'board_th': 0.0015, 
        'mifa_meander': 0.0015, 'mifa_meander_edge_distance': 0.0005, 
        'f1': 2.4e+09, 'f0': 2.45e+09, 'f2': 2.5e+09, 'freq_points': 3, 
        'mesh_boundry_size_divisor': 0.5, 'mesh_wavelength_fraction': 0.5, 'lambda_scale': 0.5 }

# Choose knobs with clear L/C effects:
keys = [
    'ifa_l',                      # strong L (resonance)
    'mifa_meander',               # L (via path length / coupling)
    'mifa_meander_edge_distance', # C_to_gnd
    'ifa_fp',                     # feed pad area → C
    'ifa_w1','ifa_w2','ifa_wf'    # widths affect L (↓) and R (↓) and some C
]

# Set bounds you’re comfortable with (meters):
bounds = {
    k: (0.8*parameters[k], 1.2*parameters[k]) for k in keys
}
# freeze the rest
for k,v in parameters.items():
    bounds.setdefault(k, (v, v))

best_params, history = physics_optimize_mifa(
    initial_params=parameters,
    bounds=bounds,
    keys_to_tune=keys,
    max_iters=10,      # 8–15 is usually enough to “lock”
    rel_probe=0.02,    # 2% finite-diff step
    trust_frac=0.15,   # allow 15% of range per move initially
    wR=0.8, wX=1.0,
    verbose=True
)

print("\nBest params:", _fmt_params_singleline_raw(best_params))
