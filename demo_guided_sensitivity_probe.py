import math
from typing import Dict, List, Optional, Tuple

import numpy as np

mm = 1e-3

# Baseline seed (single-band, plain keys)
BASE_PARAMS: Dict[str, float] = {
    'ifa_h': 0.00773,
    'ifa_l': 0.02295,
    'ifa_w1': 0.000767,
    'ifa_w2': 0.000440,
    'ifa_wf': 0.000344,
    'ifa_fp': 0.00257,
    'ifa_e': 0.0005,
    'ifa_e2': 0.0005,
    'ifa_te': 0.0005,
    'via_size': 0.0003,
    'board_wsub': 0.014,
    'board_hsub': 0.025,
    'board_th': 0.0015,
    'mifa_meander': 0.00195,
    'mifa_low_dist': 0.00217,
    'f1': 2.3e9,
    'f0': 2.45e9,
    'f2': 2.7e9,
    'freq_points': 3,
    'mesh_boundary_size_divisor': 0.5,
    'mesh_wavelength_fraction': 0.5,
    'lambda_scale': 0.5,
}

# Start with a short list; add more once this proves useful.
PROBE_KEYS: List[str] = [
    'ifa_l',
    'ifa_h',
    'ifa_fp',
    'ifa_w2',
    'mifa_meander',
    'mifa_low_dist',
]

REL_STEP = 0.03      # +/- 3%
ABS_MIN_STEP = 0.05 * mm
SOLVER_NAME = 'CUDSS'  # fallback to PARDISO if unavailable

# Guided-update knobs
GUIDED_ITERS = 3
MAX_STEP_PER_PARAM_MM = 0.25
MAX_TOTAL_STEP_NORM_MM = 0.35
RIDGE_DAMPING = 1e-3

# Error vector = [f_res - f0 (MHz), Re(Zin)-50 (ohm), Im(Zin)-0 (ohm)]
ERROR_WEIGHTS = np.array([1.0, 0.25, 0.40], dtype=float)


def _interp_complex(x: np.ndarray, y: np.ndarray, xq: float) -> complex:
    re = np.interp(xq, x, y.real)
    im = np.interp(xq, x, y.imag)
    return re + 1j * im


def _gamma_to_zin(gamma: complex, z0: float = 50.0) -> complex:
    # Zin = Z0 * (1 + Gamma) / (1 - Gamma)
    denom = 1.0 - gamma
    if abs(denom) < 1e-12:
        return complex(np.inf, np.inf)
    return z0 * (1.0 + gamma) / denom


def _validate(params: Dict[str, float]) -> Tuple[bool, Optional[str]]:
    from ifa_validation import validate_ifa_params

    errs, _, _ = validate_ifa_params(params)
    if errs:
        return False, '; '.join(errs[:2])
    return True, None


def _run_metrics(params: Dict[str, float], solver_name: str = SOLVER_NAME) -> Dict[str, float]:
    import emerge as em
    from ifalib2 import build_mifa

    solver = getattr(em.EMSolver, solver_name, em.EMSolver.PARDISO)
    _, s11, freq_dense, *_ = build_mifa(
        params,
        view_skeleton=False,
        view_mesh=False,
        view_model=False,
        run_simulation=True,
        compute_farfield=False,
        solver=solver,
        loglevel='ERROR',
    )

    if s11 is None or freq_dense is None:
        raise RuntimeError('Simulation did not return S11/frequency data.')

    rl_db = -20.0 * np.log10(np.abs(s11))
    idx_res = int(np.argmax(rl_db))
    f_res = float(freq_dense[idx_res])

    f0 = float(params['f0'])
    gamma_f0 = _interp_complex(np.asarray(freq_dense), np.asarray(s11), f0)
    zin_f0 = _gamma_to_zin(gamma_f0, z0=50.0)

    return {
        'f_res_hz': f_res,
        'f0_hz': f0,
        'rl_f0_db': float(-20.0 * math.log10(abs(gamma_f0))),
        're_zin_f0': float(np.real(zin_f0)),
        'im_zin_f0': float(np.imag(zin_f0)),
    }


def _one_sided_derivative(base: Dict[str, float], shifted: Dict[str, float], delta: float) -> Dict[str, float]:
    return {
        'df_res_dparam': (shifted['f_res_hz'] - base['f_res_hz']) / delta,
        'dReZ_dparam': (shifted['re_zin_f0'] - base['re_zin_f0']) / delta,
        'dImZ_dparam': (shifted['im_zin_f0'] - base['im_zin_f0']) / delta,
        'dRL_dparam': (shifted['rl_f0_db'] - base['rl_f0_db']) / delta,
    }


def _error_vector(base: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            (base['f_res_hz'] - base['f0_hz']) / 1e6,
            base['re_zin_f0'] - 50.0,
            base['im_zin_f0'],
        ],
        dtype=float,
    )


def _print_metrics(label: str, m: Dict[str, float]) -> None:
    err = _error_vector(m)
    print(
        f"{label}: f_res={m['f_res_hz']/1e9:.6f} GHz, "
        f"RL@f0={m['rl_f0_db']:.2f} dB, "
        f"Zin@f0={m['re_zin_f0']:.2f} + j{m['im_zin_f0']:.2f} ohm, "
        f"err=[df={err[0]:+.2f} MHz, dRe={err[1]:+.2f} ohm, dIm={err[2]:+.2f} ohm]"
    )


def _collect_sensitivities(
    params: Dict[str, float],
    keys: List[str],
    *,
    verbose: bool = True,
) -> Tuple[Dict[str, float], List[Tuple[str, float, str, Dict[str, float]]]]:
    ok, msg = _validate(params)
    if not ok:
        raise ValueError(f'Baseline params invalid: {msg}')

    base = _run_metrics(params)
    if verbose:
        _print_metrics('Baseline', base)
        print('\nSensitivity probes (small perturbations):')

    results: List[Tuple[str, float, str, Dict[str, float]]] = []

    for key in keys:
        if key not in params:
            if verbose:
                print(f'- {key}: skipped (not in params)')
            continue

        x0 = float(params[key])
        step = max(abs(x0) * REL_STEP, ABS_MIN_STEP)

        p_plus = dict(params)
        p_plus[key] = x0 + step

        p_minus = dict(params)
        p_minus[key] = max(1e-9, x0 - step)

        plus_valid, plus_msg = _validate(p_plus)
        minus_valid, minus_msg = _validate(p_minus)

        m_plus = None
        m_minus = None

        if plus_valid:
            try:
                m_plus = _run_metrics(p_plus)
            except Exception as e:
                if verbose:
                    print(f'- {key} (+): simulation error: {e}')
        elif verbose:
            print(f'- {key} (+): invalid -> {plus_msg}')

        if minus_valid:
            try:
                m_minus = _run_metrics(p_minus)
            except Exception as e:
                if verbose:
                    print(f'- {key} (-): simulation error: {e}')
        elif verbose:
            print(f'- {key} (-): invalid -> {minus_msg}')

        deriv = None
        mode = None
        if (m_plus is not None) and (m_minus is not None):
            d_plus = _one_sided_derivative(base, m_plus, step)
            d_minus = _one_sided_derivative(base, m_minus, -step)
            deriv = {k: 0.5 * (d_plus[k] + d_minus[k]) for k in d_plus.keys()}
            mode = 'central'
        elif m_plus is not None:
            deriv = _one_sided_derivative(base, m_plus, step)
            mode = 'forward'
        elif m_minus is not None:
            deriv = _one_sided_derivative(base, m_minus, -step)
            mode = 'backward'
        else:
            if verbose:
                print(f'- {key}: no valid perturbation runs')
            continue

        results.append((key, step, mode, deriv))
        if verbose:
            print(
                f"- {key} ({mode}, step={step/mm:.4f} mm): "
                f"df_res/dp={deriv['df_res_dparam']/(1e6/mm):.3e} MHz/mm, "
                f"dReZ/dp={deriv['dReZ_dparam']*mm:.3e} ohm/mm, "
                f"dImZ/dp={deriv['dImZ_dparam']*mm:.3e} ohm/mm"
            )

    return base, results


def _rank_and_print(results: List[Tuple[str, float, str, Dict[str, float]]]) -> None:
    if not results:
        print('\nNo usable sensitivity results.')
        return

    print('\nTop levers by |df_res/dp|:')
    for key, _, _, deriv in sorted(results, key=lambda x: abs(x[3]['df_res_dparam']), reverse=True):
        print(f"- {key}: |df_res/dp|={abs(deriv['df_res_dparam'])/(1e6/mm):.3e} MHz/mm")

    print('\nTop levers by |dImZ/dp| (reactive tuning at f0):')
    for key, _, _, deriv in sorted(results, key=lambda x: abs(x[3]['dImZ_dparam']), reverse=True):
        print(f"- {key}: |dImZ/dp|={abs(deriv['dImZ_dparam'])*mm:.3e} ohm/mm")


def _build_jacobian_mm(results: List[Tuple[str, float, str, Dict[str, float]]]) -> Tuple[List[str], np.ndarray]:
    keys = [r[0] for r in results]
    j = np.zeros((3, len(keys)), dtype=float)

    for i, (_, _, _, deriv) in enumerate(results):
        # outputs per mm of parameter change
        j[0, i] = deriv['df_res_dparam'] / (1e6 / mm)  # MHz/mm
        j[1, i] = deriv['dReZ_dparam'] * mm            # ohm/mm
        j[2, i] = deriv['dImZ_dparam'] * mm            # ohm/mm

    return keys, j


def _solve_guided_step_mm(base: Dict[str, float], keys: List[str], jac: np.ndarray) -> np.ndarray:
    err = _error_vector(base)

    w = ERROR_WEIGHTS
    jw = w[:, None] * jac
    ew = w * err

    a = jw.T @ jw + RIDGE_DAMPING * np.eye(len(keys), dtype=float)
    b = -(jw.T @ ew)
    step_mm = np.linalg.solve(a, b)

    # Per-parameter clip
    step_mm = np.clip(step_mm, -MAX_STEP_PER_PARAM_MM, MAX_STEP_PER_PARAM_MM)

    # Global trust-region-like clip
    norm = float(np.linalg.norm(step_mm))
    if norm > MAX_TOTAL_STEP_NORM_MM and norm > 1e-12:
        step_mm = step_mm * (MAX_TOTAL_STEP_NORM_MM / norm)

    return step_mm


def _apply_step_with_validation(
    params: Dict[str, float],
    keys: List[str],
    step_mm: np.ndarray,
) -> Tuple[Dict[str, float], float]:
    for scale in (1.0, 0.5, 0.25, 0.1):
        cand = dict(params)
        for k, dmm in zip(keys, step_mm):
            cand[k] = max(1e-9, float(cand[k]) + float(dmm) * mm * scale)

        ok, _ = _validate(cand)
        if ok:
            return cand, scale

    return dict(params), 0.0


def probe_parameter_sensitivities(params: Dict[str, float], keys: List[str]) -> None:
    print('Running baseline simulation...')
    base, results = _collect_sensitivities(params, keys, verbose=True)
    _rank_and_print(results)


def guided_probe_optimize(params: Dict[str, float], keys: List[str], iters: int = GUIDED_ITERS) -> Dict[str, float]:
    p = dict(params)

    print('Initial probe:')
    base, results = _collect_sensitivities(p, keys, verbose=True)
    _rank_and_print(results)

    for it in range(1, iters + 1):
        if not results:
            print('\nStopping: no usable sensitivities.')
            break

        skeys, jac = _build_jacobian_mm(results)
        step_mm = _solve_guided_step_mm(base, skeys, jac)

        print(f'\nIteration {it}: suggested update (mm)')
        for k, d in zip(skeys, step_mm):
            print(f'- {k}: {d:+.4f} mm')

        cand, applied_scale = _apply_step_with_validation(p, skeys, step_mm)
        if applied_scale <= 0.0:
            print('No valid step found after backoff. Stopping.')
            break

        if applied_scale < 1.0:
            print(f'Applied step scale backoff: x{applied_scale:.2f}')

        p = cand
        base, results = _collect_sensitivities(p, keys, verbose=False)
        _print_metrics(f'After iter {it}', base)

    print('\nFinal parameter snapshot:')
    print({k: p[k] for k in keys})
    return p


if __name__ == '__main__':
    # 1) quick diagnostics only:
    # probe_parameter_sensitivities(dict(BASE_PARAMS), PROBE_KEYS)

    # 2) minimal guided loop:
    guided_probe_optimize(dict(BASE_PARAMS), PROBE_KEYS, iters=GUIDED_ITERS)
