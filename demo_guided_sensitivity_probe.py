import math
import json
from datetime import datetime, timezone
from pathlib import Path
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
    'f0': 2.45e+09, 
    'f1': 2.3e+09, 
    'f2': 2.6e+09, 
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
]

REL_STEP = 0.03      # +/- 3%
ABS_MIN_STEP = 0.05 * mm
SOLVER_NAME = 'CUDSS'  # fallback to PARDISO if unavailable

# Guided-update knobs
GUIDED_ITERS = 10
MAX_STEP_PER_PARAM_MM = 0.5
MAX_TOTAL_STEP_NORM_MM = 1.0
RIDGE_DAMPING = 1e-1
LINE_SEARCH_SCALES = (1.0, 0.5, 0.25, 0.1, 0.05)
MERIT_IMPROVEMENT_EPS = 1e-6
MERIT_MIN_IMPROVEMENT_ABS = 1.0
ENFORCE_BEST_SO_FAR = True
OUTPUT_STATE_FILE = 'guided_probe_state.json'
RESUME_FROM_OUTPUT = True
OUTPUT_ARCHIVE_DIR = 'guided_probe_runs'
SMART_LEARNING = True
SMART_TOP_K = 5
SMART_MIN_HISTORY = 4
SMART_HISTORY_WINDOW = 20
SMART_RIDGE = 0.5
SMART_BLEND = 0.55

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
    merit = _merit(m)
    print(
        f"{label}: f_res={m['f_res_hz']/1e9:.6f} GHz, "
        f"RL@f0={m['rl_f0_db']:.2f} dB, "
        f"Zin@f0={m['re_zin_f0']:.2f} + j{m['im_zin_f0']:.2f} ohm, "
        f"err=[df={err[0]:+.2f} MHz, dRe={err[1]:+.2f} ohm, dIm={err[2]:+.2f} ohm], "
        f"phi={merit:.4f}"
    )


def _collect_sensitivities(
    params: Dict[str, float],
    keys: List[str],
    *,
    verbose: bool = True,
    base_override: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], List[Tuple[str, float, str, Dict[str, float]]]]:
    ok, msg = _validate(params)
    if not ok:
        raise ValueError(f'Baseline params invalid: {msg}')

    base = base_override if base_override is not None else _run_metrics(params)
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


def _merit(metrics: Dict[str, float]) -> float:
    # Weighted L2 norm of [df_MHz, dRe_ohm, dIm_ohm]
    ew = ERROR_WEIGHTS * _error_vector(metrics)
    return float(np.linalg.norm(ew))


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


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _save_state(
    path: str,
    *,
    run_id: str,
    iteration: int,
    params: Dict[str, float],
    metrics: Dict[str, float],
    keys: List[str],
    history: List[Dict[str, float]],
    note: str,
    best_phi: float,
    best_params: Dict[str, float],
) -> None:
    state = {
        'version': 1,
        'run_id': str(run_id),
        'saved_utc': _now_utc_iso(),
        'iteration': int(iteration),
        'note': str(note),
        'params': {k: float(v) for k, v in params.items()},
        'metrics': {k: float(v) for k, v in metrics.items()},
        'error_vector': {
            'df_mhz': float(_error_vector(metrics)[0]),
            'dRe_ohm': float(_error_vector(metrics)[1]),
            'dIm_ohm': float(_error_vector(metrics)[2]),
        },
        'merit_phi': float(_merit(metrics)),
        'best_phi': float(best_phi),
        'best_params': {k: float(v) for k, v in best_params.items()},
        'probe_keys': list(keys),
        'history': history,
        'config': {
            'REL_STEP': float(REL_STEP),
            'ABS_MIN_STEP_mm': float(ABS_MIN_STEP / mm),
            'MAX_STEP_PER_PARAM_MM': float(MAX_STEP_PER_PARAM_MM),
            'MAX_TOTAL_STEP_NORM_MM': float(MAX_TOTAL_STEP_NORM_MM),
            'RIDGE_DAMPING': float(RIDGE_DAMPING),
            'LINE_SEARCH_SCALES': [float(x) for x in LINE_SEARCH_SCALES],
            'ERROR_WEIGHTS': [float(x) for x in ERROR_WEIGHTS],
            'SOLVER_NAME': SOLVER_NAME,
        },
    }
    Path(path).write_text(json.dumps(state, indent=2), encoding='utf-8')


def _archive_state(
    archive_dir: str,
    *,
    run_id: str,
    iteration: int,
    params: Dict[str, float],
    metrics: Dict[str, float],
    keys: List[str],
    history: List[Dict[str, float]],
    note: str,
    best_phi: float,
    best_params: Dict[str, float],
) -> str:
    d = Path(archive_dir)
    d.mkdir(parents=True, exist_ok=True)
    fname = f'{run_id}_iter{int(iteration):03d}_{note}.json'
    out_path = str(d / fname)
    _save_state(
        out_path,
        run_id=run_id,
        iteration=iteration,
        params=params,
        metrics=metrics,
        keys=keys,
        history=history,
        note=note,
        best_phi=best_phi,
        best_params=best_params,
    )
    return out_path


def _load_state(path: str) -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def _learning_scores_from_history(
    history: List[Dict[str, float]],
    keys: List[str],
    current_err: np.ndarray,
) -> Optional[np.ndarray]:
    # Learn linear map de ≈ dp * B from accepted step history, where:
    # dp: parameter deltas [mm], de: error deltas [MHz, ohm, ohm].
    if len(history) < SMART_MIN_HISTORY:
        return None

    recent = history[-SMART_HISTORY_WINDOW:]
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    for i in range(1, len(recent)):
        h0 = recent[i - 1]
        h1 = recent[i]
        p0 = h0.get('params')
        p1 = h1.get('params')
        if not isinstance(p0, dict) or not isinstance(p1, dict):
            continue
        try:
            v0 = np.array([float(p0[k]) / mm for k in keys], dtype=float)
            v1 = np.array([float(p1[k]) / mm for k in keys], dtype=float)
            e0 = np.array([float(h0['df_mhz']), float(h0['dRe_ohm']), float(h0['dIm_ohm'])], dtype=float)
            e1 = np.array([float(h1['df_mhz']), float(h1['dRe_ohm']), float(h1['dIm_ohm'])], dtype=float)
        except Exception:
            continue
        x_rows.append(v1 - v0)
        y_rows.append(e1 - e0)

    if len(x_rows) < 2:
        return None

    x = np.vstack(x_rows)  # (n_samples, n_keys)
    y = np.vstack(y_rows)  # (n_samples, 3)
    xtx = x.T @ x + SMART_RIDGE * np.eye(x.shape[1], dtype=float)
    b = np.linalg.solve(xtx, x.T @ y)  # (n_keys, 3)

    tw = ERROR_WEIGHTS * (-current_err)
    tnorm = float(np.linalg.norm(tw)) + 1e-12
    scores = np.zeros(len(keys), dtype=float)
    for i in range(len(keys)):
        bw = ERROR_WEIGHTS * b[i, :]
        proj = float(np.dot(bw, tw))
        scores[i] = max(0.0, proj / tnorm)
    return scores


def _select_active_results(
    results: List[Tuple[str, float, str, Dict[str, float]]],
    history: List[Dict[str, float]],
    base: Dict[str, float],
) -> List[Tuple[str, float, str, Dict[str, float]]]:
    if not SMART_LEARNING or len(results) <= SMART_TOP_K:
        return results

    keys_all, jac = _build_jacobian_mm(results)
    local_scores = np.linalg.norm(ERROR_WEIGHTS[:, None] * jac, axis=0)
    learned_scores = _learning_scores_from_history(history, keys_all, _error_vector(base))

    if learned_scores is None:
        final_scores = _normalize01(local_scores)
    else:
        final_scores = (1.0 - SMART_BLEND) * _normalize01(local_scores) + SMART_BLEND * _normalize01(learned_scores)

    top_k = max(3, min(SMART_TOP_K, len(results)))
    idx = np.argsort(-final_scores)[:top_k]
    selected = [results[int(i)] for i in idx]

    print('Active keys this iter: ' + ', '.join([r[0] for r in selected]))
    return selected


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


def _line_search_candidate(
    params: Dict[str, float],
    keys: List[str],
    step_mm: np.ndarray,
    current_metrics: Dict[str, float],
    *,
    best_phi: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    current_phi = _merit(current_metrics)
    target_phi = current_phi
    if ENFORCE_BEST_SO_FAR and best_phi is not None:
        target_phi = min(target_phi, float(best_phi))

    for scale in LINE_SEARCH_SCALES:
        cand = dict(params)
        for k, dmm in zip(keys, step_mm):
            cand[k] = max(1e-9, float(cand[k]) + float(dmm) * mm * scale)

        ok, _ = _validate(cand)
        if not ok:
            continue

        try:
            cand_metrics = _run_metrics(cand)
        except Exception:
            continue

        cand_phi = _merit(cand_metrics)
        required_gain = max(MERIT_IMPROVEMENT_EPS, MERIT_MIN_IMPROVEMENT_ABS)
        if cand_phi < (target_phi - required_gain):
            return cand, cand_metrics, scale

    return dict(params), current_metrics, 0.0


def probe_parameter_sensitivities(params: Dict[str, float], keys: List[str]) -> None:
    print('Running baseline simulation...')
    base, results = _collect_sensitivities(params, keys, verbose=True)
    _rank_and_print(results)


def guided_probe_optimize(
    params: Dict[str, float],
    keys: List[str],
    iters: int = GUIDED_ITERS,
    *,
    output_path: str = OUTPUT_STATE_FILE,
    archive_dir: str = OUTPUT_ARCHIVE_DIR,
    resume_state: Optional[Dict] = None,
) -> Dict[str, float]:
    p = dict(params)
    history: List[Dict[str, float]] = []
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    start_iter = 0
    archived_paths: List[str] = []

    if resume_state is not None:
        run_id = str(resume_state.get('run_id', run_id))
        start_iter = int(resume_state.get('iteration', 0))
        st_params = resume_state.get('params')
        if isinstance(st_params, dict):
            p = {str(k): float(v) for k, v in st_params.items()}
        st_history = resume_state.get('history')
        if isinstance(st_history, list):
            history = st_history
        print(f'Resuming run_id={run_id} at iteration={start_iter}')
    else:
        print('Initial probe:')

    base, results = _collect_sensitivities(p, keys, verbose=True)
    _rank_and_print(results)
    best_phi = float(_merit(base))
    best_params = dict(p)
    if resume_state is not None:
        st_best_phi = resume_state.get('best_phi')
        st_best_params = resume_state.get('best_params')
        if isinstance(st_best_phi, (int, float)):
            best_phi = float(st_best_phi)
        if isinstance(st_best_params, dict):
            best_params = {str(k): float(v) for k, v in st_best_params.items()}

    if resume_state is None:
        history.append({
            'iteration': 0,
            'phi': best_phi,
            'df_mhz': float(_error_vector(base)[0]),
            'dRe_ohm': float(_error_vector(base)[1]),
            'dIm_ohm': float(_error_vector(base)[2]),
            'best_phi': best_phi,
            'params': {k: float(p[k]) for k in keys if k in p},
        })
    _save_state(
        output_path,
        run_id=run_id,
        iteration=start_iter,
        params=p,
        metrics=base,
        keys=keys,
        history=history,
        note='resume_probe' if resume_state is not None else 'initial_probe',
        best_phi=best_phi,
        best_params=best_params,
    )
    archived_paths.append(
        _archive_state(
            archive_dir,
            run_id=run_id,
            iteration=start_iter,
            params=p,
            metrics=base,
            keys=keys,
            history=history,
            note='resume_probe' if resume_state is not None else 'initial_probe',
            best_phi=best_phi,
            best_params=best_params,
        )
    )

    if start_iter >= iters:
        print(f'Nothing to do: checkpoint iteration={start_iter} already >= target iters={iters}')
        print('\nFinal parameter snapshot:')
        print({k: p[k] for k in keys})
        print(f'Best phi observed: {best_phi:.4f}')
        print('Best parameter snapshot:')
        print({k: best_params[k] for k in keys})
        print(f'State saved to: {output_path}')
        return p

    for it in range(start_iter + 1, iters + 1):
        if not results:
            print('\nStopping: no usable sensitivities.')
            _save_state(
                output_path,
                run_id=run_id,
                iteration=it - 1,
                params=p,
                metrics=base,
                keys=keys,
                history=history,
                note='stop_no_sensitivities',
                best_phi=best_phi,
                best_params=best_params,
            )
            archived_paths.append(
                _archive_state(
                    archive_dir,
                    run_id=run_id,
                    iteration=it - 1,
                    params=p,
                    metrics=base,
                    keys=keys,
                    history=history,
                    note='stop_no_sensitivities',
                    best_phi=best_phi,
                    best_params=best_params,
                )
            )
            break

        active_results = _select_active_results(results, history, base)
        skeys, jac = _build_jacobian_mm(active_results)
        step_mm = _solve_guided_step_mm(base, skeys, jac)

        print(f'\nIteration {it}: suggested update (mm)')
        for k, d in zip(skeys, step_mm):
            print(f'- {k}: {d:+.4f} mm')

        cand, cand_metrics, applied_scale = _line_search_candidate(
            p,
            skeys,
            step_mm,
            base,
            best_phi=best_phi,
        )
        if applied_scale <= 0.0:
            print('No improving step found after line search. Stopping.')
            _save_state(
                output_path,
                run_id=run_id,
                iteration=it - 1,
                params=p,
                metrics=base,
                keys=keys,
                history=history,
                note='stop_no_improving_step',
                best_phi=best_phi,
                best_params=best_params,
            )
            archived_paths.append(
                _archive_state(
                    archive_dir,
                    run_id=run_id,
                    iteration=it - 1,
                    params=p,
                    metrics=base,
                    keys=keys,
                    history=history,
                    note='stop_no_improving_step',
                    best_phi=best_phi,
                    best_params=best_params,
                )
            )
            break

        if applied_scale < 1.0:
            print(f'Applied line-search scale: x{applied_scale:.2f}')
        else:
            print('Applied full step.')

        p = cand
        base = cand_metrics
        cand_phi = float(_merit(base))
        if cand_phi < best_phi:
            best_phi = cand_phi
            best_params = dict(p)

        base, results = _collect_sensitivities(p, keys, verbose=False, base_override=base)
        _print_metrics(f'After iter {it}', base)
        history.append({
            'iteration': int(it),
            'phi': cand_phi,
            'df_mhz': float(_error_vector(base)[0]),
            'dRe_ohm': float(_error_vector(base)[1]),
            'dIm_ohm': float(_error_vector(base)[2]),
            'applied_scale': float(applied_scale),
            'best_phi': float(best_phi),
            'params': {k: float(p[k]) for k in keys if k in p},
            'active_keys': list(skeys),
        })
        _save_state(
            output_path,
            run_id=run_id,
            iteration=it,
            params=p,
            metrics=base,
            keys=keys,
            history=history,
            note='iter_accepted',
            best_phi=best_phi,
            best_params=best_params,
        )
        archived_paths.append(
            _archive_state(
                archive_dir,
                run_id=run_id,
                iteration=it,
                params=p,
                metrics=base,
                keys=keys,
                history=history,
                note='iter_accepted',
                best_phi=best_phi,
                best_params=best_params,
            )
        )

    print('\nFinal parameter snapshot:')
    print({k: p[k] for k in keys})
    print(f'Best phi observed: {best_phi:.4f}')
    print('Best parameter snapshot:')
    print({k: best_params[k] for k in keys})
    print(f'State saved to: {output_path}')
    if archived_paths:
        print(f'Archived snapshots: {len(archived_paths)} files in {archive_dir}')
        print(f'Latest archive: {archived_paths[-1]}')
    return p


if __name__ == '__main__':
    # 1) quick diagnostics only:
    # probe_parameter_sensitivities(dict(BASE_PARAMS), PROBE_KEYS)

    # 2) minimal guided loop:
    start_params = dict(BASE_PARAMS)
    loaded_state = None
    if RESUME_FROM_OUTPUT:
        loaded_state = _load_state(OUTPUT_STATE_FILE)
        if loaded_state is not None:
            loaded_params = loaded_state.get('params')
            if isinstance(loaded_params, dict):
                start_params.update({str(k): float(v) for k, v in loaded_params.items()})
            print(f'Resuming from state file: {OUTPUT_STATE_FILE}')

    guided_probe_optimize(
        start_params,
        PROBE_KEYS,
        iters=GUIDED_ITERS,
        output_path=OUTPUT_STATE_FILE,
        archive_dir=OUTPUT_ARCHIVE_DIR,
        resume_state=loaded_state if RESUME_FROM_OUTPUT else None,
    )
