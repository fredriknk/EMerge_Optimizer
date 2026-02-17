import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

mm = 1e-3

# Baseline seed (single-band, plain keys)
BASE_PARAMS: Dict[str, float] = {
    'ifa_h': 0.00773,
    'ifa_l': 0.02095,
    'ifa_w1': 0.000767,
    'ifa_w2': 0.000440,
    'ifa_wf': 0.000344,
    'ifa_fp': 0.00257,
    'ifa_e': 0.0005,
    'ifa_e2': 0.0005,
    'ifa_te': 0.0005,
    'via_size': 0.0003,
    'board_wsub': 0.030,
    'board_hsub': 0.030,
    'board_th': 0.0015,
    'mifa_meander': 0.00195,
    'mifa_low_dist': 0.00217,
    'f0': 2.45e9,
    'f1': 2.45e9-1e9,
    'f2': 2.45e9+1e9,
    'freq_points': 3,
    'mesh_boundary_size_divisor': 0.6,
    'mesh_wavelength_fraction': 0.6,
    'lambda_scale': 0.5,
}

# Stage 1 (resonance) key set
PROBE_KEYS: List[str] = [
    'ifa_l',
    'ifa_h',
    'ifa_fp',
    'ifa_w2',
    'mifa_meander',
    'mifa_low_dist',
]

# Stage 2 (impedance) key set
PHASE2_KEYS: List[str] = [
    'ifa_fp',
    'ifa_w2',
    'ifa_w1',
    'ifa_wf',
    'mifa_meander',
    'mifa_low_dist',
]

# Probe + solver settings
REL_STEP = 0.03
ABS_MIN_STEP = 0.05 * mm
SOLVER_NAME = 'CUDSS'  # fallback to PARDISO

# Stage 1: Resonance-only tuning knobs
TUNE_ITERS = 50
PHASE1_STEP_REL_LIMIT = 0.03
PHASE1_STEP_MIN_MM = 0.008
PHASE1_STEP_MAX_MM = 0.40
MAX_TOTAL_STEP_NORM_MM = 0.10
RIDGE_DAMPING = 1e-2
LINE_SEARCH_SCALES = (1.0, 0.5, 0.25, 0.1, 0.05)
IMPROVEMENT_EPS_MHZ = 0.2
TARGET_TOL_MHZ = 1.0

# Stage 2: Impedance tuning with resonance guard
PHASE2_ENABLE = True
PHASE2_ITERS = 8
PHASE2_STEP_REL_LIMIT = 0.08
PHASE2_STEP_MIN_MM = 0.003
PHASE2_STEP_MAX_MM = 0.10
PHASE2_MAX_TOTAL_STEP_NORM_MM = 0.07
PHASE2_RIDGE_DAMPING = 1e-1
PHASE2_LINE_SEARCH_SCALES = (1.0, 0.5, 0.25, 0.1, 0.05)
PHASE2_MERIT_EPS = 0.2
PHASE2_RES_GUARD_MHZ = 6.0
PHASE2_WEIGHTS = np.array([0.20, 1.00, 1.00], dtype=float)  # [df_mhz, dRe, dIm]

# State / resume (split by phase)
PHASE1_STATE_FILE = 'guided_resonance_phase1_state.json'
PHASE1_ARCHIVE_DIR = 'guided_resonance_phase1_runs'
PHASE1_RESUME = True

PHASE2_STATE_FILE = 'guided_resonance_phase2_state.json'
PHASE2_ARCHIVE_DIR = 'guided_resonance_phase2_runs'
PHASE2_RESUME = True


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _interp_complex(x: np.ndarray, y: np.ndarray, xq: float) -> complex:
    re = np.interp(xq, x, y.real)
    im = np.interp(xq, x, y.imag)
    return re + 1j * im


def _gamma_to_zin(gamma: complex, z0: float = 50.0) -> complex:
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
    rl_f0_db = float(-20.0 * math.log10(abs(gamma_f0)))
    zin_f0 = _gamma_to_zin(gamma_f0, z0=50.0)

    return {
        'f_res_hz': f_res,
        'f0_hz': f0,
        'df_mhz': (f_res - f0) / 1e6,
        'rl_f0_db': rl_f0_db,
        're_zin_f0': float(np.real(zin_f0)),
        'im_zin_f0': float(np.imag(zin_f0)),
    }


def _one_sided_df(base: Dict[str, float], shifted: Dict[str, float], delta_m: float) -> float:
    # Returns derivative in MHz/mm
    return ((shifted['f_res_hz'] - base['f_res_hz']) / delta_m) / (1e6 / mm)


def _per_param_caps_mm(
    params: Dict[str, float],
    keys: List[str],
    *,
    rel_limit: float,
    min_mm: float,
    max_mm: float,
) -> np.ndarray:
    caps = []
    for k in keys:
        val_mm = abs(float(params[k]) / mm)
        c = val_mm * rel_limit
        c = max(min_mm, min(max_mm, c))
        caps.append(c)
    return np.asarray(caps, dtype=float)


def _collect_df_sensitivities(
    params: Dict[str, float],
    keys: List[str],
    *,
    verbose: bool = True,
    base_override: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], List[Tuple[str, float, str, float]]]:
    ok, msg = _validate(params)
    if not ok:
        raise ValueError(f'Params invalid: {msg}')

    base = base_override if base_override is not None else _run_metrics(params)

    if verbose:
        print(
            f"Baseline: f_res={base['f_res_hz']/1e9:.6f} GHz, "
            f"f0={base['f0_hz']/1e9:.6f} GHz, df={base['df_mhz']:+.3f} MHz, "
            f"RL@f0={base['rl_f0_db']:.2f} dB"
        )
        print('\nResonance sensitivity probes:')

    out: List[Tuple[str, float, str, float]] = []
    for k in keys:
        if k not in params:
            if verbose:
                print(f'- {k}: skipped (not in params)')
            continue

        x0 = float(params[k])
        step = max(abs(x0) * REL_STEP, ABS_MIN_STEP)

        p_plus = dict(params)
        p_plus[k] = x0 + step

        p_minus = dict(params)
        p_minus[k] = max(1e-9, x0 - step)

        m_plus = None
        m_minus = None

        v, _ = _validate(p_plus)
        if v:
            try:
                m_plus = _run_metrics(p_plus)
            except Exception:
                m_plus = None

        v, _ = _validate(p_minus)
        if v:
            try:
                m_minus = _run_metrics(p_minus)
            except Exception:
                m_minus = None

        deriv = None
        mode = None
        if m_plus is not None and m_minus is not None:
            d_plus = _one_sided_df(base, m_plus, step)
            d_minus = _one_sided_df(base, m_minus, -step)
            deriv = 0.5 * (d_plus + d_minus)
            mode = 'central'
        elif m_plus is not None:
            deriv = _one_sided_df(base, m_plus, step)
            mode = 'forward'
        elif m_minus is not None:
            deriv = _one_sided_df(base, m_minus, -step)
            mode = 'backward'
        else:
            if verbose:
                print(f'- {k}: no valid perturbation runs')
            continue

        out.append((k, step, mode, float(deriv)))
        if verbose:
            print(f'- {k} ({mode}, step={step/mm:.4f} mm): df_res/dp={deriv:+.3e} MHz/mm')

    if verbose and out:
        print('\nTop levers by |df_res/dp|:')
        for k, _, _, d in sorted(out, key=lambda x: abs(x[3]), reverse=True):
            print(f'- {k}: |df_res/dp|={abs(d):.3e} MHz/mm')

    return base, out


def _solve_step_mm(
    params: Dict[str, float],
    base: Dict[str, float],
    sens: List[Tuple[str, float, str, float]],
) -> Tuple[List[str], np.ndarray]:
    keys = [r[0] for r in sens]
    s = np.array([r[3] for r in sens], dtype=float)  # MHz/mm

    df = float(base['df_mhz'])
    denom = float(np.dot(s, s) + RIDGE_DAMPING)
    if denom < 1e-12:
        step = np.zeros_like(s)
    else:
        step = -(df / denom) * s

    caps = _per_param_caps_mm(
        params,
        keys,
        rel_limit=PHASE1_STEP_REL_LIMIT,
        min_mm=PHASE1_STEP_MIN_MM,
        max_mm=PHASE1_STEP_MAX_MM,
    )
    step = np.clip(step, -caps, caps)

    norm = float(np.linalg.norm(step))
    if norm > MAX_TOTAL_STEP_NORM_MM and norm > 1e-12:
        step = step * (MAX_TOTAL_STEP_NORM_MM / norm)

    return keys, step


def _apply_candidate(params: Dict[str, float], keys: List[str], step_mm: np.ndarray, scale: float) -> Dict[str, float]:
    c = dict(params)
    for k, d in zip(keys, step_mm):
        c[k] = max(1e-9, float(c[k]) + float(d) * mm * scale)
    return c


def _line_search(
    params: Dict[str, float],
    base: Dict[str, float],
    keys: List[str],
    step_mm: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    base_abs = abs(float(base['df_mhz']))
    for scale in LINE_SEARCH_SCALES:
        cand = _apply_candidate(params, keys, step_mm, scale)
        ok, _ = _validate(cand)
        if not ok:
            continue
        try:
            cm = _run_metrics(cand)
        except Exception:
            continue

        cand_abs = abs(float(cm['df_mhz']))
        if cand_abs < (base_abs - IMPROVEMENT_EPS_MHZ):
            return cand, cm, scale

    return dict(params), base, 0.0


def _collect_full_sensitivities(
    params: Dict[str, float],
    keys: List[str],
    *,
    verbose: bool = True,
    base_override: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], List[Tuple[str, float, str, np.ndarray]]]:
    ok, msg = _validate(params)
    if not ok:
        raise ValueError(f'Params invalid: {msg}')

    base = base_override if base_override is not None else _run_metrics(params)
    if verbose:
        print(
            f"Baseline(phase2): f_res={base['f_res_hz']/1e9:.6f} GHz, "
            f"df={base['df_mhz']:+.3f} MHz, RL@f0={base['rl_f0_db']:.2f} dB, "
            f"Zin@f0={base['re_zin_f0']:.2f} + j{base['im_zin_f0']:.2f} ohm"
        )
        print('\nImpedance sensitivity probes:')

    out: List[Tuple[str, float, str, np.ndarray]] = []
    for k in keys:
        if k not in params:
            if verbose:
                print(f'- {k}: skipped (not in params)')
            continue

        x0 = float(params[k])
        step = max(abs(x0) * REL_STEP, ABS_MIN_STEP)
        p_plus = dict(params); p_plus[k] = x0 + step
        p_minus = dict(params); p_minus[k] = max(1e-9, x0 - step)

        m_plus = None
        m_minus = None

        v, _ = _validate(p_plus)
        if v:
            try:
                m_plus = _run_metrics(p_plus)
            except Exception:
                m_plus = None

        v, _ = _validate(p_minus)
        if v:
            try:
                m_minus = _run_metrics(p_minus)
            except Exception:
                m_minus = None

        deriv = None
        mode = None
        if m_plus is not None and m_minus is not None:
            dplus = np.array(
                [
                    _one_sided_df(base, m_plus, step),
                    (m_plus['re_zin_f0'] - base['re_zin_f0']) / (step / mm),
                    (m_plus['im_zin_f0'] - base['im_zin_f0']) / (step / mm),
                ],
                dtype=float,
            )
            dminus = np.array(
                [
                    _one_sided_df(base, m_minus, -step),
                    (m_minus['re_zin_f0'] - base['re_zin_f0']) / ((-step) / mm),
                    (m_minus['im_zin_f0'] - base['im_zin_f0']) / ((-step) / mm),
                ],
                dtype=float,
            )
            deriv = 0.5 * (dplus + dminus)
            mode = 'central'
        elif m_plus is not None:
            deriv = np.array(
                [
                    _one_sided_df(base, m_plus, step),
                    (m_plus['re_zin_f0'] - base['re_zin_f0']) / (step / mm),
                    (m_plus['im_zin_f0'] - base['im_zin_f0']) / (step / mm),
                ],
                dtype=float,
            )
            mode = 'forward'
        elif m_minus is not None:
            deriv = np.array(
                [
                    _one_sided_df(base, m_minus, -step),
                    (m_minus['re_zin_f0'] - base['re_zin_f0']) / ((-step) / mm),
                    (m_minus['im_zin_f0'] - base['im_zin_f0']) / ((-step) / mm),
                ],
                dtype=float,
            )
            mode = 'backward'
        else:
            if verbose:
                print(f'- {k}: no valid perturbation runs')
            continue

        out.append((k, step, mode, deriv))
        if verbose:
            print(
                f"- {k} ({mode}, step={step/mm:.4f} mm): "
                f"df/dp={deriv[0]:+.3e} MHz/mm, "
                f"dRe/dp={deriv[1]:+.3e} ohm/mm, "
                f"dIm/dp={deriv[2]:+.3e} ohm/mm"
            )

    return base, out


def _phase2_error(metrics: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            float(metrics['df_mhz']),
            float(metrics['re_zin_f0']) - 50.0,
            float(metrics['im_zin_f0']),
        ],
        dtype=float,
    )


def _phase2_merit(metrics: Dict[str, float]) -> float:
    e = PHASE2_WEIGHTS * _phase2_error(metrics)
    return float(np.linalg.norm(e))


def _solve_phase2_step_mm(
    params: Dict[str, float],
    base: Dict[str, float],
    sens: List[Tuple[str, float, str, np.ndarray]],
) -> Tuple[List[str], np.ndarray]:
    keys = [r[0] for r in sens]
    j = np.vstack([r[3] for r in sens]).T  # 3 x n
    e = _phase2_error(base)                # 3
    jw = PHASE2_WEIGHTS[:, None] * j
    ew = PHASE2_WEIGHTS * e
    a = jw.T @ jw + PHASE2_RIDGE_DAMPING * np.eye(j.shape[1], dtype=float)
    b = -(jw.T @ ew)
    step = np.linalg.solve(a, b)

    caps = _per_param_caps_mm(
        params,
        keys,
        rel_limit=PHASE2_STEP_REL_LIMIT,
        min_mm=PHASE2_STEP_MIN_MM,
        max_mm=PHASE2_STEP_MAX_MM,
    )
    step = np.clip(step, -caps, caps)
    norm = float(np.linalg.norm(step))
    if norm > PHASE2_MAX_TOTAL_STEP_NORM_MM and norm > 1e-12:
        step = step * (PHASE2_MAX_TOTAL_STEP_NORM_MM / norm)
    return keys, step


def _line_search_phase2(
    params: Dict[str, float],
    base: Dict[str, float],
    keys: List[str],
    step_mm: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    base_merit = _phase2_merit(base)
    for scale in PHASE2_LINE_SEARCH_SCALES:
        cand = _apply_candidate(params, keys, step_mm, scale)
        ok, _ = _validate(cand)
        if not ok:
            continue
        try:
            cm = _run_metrics(cand)
        except Exception:
            continue

        if abs(float(cm['df_mhz'])) > PHASE2_RES_GUARD_MHZ:
            continue

        m = _phase2_merit(cm)
        if m < (base_merit - PHASE2_MERIT_EPS):
            return cand, cm, scale
    return dict(params), base, 0.0


def tune_impedance_with_res_guard(
    params: Dict[str, float],
    probe_keys: List[str],
    *,
    iters: int = PHASE2_ITERS,
    state_file: str = PHASE2_STATE_FILE,
    archive_dir: str = PHASE2_ARCHIVE_DIR,
    resume_state: Optional[Dict] = None,
) -> Dict[str, float]:
    p = dict(params)
    history: List[Dict] = []
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    start_iter = 0
    archived: List[str] = []
    print('\n=== Phase 2: Impedance Tuning (Resonance Guarded) ===')

    if resume_state is not None:
        run_id = str(resume_state.get('run_id', run_id))
        start_iter = int(resume_state.get('iteration', 0))
        rp = resume_state.get('params')
        rh = resume_state.get('history')
        if isinstance(rp, dict):
            p = {str(k): float(v) for k, v in rp.items()}
        if isinstance(rh, list):
            history = rh
        print(f'Resuming phase2 run_id={run_id} at iteration={start_iter}')

    base, sens = _collect_full_sensitivities(p, probe_keys, verbose=True)

    if resume_state is None:
        history.append({
            'stage': 'phase2',
            'iteration': 0,
            'df_mhz': float(base['df_mhz']),
            're_zin_f0': float(base['re_zin_f0']),
            'im_zin_f0': float(base['im_zin_f0']),
            'phase2_merit': float(_phase2_merit(base)),
            'params': {k: float(p[k]) for k in probe_keys if k in p},
        })

    _save_state(
        state_file,
        run_id=run_id,
        iteration=start_iter,
        params=p,
        metrics=base,
        probe_keys=probe_keys,
        history=history,
        note='resume_probe' if resume_state is not None else 'initial_probe',
    )
    archived.append(
        _archive_state(
            archive_dir,
            run_id=run_id,
            iteration=start_iter,
            params=p,
            metrics=base,
            probe_keys=probe_keys,
            history=history,
            note='resume_probe' if resume_state is not None else 'initial_probe',
        )
    )

    if start_iter >= iters:
        print(f'Nothing to do phase2: checkpoint iteration={start_iter} >= target iters={iters}')
        return p

    for it in range(start_iter + 1, iters + 1):
        if not sens:
            print('Stopping phase2: no usable sensitivities.')
            _save_state(
                state_file,
                run_id=run_id,
                iteration=it - 1,
                params=p,
                metrics=base,
                probe_keys=probe_keys,
                history=history,
                note='stop_no_sensitivities',
            )
            archived.append(
                _archive_state(
                    archive_dir,
                    run_id=run_id,
                    iteration=it - 1,
                    params=p,
                    metrics=base,
                    probe_keys=probe_keys,
                    history=history,
                    note='stop_no_sensitivities',
                )
            )
            break

        keys, step_mm = _solve_phase2_step_mm(p, base, sens)
        print(f'\nPhase2 iter {it}: suggested update (mm)')
        for k, d in zip(keys, step_mm):
            print(f'- {k}: {d:+.4f} mm')

        cand, cm, scale = _line_search_phase2(p, base, keys, step_mm)
        if scale <= 0.0:
            print('Stopping phase2: no improving guarded step found.')
            _save_state(
                state_file,
                run_id=run_id,
                iteration=it - 1,
                params=p,
                metrics=base,
                probe_keys=probe_keys,
                history=history,
                note='stop_no_improving_step',
            )
            archived.append(
                _archive_state(
                    archive_dir,
                    run_id=run_id,
                    iteration=it - 1,
                    params=p,
                    metrics=base,
                    probe_keys=probe_keys,
                    history=history,
                    note='stop_no_improving_step',
                )
            )
            break

        if scale < 1.0:
            print(f'Applied phase2 line-search scale: x{scale:.2f}')
        else:
            print('Applied full phase2 step.')

        p = cand
        base = cm
        print(
            f"After phase2 iter {it}: f_res={base['f_res_hz']/1e9:.6f} GHz, "
            f"df={base['df_mhz']:+.3f} MHz, RL@f0={base['rl_f0_db']:.2f} dB, "
            f"Zin@f0={base['re_zin_f0']:.2f} + j{base['im_zin_f0']:.2f} ohm, "
            f"phase2_merit={_phase2_merit(base):.3f}"
        )

        history.append({
            'stage': 'phase2',
            'iteration': int(it),
            'df_mhz': float(base['df_mhz']),
            're_zin_f0': float(base['re_zin_f0']),
            'im_zin_f0': float(base['im_zin_f0']),
            'phase2_merit': float(_phase2_merit(base)),
            'applied_scale': float(scale),
            'active_keys': list(keys),
            'params': {k: float(p[k]) for k in probe_keys if k in p},
        })
        _save_state(
            state_file,
            run_id=run_id,
            iteration=it,
            params=p,
            metrics=base,
            probe_keys=probe_keys,
            history=history,
            note='iter_accepted',
        )
        archived.append(
            _archive_state(
                archive_dir,
                run_id=run_id,
                iteration=it,
                params=p,
                metrics=base,
                probe_keys=probe_keys,
                history=history,
                note='iter_accepted',
            )
        )

        base, sens = _collect_full_sensitivities(p, probe_keys, verbose=False, base_override=base)

    print(f'Phase2 state saved to: {state_file}')
    if archived:
        print(f'Phase2 archived snapshots: {len(archived)} files in {archive_dir}')
        print(f'Phase2 latest archive: {archived[-1]}')
    return p


def _save_state(
    path: str,
    *,
    run_id: str,
    iteration: int,
    params: Dict[str, float],
    metrics: Dict[str, float],
    probe_keys: List[str],
    history: List[Dict],
    note: str,
) -> None:
    state = {
        'version': 2,
        'run_id': str(run_id),
        'saved_utc': _now_utc_iso(),
        'iteration': int(iteration),
        'note': str(note),
        'params': {k: float(v) for k, v in params.items()},
        'metrics': {k: float(v) for k, v in metrics.items()},
        'probe_keys': list(probe_keys),
        'history': history,
        'config': {
            'REL_STEP': float(REL_STEP),
            'ABS_MIN_STEP_mm': float(ABS_MIN_STEP / mm),
            'PHASE1_STEP_REL_LIMIT': float(PHASE1_STEP_REL_LIMIT),
            'PHASE1_STEP_MIN_MM': float(PHASE1_STEP_MIN_MM),
            'PHASE1_STEP_MAX_MM': float(PHASE1_STEP_MAX_MM),
            'MAX_TOTAL_STEP_NORM_MM': float(MAX_TOTAL_STEP_NORM_MM),
            'RIDGE_DAMPING': float(RIDGE_DAMPING),
            'LINE_SEARCH_SCALES': [float(x) for x in LINE_SEARCH_SCALES],
            'IMPROVEMENT_EPS_MHZ': float(IMPROVEMENT_EPS_MHZ),
            'TARGET_TOL_MHZ': float(TARGET_TOL_MHZ),
            'PHASE2_ENABLE': bool(PHASE2_ENABLE),
            'PHASE2_ITERS': int(PHASE2_ITERS),
            'PHASE2_STEP_REL_LIMIT': float(PHASE2_STEP_REL_LIMIT),
            'PHASE2_STEP_MIN_MM': float(PHASE2_STEP_MIN_MM),
            'PHASE2_STEP_MAX_MM': float(PHASE2_STEP_MAX_MM),
            'PHASE2_WEIGHTS': [float(x) for x in PHASE2_WEIGHTS],
            'PHASE2_RES_GUARD_MHZ': float(PHASE2_RES_GUARD_MHZ),
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
    probe_keys: List[str],
    history: List[Dict],
    note: str,
) -> str:
    d = Path(archive_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = str(d / f'{run_id}_iter{int(iteration):03d}_{note}.json')
    _save_state(
        out,
        run_id=run_id,
        iteration=iteration,
        params=params,
        metrics=metrics,
        probe_keys=probe_keys,
        history=history,
        note=note,
    )
    return out


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


def tune_resonance(
    params: Dict[str, float],
    probe_keys: List[str],
    *,
    iters: int = TUNE_ITERS,
    state_file: str = PHASE1_STATE_FILE,
    archive_dir: str = PHASE1_ARCHIVE_DIR,
    resume_state: Optional[Dict] = None,
) -> Dict[str, float]:
    p = dict(params)
    history: List[Dict] = []
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    start_iter = 0
    archived: List[str] = []

    if resume_state is not None:
        run_id = str(resume_state.get('run_id', run_id))
        start_iter = int(resume_state.get('iteration', 0))
        rp = resume_state.get('params')
        rh = resume_state.get('history')
        if isinstance(rp, dict):
            p = {str(k): float(v) for k, v in rp.items()}
        if isinstance(rh, list):
            history = rh
        print(f'Resuming run_id={run_id} at iteration={start_iter}')
    else:
        print('Initial resonance probe:')

    base, sens = _collect_df_sensitivities(p, probe_keys, verbose=True)

    if resume_state is None:
        history.append({
            'stage': 'phase1',
            'iteration': 0,
            'df_mhz': float(base['df_mhz']),
            'rl_f0_db': float(base['rl_f0_db']),
            'params': {k: float(p[k]) for k in probe_keys if k in p},
        })

    _save_state(
        state_file,
        run_id=run_id,
        iteration=start_iter,
        params=p,
        metrics=base,
        probe_keys=probe_keys,
        history=history,
        note='resume_probe' if resume_state is not None else 'initial_probe',
    )
    archived.append(
        _archive_state(
            archive_dir,
            run_id=run_id,
            iteration=start_iter,
            params=p,
            metrics=base,
            probe_keys=probe_keys,
            history=history,
            note='resume_probe' if resume_state is not None else 'initial_probe',
        )
    )

    if start_iter >= iters:
        print(f'Nothing to do: checkpoint iteration={start_iter} >= target iters={iters}')
    else:
        for it in range(start_iter + 1, iters + 1):
            if not sens:
                print('\nStopping: no usable resonance sensitivities.')
                break

            keys, step_mm = _solve_step_mm(p, base, sens)
            print(f'\nIteration {it}: suggested resonance update (mm)')
            for k, d in zip(keys, step_mm):
                print(f'- {k}: {d:+.4f} mm')

            cand, cm, scale = _line_search(p, base, keys, step_mm)
            if scale <= 0.0:
                print('No improving resonance step found. Stopping.')
                _save_state(
                    state_file,
                    run_id=run_id,
                    iteration=it - 1,
                    params=p,
                    metrics=base,
                    probe_keys=probe_keys,
                    history=history,
                    note='stop_no_improving_step',
                )
                archived.append(
                    _archive_state(
                        archive_dir,
                        run_id=run_id,
                        iteration=it - 1,
                        params=p,
                        metrics=base,
                        probe_keys=probe_keys,
                        history=history,
                        note='stop_no_improving_step',
                    )
                )
                break

            if scale < 1.0:
                print(f'Applied line-search scale: x{scale:.2f}')
            else:
                print('Applied full step.')

            p = cand
            base = cm
            print(
                f"After iter {it}: f_res={base['f_res_hz']/1e9:.6f} GHz, "
                f"df={base['df_mhz']:+.3f} MHz, RL@f0={base['rl_f0_db']:.2f} dB"
            )

            history.append({
                'stage': 'phase1',
                'iteration': int(it),
                'df_mhz': float(base['df_mhz']),
                'rl_f0_db': float(base['rl_f0_db']),
                'applied_scale': float(scale),
                'active_keys': list(keys),
                'params': {k: float(p[k]) for k in probe_keys if k in p},
            })

            _save_state(
                state_file,
                run_id=run_id,
                iteration=it,
                params=p,
                metrics=base,
                probe_keys=probe_keys,
                history=history,
                note='iter_accepted',
            )
            archived.append(
                _archive_state(
                    archive_dir,
                    run_id=run_id,
                    iteration=it,
                    params=p,
                    metrics=base,
                    probe_keys=probe_keys,
                    history=history,
                    note='iter_accepted',
                )
            )

            if abs(float(base['df_mhz'])) <= TARGET_TOL_MHZ:
                print(f'Resonance target met: |df| <= {TARGET_TOL_MHZ:.2f} MHz')
                break

            base, sens = _collect_df_sensitivities(p, probe_keys, verbose=False, base_override=base)

    print('\nFinal resonance-tuned snapshot:')
    print({k: p[k] for k in probe_keys})
    print(f'Phase1 state saved to: {state_file}')
    if archived:
        print(f'Phase1 archived snapshots: {len(archived)} files in {archive_dir}')
        print(f'Phase1 latest archive: {archived[-1]}')

    if PHASE2_ENABLE:
        phase2_resume_state = None
        if PHASE2_RESUME:
            phase2_resume_state = _load_state(PHASE2_STATE_FILE)
            if phase2_resume_state is not None:
                print(f'Resuming phase2 from state file: {PHASE2_STATE_FILE}')

        p = tune_impedance_with_res_guard(
            p,
            PHASE2_KEYS,
            iters=PHASE2_ITERS,
            state_file=PHASE2_STATE_FILE,
            archive_dir=PHASE2_ARCHIVE_DIR,
            resume_state=phase2_resume_state if PHASE2_RESUME else None,
        )

    print('\nFinal two-stage snapshot:')
    print({k: p[k] for k in probe_keys})
    print({k: p[k] for k in PHASE2_KEYS if k in p})
    return p


if __name__ == '__main__':
    start_params = dict(BASE_PARAMS)
    loaded_state = None

    if PHASE1_RESUME:
        loaded_state = _load_state(PHASE1_STATE_FILE)
        if loaded_state is not None:
            lp = loaded_state.get('params')
            if isinstance(lp, dict):
                start_params.update({str(k): float(v) for k, v in lp.items()})
            print(f'Resuming phase1 from state file: {PHASE1_STATE_FILE}')

    tune_resonance(
        start_params,
        PROBE_KEYS,
        iters=TUNE_ITERS,
        state_file=PHASE1_STATE_FILE,
        archive_dir=PHASE1_ARCHIVE_DIR,
        resume_state=loaded_state if PHASE1_RESUME else None,
    )
