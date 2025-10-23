import copy
import time
import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.optimize import differential_evolution

import emerge as em
from ifalib import build_mifa, get_s11_at_freq, get_loss
from emerge.plot import plot_sp

import os, multiprocessing as mp
import numpy as np

mm = 1e-3  # meters per millimeter

# ---------- Minimal elapsed-time logger (solver-style) ----------
class OptLogger:
    def __init__(self, enabled: bool = True):
        self.t0 = time.perf_counter()
        self.enabled = enabled

    def _stamp(self) -> str:
        dt = time.perf_counter() - self.t0
        # Format ~ "0:00:07.833792"
        h = int(dt // 3600)
        m = int((dt % 3600) // 60)
        s = dt % 60
        return f"{h}:{m:02d}:{s:06.3f}"

    def _print(self, level: str, msg: str):
        if not self.enabled:
            return
        # match the feel: "0:00:07.833792  INFO   : message"
        print(f"{self._stamp():>12}  {level:<5}  : {msg}")

    def info(self, msg: str): self._print("INFO", msg)
    def warn(self, msg: str): self._print("WARN", msg)
    def error(self, msg: str): self._print("ERROR", msg)


# ---------- Utilities ----------

def _precheck_params(p: dict):
    # Ensure positive dimensions and sane relationships (avoid negative plate sizes)
    must_be_pos = ["ifa_h","ifa_l","ifa_w1","ifa_w2","ifa_wf","ifa_fp","ifa_e","ifa_e2","ifa_te",
                   "via_size","board_wsub","board_hsub","board_th","mifa_meander",
                   "mifa_meander_edge_distance","mifa_tipdistance"]
    for k in must_be_pos:
        if k in p and not (float(p[k]) > 0):
            raise ValueError(f"Parameter {k} must be > 0 (got {p[k]!r})")

    # Ground plate depth = board_hsub - ifa_h - ifa_te must be > 0
    if ("board_hsub" in p) and ("ifa_h" in p) and ("ifa_te" in p):
        if float(p["board_hsub"]) - float(p["ifa_h"]) - float(p["ifa_te"]) <= 0:
            raise ValueError("board_hsub must be > ifa_h + ifa_te (ground plate depth > 0)")

    # Via must fit through board
    if ("via_size" in p) and ("board_th" in p):
        if float(p["via_size"]) <= 0 or float(p["board_th"]) <= 0:
            raise ValueError("via_size and board_th must be > 0")

# --- child worker that runs one simulation and returns (freq_dense, RL_dB array) ---
def _eval_worker_multi(params: dict, queue: mp.Queue, solver_name: str = "PARDISO"):
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_MAX_THREADS"] = "1"
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # debug only

        import emerge as em
        from ifalib import build_mifa, get_loss

        # Map string -> enum safely inside the child
        try:
            solver = getattr(em.EMSolver, solver_name)
        except AttributeError:
            solver = em.EMSolver.PARDISO  # safe fallback

        # (Optional) quick geometry guard to avoid impossible plates/boxes
        _precheck_params(params)

        model, S11, freq_dense, *_ = build_mifa(
            params,
            view_model=False,
            run_simulation=True,
            compute_farfield=False,
            solver=solver
        )

        import numpy as _np
        rl_db = _np.array([float(get_loss(S11, float(f), freq_dense)) for f in freq_dense], dtype=float)
        queue.put(("ok", (list(map(float, freq_dense)), rl_db.tolist())))
    except Exception as e:
        # Return the full message so parent can log it
        queue.put(("pyerr", f"{type(e).__name__}: {e}"))

def _safe_sim_rl_multi(params: dict, timeout: float = 600.0, solver: em.EMSolver = em.EMSolver.PARDISO):
    """
    Isolated process evaluation.
    Returns (ok, (freq_list, rl_db_list)) or (False, reason_str).
    """
    # Only pass primitives to child
    solver_name = getattr(solver, "name", None) or str(solver).split(".")[-1]

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_eval_worker_multi, args=(params, q, solver_name))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join(5)

    if not q.empty():
        status, payload = q.get_nowait()
        if status == "ok":
            return True, payload
        else:
            return False, payload
    else:
        return False, "native_crash_or_timeout"


_MM_KEYS = {
    'ifa_h','ifa_l','ifa_w1','ifa_w2','ifa_wf','ifa_fp','ifa_e','ifa_e2','ifa_te','via_size',
    'board_wsub','board_hsub','board_th','mifa_meander','mifa_meander_edge_distance','mifa_tipdistance'
}
_FREQ_KEYS = {'f0','f1','f2'}

def _fmt_params_singleline_mm(params: dict, precision: int = 3) -> str:
    """
    Return a single-line, copy-pastable dict like:
    { 'ifa_h': 6.0*mm, 'ifa_l': 26*mm, 'f0': 2.45e9, ... }
    - Geometry keys use '*mm'
    - Frequencies are plain Hz numbers (no GHz)
    - Others are printed with repr()
    """
    pairs = []
    for k in sorted(params.keys()):
        v = params[k]
        # normalize numpy types
        if hasattr(v, 'item'):
            v = v.item()
        if k in _MM_KEYS and isinstance(v, (int, float)):
            val_mm = float(v) / mm
            s = f"{val_mm:.{precision}f}*mm"
        elif k in _FREQ_KEYS and isinstance(v, (int, float)):
            # Hz, no units; keep in scientific/compact form
            s = f"{float(v):.9g}"
        else:
            s = repr(v)
        pairs.append(f"'{k}': {s}")
    return "{ " + ", ".join(pairs) + " }"

def _fmt_params_singleline_raw(params: dict, float_fmt: str = ".9g", sort_keys: bool = False) -> str:
    """
    Return a single-line, copy-pastable dict with RAW values from `params`.
    - Geometry stays in meters (no '*mm')
    - Frequencies stay in Hz (no 'GHz')
    - Floats/ints formatted with `float_fmt` (default '.9g'), others via repr()
    - Set sort_keys=False to preserve insertion order
    """
    def norm_num(x):
        try:
            # handle numpy scalars cleanly
            if hasattr(x, "item"):
                x = x.item()
        except Exception:
            pass
        return x

    items = params.items()
    if sort_keys:
        items = sorted(items, key=lambda kv: kv[0])

    parts = []
    for k, v in items:
        v = norm_num(v)
        if isinstance(v, (int, float)):
            s = format(v, float_fmt)
        else:
            s = repr(v)
        parts.append(f"'{k}': {s}")
    return "{ " + ", ".join(parts) + " }"

def _ensure_bounds_in_meters(optimize_parameters: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    cleaned = {}
    for k, v in optimize_parameters.items():
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise ValueError(f"Bounds for '{k}' must be a 2-tuple/list in METERS (e.g., [0.5*mm, 1.0*mm]). Got: {v}")
        lo, hi = float(v[0]), float(v[1])
        if not (lo < hi):
            raise ValueError(f"Lower bound must be < upper bound for '{k}'. Got {v}")
        cleaned[k] = (lo, hi)
    return cleaned

def _pack_params(base: Dict[str, float], var_keys: List[str], x: np.ndarray) -> Dict[str, float]:
    p = copy.deepcopy(base)
    for k, val in zip(var_keys, x):
        p[k] = float(val)
    return p


# ---------- Objective factory with logging ----------
def _objective_factory(
    base_parameters,
    var_bounds_m,
    *,
    logger: OptLogger,
    log_every_eval: bool = False,
    penalty_if_fail: float = 1e6,
    bandwidth_target_db: Optional[float] = None,
    bandwidth_span: Optional[Tuple[float, float]] = None,
    bandwidth_weight: float = 0.0,
    solver: em.EMSolver = em.EMSolver.PARDISO,
    timeout: float = 600.0
):
    var_keys = list(var_bounds_m.keys())
    state = {"evals": 0, "best_obj": np.inf, "best_rl": -np.inf}

    def _objective(x: np.ndarray) -> float:
        state["evals"] += 1
        params = _pack_params(base_parameters, var_keys, x)

        if log_every_eval:
            short = ", ".join([f"{k}={params[k]/mm:.3f}mm" for k in var_keys])
            logger.info(f"[eval {state['evals']:04d}] Simulating with {short}")

        ok, payload = _safe_sim_rl_multi(params, timeout=timeout, solver=solver)
        if not ok:
            logger.warn(f"[eval {state['evals']:04d}] simulation failed ({payload}); applying penalty {penalty_if_fail:g}")
            return float(penalty_if_fail)

        freq_list, rl_list = payload
        # Convert to numpy for convenience
        freq = np.array(freq_list, dtype=float)
        rl   = np.array(rl_list, dtype=float)

        # RL at f0 (interpolate)
        f0 = float(params['f0'])
        # robust interp: clamp to range
        f0_clamped = min(max(f0, freq.min()), freq.max())
        rl_f0 = float(np.interp(f0_clamped, freq, rl))

        # Optional bandwidth reward over [f_lo, f_hi]
        bandwidth_bonus = 0.0
        frac_ok = None
        if (bandwidth_target_db is not None) and (bandwidth_span is not None) and (bandwidth_weight > 0.0):
            f_lo, f_hi = float(bandwidth_span[0]), float(bandwidth_span[1])
            if f_hi < f_lo:
                f_lo, f_hi = f_hi, f_lo
            m = (freq >= f_lo) & (freq <= f_hi)
            if np.any(m):
                # fraction of points meeting/exceeding the |S11| target
                frac_ok = float(np.mean(rl[m] >= abs(bandwidth_target_db)))
                bandwidth_bonus = bandwidth_weight * frac_ok
            else:
                frac_ok = 0.0  # no points in band

        obj = float(-(rl_f0) - bandwidth_bonus)

        improved = obj < state["best_obj"]
        if improved or log_every_eval:
            base_msg = f"[eval {state['evals']:04d}] RL@f0={rl_f0:.2f} dB"
            if frac_ok is None:
                logger.info(f"{base_msg}, objective={obj:.4f}" + ("  [NEW BEST]" if improved else ""))
            else:
                logger.info(
                    f"{base_msg}, BW_frac≥{abs(bandwidth_target_db):.1f}dB={frac_ok:.3f}, objective={obj:.4f}"
                    + ("  [NEW BEST]" if improved else "")
                )

        if improved:
            state["best_obj"] = obj
            state["best_rl"]  = rl_f0
            logger.info("NEW BEST PARAMS: " + _fmt_params_singleline_raw(params))

        return obj

    return _objective, var_keys


# ---------- Top-level optimizer with iteration callback ----------
# ---------- Top-level optimizer with iteration callback ----------
def optimize_ifa(
    start_parameters: Dict[str, float],
    optimize_parameters: Dict[str, Tuple[float, float]],
    *,
    maxiter: int = 25,
    popsize: int = 12,
    seed: int = 42,
    polish: bool = True,
    log_every_eval: bool = False,      # set True if you want every evaluation logged
    bandwidth_target_db: Optional[float] = None,
    bandwidth_span: Optional[Tuple[float, float]] = None,
    bandwidth_weight: float = 0.0,
    solver: em.EMSolver = em.EMSolver.CUDSS,
    timeout: float = 600.0,
    include_start: bool = True,        # NEW: ensure start point is evaluated
    start_jitter: float = 0.05         # NEW: fraction of bound span for Gaussian jitter
):
    logger = OptLogger(enabled=True)

    # Announce optimization plan
    dims = len(optimize_parameters)
    logger.info(f"Initializing optimizer over {dims} parameters.")
    for k, (lo, hi) in _ensure_bounds_in_meters(optimize_parameters).items():
        logger.info(f"Bounds {k}: [{lo/mm:.3f}, {hi/mm:.3f}] mm")

    bounds_m = _ensure_bounds_in_meters(optimize_parameters)
    objective, var_keys = _objective_factory(
        start_parameters,
        bounds_m,
        logger=logger,
        log_every_eval=log_every_eval,
        bandwidth_target_db=bandwidth_target_db,
        bandwidth_span=bandwidth_span,
        bandwidth_weight=bandwidth_weight,
        solver=solver,
        timeout=timeout,
    )
    bounds_list = [bounds_m[k] for k in var_keys]

    # SciPy iteration callback
    iter_state = {"iter": 0}
    def _cb(xk, convergence):
        iter_state["iter"] += 1
        # xk is current best vector
        p = _pack_params(start_parameters, var_keys, xk)
        msg = ", ".join([f"{k}={p[k]/mm:.3f}mm" for k in var_keys])
        logger.info(f"[iter {iter_state['iter']:03d}] Best-so-far: {msg}  (convergence={convergence:.3g})")
        return False  # continue

    # ---- NEW: build initial population that includes your exact start (clamped) + jittered neighbors
    init_arg = "latinhypercube"  # SciPy default if we don't override
    if include_start:
        dim = len(var_keys)
        pop_n = popsize * dim
        rng = np.random.default_rng(seed)

        # Exact start vector, clamped to bounds
        start_vec = np.array(
            [np.clip(start_parameters[k], *bounds_m[k]) for k in var_keys], dtype=float
        )

        init_pop = [start_vec]
        for _ in range(pop_n - 1):
            row = start_vec.copy()
            for i, k in enumerate(var_keys):
                lo, hi = bounds_m[k]
                span = hi - lo
                if start_jitter and span > 0:
                    row[i] = np.clip(rng.normal(loc=start_vec[i], scale=start_jitter*span), lo, hi)
                else:
                    row[i] = rng.uniform(lo, hi)
            init_pop.append(row)
        init_pop = np.vstack(init_pop)
        init_arg = init_pop  # supply explicit initial population to DE
        logger.info(f"Starting differential evolution with start included; pop={pop_n}, dim={dim}, jitter={start_jitter:.3f}")

    logger.info(f"Starting differential evolution: maxiter={maxiter}, popsize={popsize}, polish={polish}")
    result = differential_evolution(
        objective,
        bounds=bounds_list,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=polish,
        updating='deferred',
        workers=1,          # set to -1 if thread-safe (and ensure __main__ guard)
        callback=_cb,
        tol=0,              # run full maxiter; no early stop on convergence
        init=init_arg       # <- includes your start (and jittered samples) if include_start=True
    )

    best_params = _pack_params(start_parameters, var_keys, result.x)

    # Final simulate + plot
    logger.info("Optimization complete. Running final verification simulation for best parameters.")
    model, S11, freq_dense, *_ = build_mifa(
        best_params,
        view_model=False,
        run_simulation=True,
        compute_farfield=False
    )
    rl_best = get_loss(S11, best_params['f0'], freq_dense)

    logger.info(f"Best RL@f0 = {rl_best:.2f} dB")
    for k in var_keys:
        logger.info(f"  {k:>24s} = {best_params[k]/mm:.3f} mm")

    # Single-line, copy-pastable RAW dict for the winner
    logger.info("FINAL BEST PARAMS: " + _fmt_params_singleline_raw(best_params))

    plot_sp(freq_dense, S11)

    summary = {
        "best_return_loss_dB_at_f0": float(rl_best),
        "best_params": {k: float(best_params[k]) for k in best_params},
        "optimizer_message": result.message,
        "optimizer_nfev": int(result.nfev),
        "optimizer_success": bool(result.success),
        "optimizer_fun": float(result.fun),
    }
    return best_params, result, summary
