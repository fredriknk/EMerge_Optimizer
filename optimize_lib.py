import os, sys, time, copy
import numpy as np
import multiprocessing as mp
from typing import Dict, Tuple, List, Optional
from scipy.optimize import differential_evolution
import json, os, math
from dataclasses import asdict
from ifalib2 import get_loss_at_freq, normalize_params_sequence

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
    pos_keys = {
        "ifa_h","ifa_l","ifa_w1","ifa_w2","ifa_wf","ifa_fp","ifa_e","ifa_e2","ifa_te","via_size",
        "wsub","hsub","th","board_wsub","board_hsub","board_th",
        "mifa_meander","mifa_low_dist","mifa_tipdistance"
    }
    for k in pos_keys:
        if k in p and not (float(p[k]) > 0.0):
            raise ValueError(f"{k} must be > 0 (got {p[k]!r})")

    hsub = p.get("hsub", p.get("board_hsub"))
    if hsub is not None and "ifa_h" in p and "ifa_te" in p:
        if float(hsub) - float(p["ifa_h"]) - float(p["ifa_te"]) <= 0.0:
            raise ValueError("substrate height must be > ifa_h + ifa_te (ground clearance > 0)")
        
# -------- Infeasible handling (uses your validator) --------
def _is_valid_params(params: dict) -> Tuple[bool, List[str]]:
    try:
        from ifa_validation import validate_ifa_params
    except Exception as e:
        # If validator missing, consider everything valid (no surprise crashes)
        return True, [f"validator_unavailable: {e}"]
    errs, warns, drv = validate_ifa_params(params)
    return (len(errs) == 0), errs

def _random_valid_vector(
    base_parameters: Dict[str, float],
    var_keys: List[str],
    bounds_m: Dict[str, Tuple[float, float]],
    rng: np.random.Generator,
    *,
    max_tries: int = 500
) -> np.ndarray:
    """Sample uniformly in bounds until validate_ifa_params() passes."""
    for _ in range(max_tries):
        row = []
        for k in var_keys:
            lo, hi = bounds_m[k]
            row.append(rng.uniform(lo, hi))
        params = _pack_params(base_parameters, var_keys, np.array(row, dtype=float))
        ok, _ = _is_valid_params(params)
        if ok:
            return np.array(row, dtype=float)
    # Fallback: return uniform draw (may be invalid; objective will penalize)
    return np.array([rng.uniform(*bounds_m[k]) for k in var_keys], dtype=float)


# --- child worker that runs one simulation and returns (freq_dense, RL_dB array) ---
def _eval_worker_multi(params: dict, conn, solver_name: str = "PARDISO", project_root: Optional[str] = None):
    """
    Quiet worker with heartbeat. Emits:
      - ('hb', {'label': <stage>, 'elapsed': <seconds>, 'n': <count>}) every few seconds
      - ('log', 'build_mifa_done') and ('log', 'postprocess_done') at milestones
      - ('ok', (freq_list, rl_list)) on success, or ('pyerr', '...') on error
    """
    import time, threading, os, sys
    try:
        # hygiene
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_MAX_THREADS"] = "1"

        if project_root and project_root not in sys.path:
            sys.path.insert(0, project_root)

        import numpy as _np
        import emerge as em
        from ifalib2 import build_mifa

        try:
            solver = getattr(em.EMSolver, solver_name)
        except Exception:
            solver = em.EMSolver.PARDISO  # safe fallback
            
        _precheck_params(params)

        # heartbeat helper
        def heartbeat(label: str, stop_evt: threading.Event, period: float = 5.0):
            t0 = time.monotonic(); n = 0
            while not stop_evt.wait(period):
                n += 1
                try:
                    conn.send(("hb", {"label": label, "elapsed": time.monotonic()-t0, "n": n}))
                except Exception:
                    break

        # --- build & run with heartbeat ---
        stop_hb = threading.Event()
        th = threading.Thread(target=heartbeat, args=("build", stop_hb, 5.0), daemon=True)
        th.start()
        
        model, S11, freq_dense, *_ = build_mifa(
            params,
            view_model=False,
            run_simulation=True,
            compute_farfield=False,
            solver=solver,
        )

        stop_hb.set(); th.join(timeout=1.0)
        try: 
            conn.send(("log", "build_mifa_done")) 
        except: pass

        try: 
            conn.send(("log", "postprocess_done"))
        except: pass

        conn.send(("ok", (list(map(float, freq_dense)), S11)))
    except Exception as e:
        try:
            conn.send(("pyerr", f"{type(e).__name__}: {e}"))
        except Exception:
            pass
    finally:
        try: conn.close()
        except Exception: pass


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _safe_sim_rl_multi(
    params: dict,
    timeout: float = 600.0,
    solver_name: str = "PARDISO",
    logger: OptLogger = None,
    *,
    # Heartbeat options
    hb_print: bool = True,
    hb_min_interval: float = 10.0,  # seconds between prints
    # ETA hint from caller (coarse)
    eta_done_evals: Optional[int] = None,
    eta_total_evals: Optional[int] = None,
    eta_avg_eval_s: Optional[float] = None,
):
    """
    Isolated process evaluation over a Pipe.
    Streams ('hb', {...}) and only prints every hb_min_interval seconds.
    If eta_* hints are provided, prints an ETA for the whole DE run.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    project_root = os.path.dirname(os.path.abspath(__file__))
    p = ctx.Process(target=_eval_worker_multi, args=(params, child_conn, solver_name, project_root))
    p.start()
    child_conn.close()

    start = time.monotonic()
    final_status = final_payload = None
    last_hb_print = 0.0

    def _maybe_print_hb(label: str, elapsed: float):
        nonlocal last_hb_print
        if not hb_print:
            return
        now = time.monotonic()
        if now - last_hb_print < hb_min_interval:
            return
        last_hb_print = now
        msg = f"[hb] {label} elapsed={_fmt_hms(elapsed)}"
        # ETA if we have hints
        if eta_done_evals is not None and eta_total_evals is not None and eta_avg_eval_s:
            remaining = max(eta_total_evals - eta_done_evals, 0)
            eta_sec = remaining * float(eta_avg_eval_s)
            msg += f", ETA(total) ~ { _fmt_hms(eta_sec) }, Average Runtime: {eta_avg_eval_s:.1f}s"
        if logger: logger.info(msg)
        else: print(msg, flush=True)

    while True:
        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0:
            final_status, final_payload = None, "pipe_timeout_waiting_for_child"
            break

        if parent_conn.poll(max(0.2, min(2.0, remaining))):
            try:
                status, payload = parent_conn.recv()
            except EOFError:
                final_status, final_payload = None, "child_closed_pipe"
                break

            if status == "hb":
                label = payload.get("label", "?"); elapsed = float(payload.get("elapsed", 0.0))
                _maybe_print_hb(label, elapsed)
                continue

            if status == "log":
                # quiet milestone logs: one-liners
                if logger: logger.info(f"[child] {payload}")
                continue

            final_status, final_payload = status, payload
            break
        else:
            if not p.is_alive():
                final_status, final_payload = None, "child_exited_without_message"
                break

    try:
        parent_conn.close()
    except Exception:
        pass

    p.join(3.0)
    if p.is_alive():
        p.terminate()
        p.join(3.0)

    if final_status == "ok":
        return True, final_payload
    elif final_status == "pyerr":
        return False, final_payload
    else:
        return False, final_payload or "native_crash_or_timeout"




_MM_KEYS = {
    'ifa_h','ifa_l','ifa_w1','ifa_w2','ifa_wf','ifa_fp','ifa_e','ifa_e2','ifa_te','via_size',
    'board_wsub','board_hsub','board_th','mifa_meander','mifa_low_dist','mifa_tipdistance'
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
    solver_name: str = "PARDISO", timeout: float = 600.0, maxiter_hint: int = None, popsize_hint: int = None,
    stage_name: str = "default",
    bandwidth_parameters: Optional[Dict[str, float]],
    ):
    
    if bandwidth_parameters is None:
        bandwidth_parameters = {"mean_excess_weight":1.0,"max_excess_factor":0.2,"center_weighting_factor":0.2,"mean_power_weight":0.1}
        
    var_keys = list(var_bounds_m.keys())
    state = {
        "evals": 0,
        "best_obj": np.inf,
        "best_rl": -np.inf,
        "t_last_start": None,
        "avg_eval_s": None,
        "total_evals_est": None,
    
    }

    # estimate total evals once
    if (maxiter_hint is not None) and (popsize_hint is not None):
        pop = popsize_hint * len(var_keys)
        state["total_evals_est"] = pop * (maxiter_hint + 1)  # initial pop + iterations

    def _objective(x: np.ndarray) -> float:
        state["evals"] += 1
        params = _pack_params(base_parameters, var_keys, x)

        # start timer for this eval
        state["t_last_start"] = time.perf_counter()
        ok_params, errs = _is_valid_params(params)
        if not ok_params:
            if logger:
                logger.warn(
                    f"[eval {state['evals']:04d}] infeasible params -> penalty; reasons: {', '.join(errs[:3])}"
                    + ("..." if len(errs) > 3 else "")
                )
            return float(penalty_if_fail)
        
        ok, payload = _safe_sim_rl_multi(
            params,
            timeout=timeout,
            solver_name=solver_name,
            logger=logger,
            hb_print=True,
            hb_min_interval=10.0,
            eta_done_evals=(state["evals"] - 1),             # completed before this eval
            eta_total_evals=state["total_evals_est"],
            eta_avg_eval_s=state["avg_eval_s"],
        )
        params_dict = dict(params)
        # update avg eval duration
        dt = time.perf_counter() - state["t_last_start"]
        if state["avg_eval_s"] is None:
            state["avg_eval_s"] = dt
        else:
            state["avg_eval_s"] = 0.8 * state["avg_eval_s"] + 0.2 * dt  # EMA smoothing
            
        if not ok:
            logger.warn(f"[eval {state['evals']:04d}] simulation failed ({payload}); applying penalty {penalty_if_fail:g}")
            return float(penalty_if_fail)
        # --- Normalize payload to RL[dB] no matter what the simulator returned ---
        freq_dense, S11 = payload
        
        def _as_rl_db(y: np.ndarray) -> np.ndarray:
                """y can be RL[dB] (negative), |S11| (0..1), or complex S11."""
                y = np.asarray(y)
                if np.iscomplexobj(y):
                    mag = np.abs(y)
                    return -20.0 * np.log10(np.clip(mag, 1e-12, 1.0))
                y = y.astype(float)
                if np.nanmax(y) <= 1.0 and np.nanmin(y) >= 0.0:
                    # Looks like |S11| magnitude
                    return -20.0 * np.log10(np.clip(y, 1e-12, 1.0))
                # Assume already RL in dB; force negative convention
                return -np.abs(y)
        params = asdict(normalize_params_sequence(params)[0])
        print(params["sweep_freqs"])
        
        if "sweep_freqs" in params and params["sweep_freqs"] is not None:
            s11_db = np.asarray(get_loss_at_freq(S11, params["sweep_freqs"], freq_dense), dtype=float)  # <=0 dB
            print(f"S11 return loss (dB) at {params['sweep_freqs']/1e9} GHz: {get_loss_at_freq(S11, params['sweep_freqs'], freq_dense)} dB")
            # Convert S11(dB) -> |Γ|
            gamma = 10.0 ** (-s11_db / 20.0)          # |Γ| in [0,1)
            rl = s11_db                          # Return loss (positive dB) for logging

            if "sweep_weights" in params:
                weights = np.asarray(params["sweep_weights"], dtype=float)
                weights = weights / np.sum(weights)
                obj = float(np.sum(weights * (gamma ** 2)))    # minimize mean |Γ|^2
            else:
                obj = float(np.mean(gamma ** 2))

            logger.info(
                f"[eval {state['evals']:04d}] RL@f_sweep={rl} dB, obj={obj:.6f}"
                + ("  [NEW BEST]" if obj < state.get("best_obj", np.inf) else "")
            )
            
        else:
            # Numpy-ize
            freq  = np.array(freq_dense, dtype=float)
            rl_db = _as_rl_db(np.array(S11))    # this may be negative; keep as-is for logging

            # Use *positive* RL for all math:
            rl_pos = np.abs(rl_db)  
            def _gamma_from_rl_pos_db(rl_pos_db_arr: np.ndarray) -> np.ndarray:
                # RL_pos_dB = -20*log10|Γ|  -> |Γ| = 10^(-RL_pos/20)  (always ≤ 1)
                return 10.0 ** (-np.asarray(rl_pos_db_arr, dtype=float) / 20.0)
            def _gamma_from_rl_db(rl_db_arr: np.ndarray) -> np.ndarray:
                # RL_dB = -20*log10|Γ|  -> |Γ| = 10^(-(-RL)/20) = 10^(RL/20)
                return 10.0 ** (np.asarray(rl_db_arr, dtype=float) / 20.0)

            # Interpolate RL at f0 for logging and (optional) center penalty
            f0 = float(params['f0'])
            f0_clamped = min(max(f0, freq.min()), freq.max())
            rl_f0 = float(np.interp(f0_clamped, freq, rl_db))

            # If we have a target and span -> optimize bandwidth (band-integrated excess |Γ|)
            use_band = (bandwidth_target_db is not None) and (bandwidth_span is not None)
            frac_ok = None  # for logging
            
            if use_band:
                f_lo, f_hi = float(bandwidth_span[0]), float(bandwidth_span[1])
                if f_hi < f_lo:
                    f_lo, f_hi = f_hi, f_lo
                m = (freq >= f_lo) & (freq <= f_hi)
                if not np.any(m):
                    # Band does not overlap simulated grid -> hard penalty
                    if logger:
                        logger.warn(f"[eval {state['evals']:04d}] band [{f_lo:.3g},{f_hi:.3g}] not in freq grid -> penalty")
                    return float(penalty_if_fail)

                # Linear reflection (|Γ|) in-band
                gamma  = _gamma_from_rl_pos_db(rl_pos[m])          # |Γ|
                gamma2 = gamma * gamma                              # |Γ|^2 (power)

                # Target in linear (e.g., target -10 dB -> |Γ|_t = 0.316)
                rl_target = abs(float(bandwidth_target_db))         # e.g. 10 for -10 dB
                g_t  = 10.0 ** (-rl_target / 20.0)                  # |Γ| target
                g2_t = g_t * g_t                                    # power target

                # Excess above target (≥ 0)
                excess = np.clip(gamma - g_t, 0.0, None)

                # Trapezoidal mean excess normalized by band width (handles non-uniform freq grids)
                band_width = float(f_hi - f_lo)
                if band_width <= 0.0:
                    return float(penalty_if_fail)
                mean_excess_weight = bandwidth_parameters["mean_excess_weight"]
                mean_excess = float(np.trapezoid(excess, freq[m]) / band_width)

                # Robustness: small worst-case term to suppress narrow spikes
                max_excess_factor = bandwidth_parameters["max_excess_factor"]

                max_excess = float(np.max(excess))

                # Optional center weighting (very light)
                center_weighting_factor = bandwidth_parameters["center_weighting_factor"]
                ex0 = float(max(_gamma_from_rl_pos_db([abs(rl_f0)])[0] - g_t, 0.0))

                # Gentle preference for deeper-than-target match (smaller |Γ|^2)
                mean_power = float(np.trapezoid(gamma2, freq[m]) / band_width)   # average |Γ|^2 over band
                mean_power_weight = bandwidth_parameters["mean_power_weight"]

                obj = mean_excess_weight * mean_excess + max_excess_factor * max_excess + center_weighting_factor * ex0 + mean_power_weight * (mean_power / g2_t)  # minimize

                # Logging aids (meeting spec means RL[dB] <= -rl_target)
                rl_spec_db = -rl_target
                frac_ok = float(np.mean(rl_db[m] <= rl_spec_db))
                rl_min_band = float(np.min(rl_db[m]))
                rl_frequency_band = freq[m][np.argmin(rl_db[m])]
                rl = rl_f0  # for logging
                if log_every_eval or obj < state["best_obj"]:
                    logger.info(
                        f"[eval {state['evals']:04d}] RL@f0={rl_f0:.2f} dB, "
                        f"band[{f_lo/1e9:.3f}-{f_hi/1e9:.3f} GHz]: "
                        f"minRL={rl_min_band:.2f} dB, resonant_freq={rl_frequency_band/1e9:.3f} GHz, frac≥{rl_target:.0f}dB={frac_ok:.3f}, "
                        f"obj={obj:.6f}"
                        + ("  [NEW BEST]" if obj < state["best_obj"] else "")
                    )
            else:
                # Fallback: minimize |Γ| at f0 (single-point)
                rl = rl_f0  # for logging
                gam_f0 = float(_gamma_from_rl_pos_db(np.array([rl_f0]))[0])
                obj = gam_f0
                if log_every_eval or obj < state["best_obj"]:
                    logger.info(
                        f"[eval {state['evals']:04d}] RL@f0={rl_f0:.2f} dB, obj(|Γ(f0)|)={obj:.6f}"
                        + ("  [NEW BEST]" if obj < state["best_obj"] else "")
                    )

        # Track best & persist
        improved = obj < state["best_obj"]
        if improved:
            state["best_obj"] = obj
            state["best_rl"]  = rl
            logger.info("NEW BEST PARAMS: " + _fmt_params_singleline_raw(params_dict))
            os.makedirs("best_params_logs", exist_ok=True)
            with open(f"best_params_logs/{stage_name}.log","a",encoding="utf-8") as f:
                f.write(f"{state['evals']},{rl},{obj:.9f}," + _fmt_params_singleline_raw(params_dict) + "\n")

        return obj


    return _objective, var_keys

# ---------- Bounds shrinking utility ----------
def clamp(x, lo, hi): return max(lo, min(hi, x))

def shrink_bounds_around_best(best: dict, prev_bounds: Dict[str, Tuple[float, float]],
                              *, shrink: float, min_span_mm: float = 0.1) -> Dict[str, Tuple[float, float]]:
    """
    Create new bounds centered at best[k] with span = shrink * previous span.
    min_span_mm is the minimum *total* span (meters) to keep search alive.
    """
    out = {}
    min_span = min_span_mm * mm
    for k, (lo0, hi0) in prev_bounds.items():
        mid = float(best[k])
        span0 = hi0 - lo0
        new_span = max(shrink * span0, min_span)
        lo = clamp(mid - 0.5*new_span, lo0, hi0)
        hi = clamp(mid + 0.5*new_span, lo0, hi0)
        # If clamped collapsed, re-center within original window
        if hi - lo < min_span:
            pad = 0.5*min_span
            lo = clamp(mid - pad, lo0, hi0)
            hi = clamp(mid + pad, lo0, hi0)
        out[k] = (lo, hi)
    return out

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def append_trace(csv_path: str, stage: str, evals: int, best_rl: float, obj: float, params: dict):
    hdr_needed = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if hdr_needed:
            f.write("stage,evals,best_rl_dB,objective,params\n")
        f.write(f"{stage},{evals},{best_rl:.3f},{obj:.6f},\"{_fmt_params_singleline_raw(params, sort_keys=False)}\"\n")

def global_optimizer(stage_name: str, params: dict, opt_bounds: Dict[str, Tuple[float, float]],
              *, maxiter: int, popsize: int, seed: int,
              solver_name: str, timeout: float,
              bandwidth_target_db: float = None, bandwidth_span = None,
              bandwidth_parameters: Optional[Dict[str, float]] = None,
              include_start: bool, log_every_eval: bool):
    print(f"\n=== Stage: {stage_name} ===")
    best_params, result, summary = optimize_ifa(
        start_parameters=params,
        optimize_parameters=opt_bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=False,                      # keep fast in staged passes; final verify happens inside
        solver_name=solver_name,
        timeout=timeout,
        bandwidth_target_db=bandwidth_target_db,
        bandwidth_span=bandwidth_span,
        bandwidth_parameters=bandwidth_parameters,
        include_start=include_start,
        log_every_eval=log_every_eval,
        stage_name=stage_name
    )
    os.makedirs("best_params_logs", exist_ok=True)
    print(f"[{stage_name}] best RL@f0: {summary['best_return_loss_dB_at_f0']:.2f} dB")
    print(f"[{stage_name}] best params: {_fmt_params_singleline_raw(summary['best_params'], sort_keys=False)}")
    append_trace("best_params_logs/best_trace.csv", stage_name, summary["optimizer_nfev"],
                 summary["best_return_loss_dB_at_f0"], summary["optimizer_fun"], summary["best_params"])
    write_json(f"best_params_logs/best_params_{stage_name}.json", summary["best_params"])
    # Update our live params for the next stage
    params.update(summary["best_params"])
    return best_params, result, summary

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
    bandwidth_parameters: Optional[Dict[str, float]] = None,
    solver_name: str = "CUDSS",
    timeout: float = 600.0,
    include_start: bool = False,        # NEW: ensure start point is evaluated
    stage_name: str = "default"
):
    logger = OptLogger(enabled=True)

    # Announce optimization plan
    dims = len(optimize_parameters)
    logger.info(f"Initializing optimizer over {dims} parameters.")
    for k, (lo, hi) in optimize_parameters.items():
        logger.info(f"Bounds {k}: [{lo/mm:.3f}, {hi/mm:.3f}] mm")

    bounds_m = optimize_parameters
    
    objective, var_keys = _objective_factory(
    start_parameters, bounds_m, logger=logger, log_every_eval=log_every_eval,
    bandwidth_target_db=bandwidth_target_db, bandwidth_span=bandwidth_span,
    solver_name=solver_name, timeout=timeout,
    maxiter_hint=maxiter, popsize_hint=popsize, stage_name=stage_name,
    bandwidth_parameters=bandwidth_parameters
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
    init_arg = "latinhypercube"
    if include_start:
        dim = len(var_keys)
        pop_n = popsize * dim
        rng = np.random.default_rng(seed)

        # Exact start vector, clamped
        start_vec = np.array([np.clip(start_parameters[k], *bounds_m[k]) for k in var_keys], dtype=float)
        start_params = _pack_params(start_parameters, var_keys, start_vec)
        ok_start, _ = _is_valid_params(start_params)

        init_rows = []
        if ok_start:
            init_rows.append(start_vec)
        else:
            # Try to nudge the start a bit toward feasibility within bounds
            # (uniform jitter up to 5% span), else skip it.
            jittered = start_vec.copy()
            for i, k in enumerate(var_keys):
                lo, hi = bounds_m[k]
                span = hi - lo
                jittered[i] = float(np.clip(rng.normal(loc=start_vec[i], scale=0.05*span), lo, hi))
            if _is_valid_params(_pack_params(start_parameters, var_keys, jittered))[0]:
                init_rows.append(jittered)

        # Fill the rest with valid randoms
        while len(init_rows) < pop_n:
            init_rows.append(_random_valid_vector(start_parameters, var_keys, bounds_m, rng))

        init_pop = np.vstack(init_rows)
        init_arg = init_pop
        logger.info(f"Starting differential evolution with validated init; pop={pop_n}, dim={dim} "
                    f"(seeded {1 if ok_start else 0} start)")

    constraints_arg = None
    try:
        from scipy.optimize import NonlinearConstraint

        def _feas_margin(x: np.ndarray) -> float:
            """
            Return >= 0 for feasible, < 0 for infeasible.
            We return +1.0 if valid, else -1.0 * min(5, n_errs) to give DE a signal.
            """
            params = _pack_params(start_parameters, var_keys, x)
            ok, errs = _is_valid_params(params)
            return 1.0 if ok else -float(min(5, max(1, len(errs))))

        # Box: [0, +inf) feasible
        constraints_arg = (NonlinearConstraint(_feas_margin, 0.0, np.inf),)
        logger.info("Nonlinear feasibility constraint enabled.")
    except Exception as e:
        logger.warn(f"NonlinearConstraint unavailable; continuing without explicit constraints ({e}).")
    
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
        init=init_arg,       # <- includes your start (and jittered samples) if include_start=True
        constraints=constraints_arg,  # (ignored if None)
    )

    best_params = _pack_params(start_parameters, var_keys, result.x)

    # Final simulate + plot
    logger.info("Optimization complete. Running final verification simulation for best parameters.")
    from ifalib2 import build_mifa, get_loss_at_freq
    import emerge as em

    solver_enum = getattr(em.EMSolver, solver_name, em.EMSolver.PARDISO)
    model, S11, freq_dense, *_ = build_mifa(
        best_params, view_model=False, run_simulation=True, compute_farfield=False, solver=solver_enum
    )
    rl_best = get_loss_at_freq(S11, best_params['p.sweep_freqs'], freq_dense)

    logger.info(f"Best RL@f0 = {rl_best:.2f} dB")
    for k in var_keys:
        logger.info(f"  {k:>24s} = {best_params[k]/mm:.3f} mm")

    # Single-line, copy-pastable RAW dict for the winner
    logger.info("FINAL BEST PARAMS: " + _fmt_params_singleline_raw(best_params))


    summary = {
        "best_return_loss_dB_at_f0": float(rl_best),
        "best_params": {k: float(best_params[k]) for k in best_params},
        "optimizer_message": result.message,
        "optimizer_nfev": int(result.nfev),
        "optimizer_success": bool(result.success),
        "optimizer_fun": float(result.fun),
    }
    return best_params, result, summary

from typing import Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from ifalib2 import resolve_linked_params

def local_minimize_ifa(
    start_parameters: Dict[str, float],
    optimize_parameters: Dict[str, Tuple[float, float]],
    *,
    method: str = "Powell",              # "Powell" or "Nelder-Mead"
    init_step_mm: float = 0.10,          # initial step ~0.10 mm per variable
    maxiter: int = 200,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    bandwidth_target_db: Optional[float] = None,
    bandwidth_span: Optional[Tuple[float, float]] = None,
    solver_name: str = "CUDSS",
    timeout: float = 600.0,
    log_every_eval: bool = False,
    stage_name: str = "scipy_local",
    bandwidth_parameters: Optional[Dict[str, float]] = None
):
    """
    Small-step local optimizer starting at start_parameters.
    - method="Powell": direction-set / line-search (bounded).
    - method="Nelder-Mead": simplex around x0 (bounded).
    init_step_mm controls the initial local step size.
    """
    from scipy.optimize import minimize, Bounds
    from ifalib2 import normalize_params_sequence
    logger = OptLogger(enabled=True)
    mm = 1e-3
    
    bounds = optimize_parameters
    var_keys = list(bounds.keys())

    # Build objective with your existing machinery
    objective, _ = _objective_factory(
        start_parameters, bounds, logger=logger, log_every_eval=log_every_eval,
        bandwidth_target_db=bandwidth_target_db, bandwidth_span=bandwidth_span,
        solver_name=solver_name, timeout=timeout,
        maxiter_hint=None, popsize_hint=None, stage_name=stage_name,
        bandwidth_parameters=bandwidth_parameters
    )
    # Seed vector (clamped)
    x0 = np.array([np.clip(start_parameters[k], *bounds[k]) for k in var_keys], dtype=float)

    # Bounds object for SciPy
    lo = np.array([bounds[k][0] for k in var_keys], dtype=float)
    hi = np.array([bounds[k][1] for k in var_keys], dtype=float)
    sbounds = Bounds(lo, hi, keep_feasible=True)

    # Initial step sizing (meters)
    step = float(init_step_mm) * mm
    # Scale steps relative to each variable span so “small” is meaningful everywhere
    spans = np.maximum(hi - lo, 1e-9)
    dvec  = np.minimum(step, 0.25 * spans)   # don’t jump more than 25% of span

    options = dict(maxiter=maxiter, xtol=xtol, ftol=ftol, maxfev=None)

    # Method-specific seeding for small local moves
    m = method.lower()
    if m == "nelder-mead":
        # Build a tiny simplex around x0 using dvec
        initial_simplex = [x0]
        for i in range(len(var_keys)):
            v = x0.copy()
            v[i] = np.clip(v[i] + dvec[i], lo[i], hi[i])
            initial_simplex.append(v)
        options["initial_simplex"] = np.vstack(initial_simplex)
    elif m == "powell":
        # Powell allows custom initial directions via 'direc'
        # Use scaled coordinate directions with size dvec
        n = len(var_keys)
        direc = np.eye(n)
        for i in range(n):
            direc[i, i] = dvec[i] if dvec[i] > 0 else 1e-6
        options["direc"] = direc

    logger.info(f"[{stage_name}] starting {method} from current best, step≈{init_step_mm} mm")

    res = minimize(
        fun=objective,
        x0=x0,
        method=method,
        bounds=sbounds,
        options=options,
    )

    x_best = np.clip(res.x, lo, hi)
    best_params = _pack_params(start_parameters, var_keys, x_best)

    # Final verification run (your usual epilogue)
    from ifalib2 import build_mifa, get_loss_at_freq
    import emerge as em
    solver_enum = getattr(em.EMSolver, solver_name, em.EMSolver.PARDISO)
    model, S11, freq_dense, *_ = build_mifa(
        best_params, view_model=False, run_simulation=True, compute_farfield=False, solver=solver_enum
    )
    
    query_params = 'p.f0'
    if 'p.sweep_freqs' in best_params:
        query_params = 'p.sweep_freqs'
    rl_best = get_loss_at_freq(S11, best_params[query_params], freq_dense)

    logger.info(f"[{stage_name}] {method} done: fun={float(res.fun):.6f} "
                f"RL@f0={rl_best:.2f} dB, iters={res.nit}, fev={res.nfev}, success={res.success}")

    summary = {
        "method": method,
        "best_params": {k: float(best_params[k]) for k in best_params},
        "optimizer_success": bool(res.success),
        "optimizer_fun": float(res.fun),
        "optimizer_message": str(res.message),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
    }
    return best_params, summary
