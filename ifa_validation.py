#!/usr/bin/env python3
# ifa_validation.py: Validate MIFA/IFA geometry & mesh params.
# validator_normalized_only.py
from typing import Any, Dict, List, Tuple, Mapping
from dataclasses import asdict
import numpy as np
from ifalib2 import normalize_params_sequence, AntennaParams

def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        # numpy scalar?
        try:
            import numpy as np
            if isinstance(x, np.generic):
                return float(x.item())
        except Exception:
            pass
        raise

def _getf(pm: Mapping[str, Any], key: str, default: Any = None) -> float | None:
    v = pm.get(key, default)
    if v is None:
        return None
    return _safe_float(v)

def _validate_ifa_core(pm: Mapping[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    # required geometric/mesh fields (freq handled separately)
    req = [
        'ifa_h','ifa_l','ifa_w1','ifa_w2','ifa_wf','ifa_fp','ifa_e','ifa_e2','ifa_te',
        'via_size','board_wsub','board_hsub','board_th',
        'mifa_meander','mifa_low_dist',
        'mesh_boundary_size_divisor','mesh_wavelength_fraction','lambda_scale'
    ]

    has_triplet = all(k in pm for k in ('f1','f0','f2','freq_points'))
    has_sweep   = 'sweep_freqs' in pm

    # guard: geometry/mesh presence
    missing = [k for k in req if k not in pm]
    if missing:
        return ([f"Missing parameters: {', '.join(missing)}"], [], {})

    if (not has_triplet) and (not has_sweep):
        return (["Provide either (f1,f0,f2,freq_points) or sweep_freqs (± sweep_weights)."], [], {})

    # --- pull values safely ---
    ifa_h  = _getf(pm,'ifa_h');  ifa_l  = _getf(pm,'ifa_l')
    ifa_w1 = _getf(pm,'ifa_w1'); ifa_w2 = _getf(pm,'ifa_w2')
    ifa_wf = _getf(pm,'ifa_wf'); ifa_fp = _getf(pm,'ifa_fp')
    ifa_e  = _getf(pm,'ifa_e');  ifa_e2 = _getf(pm,'ifa_e2')
    ifa_te = _getf(pm,'ifa_te'); via_size = _getf(pm,'via_size')
    board_wsub = _getf(pm,'board_wsub'); board_hsub = _getf(pm,'board_hsub'); board_th = _getf(pm,'board_th')
    mifa_meander = _getf(pm,'mifa_meander'); mifa_low_dist = _getf(pm,'mifa_low_dist')
    shunt = _getf(pm,'shunt')
    
    # default mifa_tipdistance to mifa_low_dist **after** we read it
    mifa_tipdistance = _getf(pm, 'mifa_tipdistance', mifa_low_dist)

    mesh_boundary_size_divisor = _getf(pm,'mesh_boundary_size_divisor')
    mesh_wavelength_fraction   = _getf(pm,'mesh_wavelength_fraction')
    lambda_scale               = _getf(pm,'lambda_scale')
    clearance = _getf(pm, 'clearance', 0.0003)

    errors: List[str] = []
    warnings: List[str] = []
    derived: Dict[str, Any] = {}

    # positivity checks
    for name, val in [
        ("ifa_h",ifa_h),("ifa_l",ifa_l),("ifa_w1",ifa_w1),("ifa_w2",ifa_w2),("ifa_wf",ifa_wf),("ifa_fp",ifa_fp),
        ("ifa_e",ifa_e),("ifa_e2",ifa_e2),("ifa_te",ifa_te),("via_size",via_size),
        ("board_wsub",board_wsub),("board_hsub",board_hsub),("board_th",board_th),
        ("mifa_meander",mifa_meander),("mifa_low_dist",mifa_low_dist),("mifa_tipdistance",mifa_tipdistance),
        ("mesh_boundary_size_divisor",mesh_boundary_size_divisor),("mesh_wavelength_fraction",mesh_wavelength_fraction),
        ("lambda_scale",lambda_scale),
    ]:
        if val is None or val <= 0:
            errors.append(f"{name} must be > 0 (got {val}).")

    # frequency validation (either triplet or sweep)
    if has_triplet and not has_sweep:
        f1 = _getf(pm,'f1'); f0 = _getf(pm,'f0'); f2 = _getf(pm,'f2')
        try:
            freq_points = int(pm['freq_points'])
        except Exception:
            freq_points = -1
        if freq_points < 1:
            errors.append(f"freq_points must be >= 1 (got {freq_points}).")
        if (f1 is None) or (f0 is None) or (f2 is None) or not (f1 <= f0 <= f2):
            errors.append(f"Frequency ordering must be f1 <= f0 <= f2 (got f1={f1}, f0={f0}, f2={f2}).")
        derived.update(dict(f1=f1, f0=f0, f2=f2, freq_points=freq_points))
    else:
        sweep_freqs = np.asarray(pm['sweep_freqs'], dtype=float).ravel()
        if sweep_freqs.size == 0:
            errors.append("sweep_freqs must be non-empty.")
        if np.any(sweep_freqs <= 0):
            errors.append("sweep_freqs must all be > 0.")
        if 'sweep_weights' in pm:
            sweep_weights = np.asarray(pm['sweep_weights'], dtype=float).ravel()
            if sweep_weights.size != sweep_freqs.size:
                errors.append("sweep_weights length must match sweep_freqs.")
            if np.any(sweep_weights < 0):
                errors.append("sweep_weights must be >= 0.")
            derived["sweep_weights_sum"] = float(sweep_weights.sum())
        derived["sweep_freqs_min"] = float(sweep_freqs.min()) if sweep_freqs.size else None
        derived["sweep_freqs_max"] = float(sweep_freqs.max()) if sweep_freqs.size else None

    # mesh sanity notices
    if mesh_wavelength_fraction is not None and not (0.05 <= mesh_wavelength_fraction <= 1.0):
        warnings.append(f"mesh_wavelength_fraction={mesh_wavelength_fraction:.3g} is unusual; typical ~0.2–0.5.")
    if mesh_boundary_size_divisor is not None and not (0.1 <= mesh_boundary_size_divisor <= 0.5):
        warnings.append(f"mesh_boundary_size_divisor={mesh_boundary_size_divisor:.3g} is unusual; typical ~0.2–0.5.")
    if lambda_scale is not None and not (0.25 <= lambda_scale <= 2.0):
        warnings.append(f"lambda_scale={lambda_scale:.3g} outside common 0.5–1.0 range.")

    # geometry rules
    if ifa_fp - ifa_e - ifa_w1 < clearance and shunt:
        errors.append(f"Feedstub Crash: ifa_e + ifa_w1 - ifa_fp = {ifa_e+ifa_w1 - ifa_fp:.3g} < {clearance*1e3:.1f}e-3.")
    if ifa_h + ifa_te > board_hsub:
        errors.append(f"Antenna height exceeds board height: ifa_h + ifa_te = {ifa_h+ifa_te:.3g} > board_hsub={board_hsub:.3g}.")
    if ifa_e + ifa_fp + ifa_wf > board_wsub:
        errors.append(f"Feed stub exceeds board width: ifa_e + ifa_fp + ifa_wf = {ifa_e+ifa_fp+ifa_wf:.3g} > board_wsub={board_wsub:.3g}.")

    if via_size > ifa_w1:
        warnings.append(f"via_size={via_size*1e3:.2f} mm is not smaller than shunt width ifa_w1={ifa_w1*1e3:.2f} mm.")

    # meander rules
    if mifa_meander < 2*ifa_w2 + clearance and ifa_l > board_wsub - ifa_e - ifa_e2:
        errors.append(f"mifa_meander={mifa_meander*1e3:.2f} mm is < 2*w2 + clearance={(2*ifa_w2+clearance)*1e3:.4f} mm.")

    vertical_room = ifa_h - ifa_w2
    derived["vertical_room_for_meander"] = vertical_room
    if mifa_low_dist >= vertical_room:
        errors.append(f"mifa_low_dist={mifa_low_dist*1e3:.2f} mm > ifa_h - w2={vertical_room*1e3:.2f} mm.")
    if mifa_tipdistance >= vertical_room:
        errors.append(f"mifa_tipdistance={mifa_tipdistance*1e3:.2f} mm > ifa_h - w2={vertical_room*1e3:.2f} mm.")

    x_tip      = ifa_e + ifa_l
    x_feed_end = ifa_e + ifa_fp + ifa_wf
    needed_gap = ifa_w2
    available_backspace = (x_tip - (x_feed_end + needed_gap))
    derived.update({
        "x_tip": x_tip,
        "x_feed_end": x_feed_end,
        "required_last_meander_clearance": needed_gap,
        "available_backspace": available_backspace
    })

    n_max = int(available_backspace // mifa_meander) if available_backspace > 0 else 0
    derived["max_meanders_by_length"] = n_max
    if n_max <= 0:
        errors.append("Backwards-grown meander cannot fit even a single segment with the required clearance to the feed area.")

    min_trace = min(ifa_w1, ifa_w2, ifa_wf)
    if min_trace < 0.15e-3:
        warnings.append(f"Very fine copper width detected ({min_trace*1e3:.2f} mm). Check fab capabilities.")

    max_num_meanders = (board_wsub - ifa_e - ifa_e2 - ifa_fp - ifa_wf - ifa_w2) / (mifa_meander)
    max_num_meanders = max_num_meanders - max_num_meanders % 2  # make even
    derived["estimated_number_of_meanders_fit"] = int(max_num_meanders)
    ant_stub = board_wsub - ifa_e - ifa_e2 - max_num_meanders * (mifa_meander) - ifa_w2
    single_meander_length = ifa_h - mifa_low_dist
    tip_length = ifa_h - mifa_tipdistance
    max_length = (max_num_meanders * (mifa_meander + single_meander_length - ifa_w2)) + tip_length + ant_stub
    derived["estimated_max_antenna_length_with_meanders"] = max_length
    if max_length < ifa_l:
        errors.append(f"Estimated max antenna length with meanders ({max_length*1e3:.2f} mm) is less than ifa_l ({ifa_l*1e3:.2f} mm).")

    return (errors, warnings, derived)

def validate_ifa_params_individual(params_any: Any) -> Dict[str, Dict[str, Any]]:
    """
    Normalized-only API. We normalize first; then validate each AntennaParams.
    Returns: {"p": {"errors":[...], "warnings":[...], "derived":{...}}, "p2": ...}
    """
    norm = normalize_params_sequence(params_any)   # -> [AntennaParams, ...]
    out: Dict[str, Dict[str, Any]] = {}
    for i, ap in enumerate(norm):
        alias = "p" if i == 0 else f"p{i+1}"
        errs, warns, drv = _validate_ifa_core(asdict(ap))
        out[alias] = {"errors": errs, "warnings": warns, "derived": drv}
    return out

def validate_ifa_params(params_any: Any) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Normalized-only API compatible with old callers.
    - Normalizes first (supports nested/links/etc.).
    - Validates each normalized profile (p, p2, ...).
    - Returns aggregated (errors, warnings, derived),
      where 'derived' is from alias 'p' (first profile).
    """
    norm = normalize_params_sequence(params_any)  # -> [AntennaParams, ...]
    
    all_errors: List[str] = []
    all_warnings: List[str] = []
    derived_first: Dict[str, Any] = {}

    for i, ap in enumerate(norm):
        alias = "p" if i == 0 else f"p{i+1}"
        if ap.validate: 
            errs, warns, drv = _validate_ifa_core(asdict(ap))
            # tag messages so you can tell which profile they came from
            all_errors.extend([f"[{alias}] {m}" for m in errs])
            all_warnings.extend([f"[{alias}] {m}" for m in warns])
            if i == 0:
                derived_first = drv

    return all_errors, all_warnings, derived_first


if __name__ == "__main__":
    #############################################################
    #|------------- board_wsub----- -------------------|
    # _______________________________________________     _ substrate_thickness
    #| A  ifa_e      |----------ifa_l(total length)-| |\  \-gndplane_position 
    #| V____          _______________     __________  | |  \_0 point
    #|               |    ___  ___   |___|  ______  | | |
    #|         ifa_h |   |   ||   |_________|    |  |_|_|_ mifa_low_dist 
    #|               |   |   ||     <----->      |__|_|_|_|
    #|               |   |   ||   mifa_meander    w2  | | |mifa_tipdistance(Optional, 
    #|_______________|___|___||_______________________| |_|will be set to edge distance if 0)
    #| <---ifa_e---->| w1|   wf\                      | |
    #|               |__fp___|  \                     | |
    #|                       |    feed point          | |
    #|                       |                        | | substrate_length
    #|<- substrate_width/2 ->|                        | |
    #|                                                | |
    #|________________________________________________| |
    # \________________________________________________\|
    #############################################################
    # --- Example usage ---
    parameters = {
        "ifa_h": 0.012,
        "ifa_l": 0.0113,
        "ifa_w1": 0.001,
        "ifa_w2": 0.001,
        "ifa_wf": 0.001,
        "ifa_fp": 0.002,
        "ifa_e": 0.0005,
        "ifa_e2": 0.0005,
        "ifa_te": 0.0005,
        "via_size": 0.0005,
        "board_wsub": 0.012,
        "board_hsub": 0.020,
        "board_th": 0.0015,
        "mifa_meander": 0.002,
        "mifa_low_dist": 0.011,
        "f1": 700000000.0,
        "f0": 800000000.0,
        "f2": 900000000.0,
        "freq_points": 3,
        "mesh_boundary_size_divisor": 0.5,
        "mesh_wavelength_fraction": 0.5,
        "lambda_scale": 0.5,
        "clearance": 0.0003,
        }
    
    parameters= { 'p.board_wsub': 0.0191, 
         'p.board_th': 0.0015, 
         'p.sweep_freqs': np.array([2.45e+09, 5.00e+09]), 
         'p.sweep_weights': np.array([1., 1.]), 
         'p.board_hsub': 0.06, 'p.ifa_e': 0.0005, 
         'p.ifa_e2': 0.000575394784, 'p.ifa_fp': 0.00364461081, 
         'p.ifa_h': 0.00909984885, 'p.ifa_l': 0.0355663827, 
         'p.ifa_te': 0.0005, 
         'p.ifa_w1': 0.00112657281, 'p.ifa_w2': 0.000445781771, 'p.ifa_wf': 0.000398836163, 
         'p.mesh_boundary_size_divisor': 0.33, 
         'p.mesh_wavelength_fraction': 0.2, 'p.mifa_meander': 0.0023, 
         'p.mifa_low_dist': '${p.ifa_h} - 0.003', 
         'p.mifa_tipdistance': '${p.mifa_low_dist}', 
         'p.via_size': 0.0005, 
         'p.lambda_scale': 1, 
         
         'p2.ifa_l': 0.00796519359, 'p2.ifa_h': '${p.mifa_low_dist}- 0.0005', 
         'p2.ifa_e': '${p.ifa_fp}', # When using no shunt set ifa_e to ifa_fp to align
         'p2.ifa_w2': 0.00033083229, 
         'p2.mifa_meander': '${p2.ifa_w2}*2+0.0003', 
         'p2.mifa_low_dist': '${p.mifa_low_dist} - 0.003', 
         'p2.mifa_tipdistance': '${p2.mifa_low_dist}', 
         'p2.shunt': 0 
         }
    
    errs, warns, drv = validate_ifa_params(parameters)

    if errs:
        print("❌ Validation errors:")
        for e in errs:
            print("  -", e)
    else:
        print("✅ No hard errors.")

    if warns:
        print("⚠️ Warnings:")
        for w in warns:
            print("  -", w)

    print("\nDerived:")
    for k,v in drv.items():
        print(f"  {k}: {v}")