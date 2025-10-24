import emerge as em
import numpy as np
from ifalib import build_mifa

def validate_ifa_params(p):
    """
    Validate MIFA/IFA geometry & mesh params.
    Assumes all linear dimensions are in meters.
    Returns (errors, warnings, derived) where:
      - errors   : list[str] (hard failures)
      - warnings : list[str] (soft notices)
      - derived  : dict with helpful computed values
    """
    req = [
        'ifa_h','ifa_l','ifa_w1','ifa_w2','ifa_wf','ifa_fp','ifa_e','ifa_e2','ifa_te',
        'via_size','board_wsub','board_hsub','board_th',
        'mifa_meander','mifa_meander_edge_distance',
        'f1','f0','f2','freq_points',
        'mesh_boundry_size_divisor','mesh_wavelength_fraction','lambda_scale'
    ]
    missing = [k for k in req if k not in p]
    if missing:
        return ([f"Missing parameters: {', '.join(missing)}"], [], {})

    # Pull out vars (readability)
    ifa_h   = float(p['ifa_h'])
    ifa_l   = float(p['ifa_l'])
    w1      = float(p['ifa_w1'])
    w2      = float(p['ifa_w2'])
    wf      = float(p['ifa_wf'])
    fp      = float(p['ifa_fp'])
    e1      = float(p['ifa_e'])
    e2      = float(p['ifa_e2'])
    te      = float(p['ifa_te'])
    via     = float(p['via_size'])
    wsub    = float(p['board_wsub'])
    hsub    = float(p['board_hsub'])
    th      = float(p['board_th'])
    mlen    = float(p['mifa_meander'])
    medge   = float(p['mifa_meander_edge_distance'])
    mtip    = float(p.get('mifa_tipdistance', medge))
    f1      = float(p['f1'])
    f0      = float(p['f0'])
    f2      = float(p['f2'])
    nfreq   = int(p['freq_points'])
    bdiv    = float(p['mesh_boundry_size_divisor'])
    wlfrac  = float(p['mesh_wavelength_fraction'])
    lscale  = float(p['lambda_scale'])

    errors   = []
    warnings = []
    derived  = {}

    # Basic positivity & size sanity
    for name, val in [
        ("ifa_h",ifa_h),("ifa_l",ifa_l),("ifa_w1",w1),("ifa_w2",w2),("ifa_wf",wf),("ifa_fp",fp),
        ("ifa_e",e1),("ifa_e2",e2),("ifa_te",te),("via_size",via),
        ("board_wsub",wsub),("board_hsub",hsub),("board_th",th),
        ("mifa_meander",mlen),("mifa_meander_edge_distance",medge),("mifa_tipdistance",mtip),
        ("f1",f1),("f0",f0),("f2",f2),("mesh_boundry_size_divisor",bdiv),
        ("mesh_wavelength_fraction",wlfrac),("lambda_scale",lscale)
    ]:
        if val <= 0:
            errors.append(f"{name} must be > 0 (got {val}).")

    if nfreq < 2:
        errors.append(f"freq_points must be >= 2 (got {nfreq}).")

    # Frequency ordering
    if not (f1 < f0 < f2):
        errors.append(f"Frequency ordering must be f1 < f0 < f2 (got f1={f1}, f0={f0}, f2={f2}).")

    # Mesh sanity bounds
    if not (0.05 <= wlfrac <= 0.8):
        warnings.append(f"mesh_wavelength_fraction={wlfrac:.3g} is unusual; typical ~0.1–0.5.")
    if not (0.1 <= bdiv <= 2.0):
        warnings.append(f"mesh_boundry_size_divisor={bdiv:.3g} is unusual; typical ~0.2–1.0.")
    if not (0.25 <= lscale <= 2.0):
        warnings.append(f"lambda_scale={lscale:.3g} outside common 0.5–1.0 range.")

    # Board fit (layout axes per your ASCII):
    # X (width) ~ substrate_width = wsub
    # Y (length) ~ substrate_length = hsub
    # Horizontal extent from left edge ~ e1 + ifa_l

    # Vertical fit: antenna height + top edge clearance must fit on board
    if ifa_h + te > hsub:
        errors.append(f"Antenna height exceeds board height: ifa_h + ifa_te = {ifa_h+te:.3g} > board_hsub={hsub:.3g}.")

    # Feed stub inside board
    if e1 + fp + wf > wsub:
        errors.append(f"Feed stub exceeds board width: ifa_e + ifa_fp + ifa_wf = {e1+fp+wf:.3g} > board_wsub={wsub:.3g}.")

    # Manufacturing sanity
    if via >= min(w1, w2, wf):
        warnings.append(f"via_size={via*1e3:.2f} mm is not smaller than trace widths (w1/w2/wf).")

    # --- Your specific meander rules ---

    # 1) "mifa_meander < 2*w2 will be too small"
    if mlen < 2*w2:
        errors.append(f"mifa_meander={mlen*1e3:.2f} mm is < 2*w2={2*w2*1e3:.2f} mm (too small for a useful meander).")

    # 2) "if mifa_meander_edge_distance or mifa_tipdistance is larger than ifa_h-w2 it won't meander"
    vertical_room = ifa_h - w2
    derived["vertical_room_for_meander"] = vertical_room
    if medge > vertical_room:
        errors.append(f"mifa_meander_edge_distance={medge*1e3:.2f} mm > ifa_h - w2={vertical_room*1e3:.2f} mm.")
    if mtip > vertical_room:
        errors.append(f"mifa_tipdistance={mtip*1e3:.2f} mm > ifa_h - w2={vertical_room*1e3:.2f} mm.")

    # 3) "Meander is grown backwards from the tip; last meander should be at least w2 from feedpoint+wf"
    # Horizontal positions:
    x_tip      = e1 + ifa_l                  # rightmost end of radiator
    x_feed_end = e1 + fp + wf                # right edge of feed stub region
    needed_gap = w2                          # clearance
    available_backspace = (x_tip - (x_feed_end + needed_gap))  # room to place meanders backward
    derived.update({
        "x_tip": x_tip,
        "x_feed_end": x_feed_end,
        "required_last_meander_clearance": needed_gap,
        "available_backspace": available_backspace
    })

    if available_backspace <= 0:
        errors.append(
            f"No room for meanders: tip at {x_tip*1e3:.2f} mm, feed_end+w2 at {(x_feed_end+needed_gap)*1e3:.2f} mm "
            f"(available_backspace={available_backspace*1e3:.2f} mm)."
        )
        n_max = 0
    else:
        # Conservative estimate: each meander step costs ~ mifa_meander in X.
        n_max = int(available_backspace // mlen)

    derived["max_meanders_by_length"] = n_max
    if n_max <= 0:
        errors.append("Backwards-grown meander cannot fit even a single segment with the required clearance to the feed area.")

    # Vertical packing estimate (very rough): if meanders alternate up/down with edge clearance (medge) and a top/bottom usable band ~ (ifa_h - w2)
    # You can refine this when your exact geometry is fixed.
    # Here we just assert there is at least some vertical room beyond the edge/tip distances.
    if vertical_room <= max(medge, mtip):
        errors.append("Vertical room for meander paths is exhausted by edge/tip distances.")

    # Nice-to-have: guard tiny copper features
    min_trace = min(w1, w2, wf)
    if min_trace < 0.15e-3:
        warnings.append(f"Very fine copper width detected ({min_trace*1e3:.2f} mm). Check fab capabilities.")
        
    max_num_meanders = (wsub-e1-e2-fp-wf-w2) //(mlen)
    max_num_meanders = max_num_meanders-max_num_meanders%2  # make even
    
    
    derived["estimated_number_of_meanders_fit"] = int(max_num_meanders)
    
    ant_stub = wsub-e1-e2-max_num_meanders*(mlen)-w2
    
    single_meander_length = ifa_h-medge
    tip_length = ifa_h - mtip
    max_length = (max_num_meanders*(mlen+single_meander_length-w2))+ tip_length + ant_stub
    print(f"mlen: {mlen}, single_meander: {single_meander_length}, tip_length: {tip_length}, ant_stub: {ant_stub}")
    derived["estimated_max_antenna_length_with_meanders"] = max_length
    return (errors, warnings, derived)

parameters = {
  "ifa_h": 0.012,
  "ifa_l": 0.013,
  "ifa_w1": 0.001,
  "ifa_w2": 0.0005,
  "ifa_wf": 0.001,
  "ifa_fp": 0.0005,
  "ifa_e": 0.0005,
  "ifa_e2": 0.0005,
  "ifa_te": 0.0005,
  "via_size": 0.0005,
  "board_wsub": 0.012,
  "board_hsub": 0.020,
  "board_th": 0.0015,
  "mifa_meander": 0.001,
  "mifa_tipdistance": 0.003,
  "mifa_meander_edge_distance": 0.002,
  "f1": 700000000.0,
  "f0": 800000000.0,
  "f2": 900000000.0,
  "freq_points": 3,
  "mesh_boundry_size_divisor": 0.5,
  "mesh_wavelength_fraction": 0.5,
  "lambda_scale": 0.5
}
if __name__ == "__main__":
    # --- Example usage ---
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
        
    parameters["ifa_l"]=drv['estimated_max_antenna_length_with_meanders']
    
    model, S11, freq_dense,ff1, ff2, ff3d = build_mifa(parameters,
                                                   view_mesh=False, view_model=True,run_simulation=False,compute_farfield=False,
                                                   loglevel="INFO",solver=em.EMSolver.PARDISO)