import emerge as em
import numpy as np

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
    ifa_w1      = float(p['ifa_w1'])
    ifa_w2      = float(p['ifa_w2'])
    ifa_wf      = float(p['ifa_wf'])
    ifa_fp      = float(p['ifa_fp'])
    ifa_e      = float(p['ifa_e'])
    ifa_e2      = float(p['ifa_e2'])
    ifa_te      = float(p['ifa_te'])
    via_size     = float(p['via_size'])
    board_wsub    = float(p['board_wsub'])
    board_hsub    = float(p['board_hsub'])
    board_th      = float(p['board_th'])
    mifa_meander    = float(p['mifa_meander'])
    mifa_meander_edge_distance   = float(p['mifa_meander_edge_distance'])
    mifa_meander_tip_distance    = float(p.get('mifa_tipdistance', mifa_meander_edge_distance))
    f1      = float(p['f1'])
    f0      = float(p['f0'])
    f2      = float(p['f2'])
    freq_points   = int(p['freq_points'])
    mesh_boundry_size_divisor    = float(p['mesh_boundry_size_divisor'])
    mesh_wavelength_fraction  = float(p['mesh_wavelength_fraction'])
    lambda_scale  = float(p['lambda_scale'])
    clearance    = float(p.get('clearance', 0.0003))

    errors   = []
    warnings = []
    derived  = {}

    # Basic positivity & size sanity
    for name, val in [
        ("ifa_h",ifa_h),("ifa_l",ifa_l),("ifa_w1",ifa_w1),("ifa_w2",ifa_w2),("ifa_wf",ifa_wf),("ifa_fp",ifa_fp),
        ("ifa_e",ifa_e),("ifa_e2",ifa_e2),("ifa_te",ifa_te),("via_size",via_size),
        ("board_wsub",board_wsub),("board_hsub",board_hsub),("board_th",board_th),
        ("mifa_meander",mifa_meander),("mifa_meander_edge_distance",mifa_meander_edge_distance),("mifa_tipdistance",mifa_meander_edge_distance),
        ("f1",f1),("f0",f0),("f2",f2),("mesh_boundry_size_divisor",mesh_boundry_size_divisor),
        ("mesh_wavelength_fraction",mesh_wavelength_fraction),("lambda_scale",lambda_scale)
    ]:
        if val <= 0:
            errors.append(f"{name} must be > 0 (got {val}).")

    if freq_points < 2:
        errors.append(f"freq_points must be >= 2 (got {freq_points}).")

    # Frequency ordering
    if not (f1 < f0 < f2):
        errors.append(f"Frequency ordering must be f1 < f0 < f2 (got f1={f1}, f0={f0}, f2={f2}).")

    # Mesh sanity bounds
    if not (0.05 <= mesh_wavelength_fraction <= 1.0):
        warnings.append(f"mesh_wavelength_fraction={mesh_wavelength_fraction:.3g} is unusual; typical ~0.1–0.5.")
    if not (0.1 <= mesh_boundry_size_divisor <= 2.0):
        warnings.append(f"mesh_boundry_size_divisor={mesh_boundry_size_divisor:.3g} is unusual; typical ~0.2–1.0.")
    if not (0.25 <= lambda_scale <= 2.0):
        warnings.append(f"lambda_scale={lambda_scale:.3g} outside common 0.5–1.0 range.")

    # Board fit (layout axes per your ASCII):
    # X (width) ~ substrate_width = wsub
    # Y (length) ~ substrate_length = hsub
    # Horizontal extent from left edge ~ e1 + ifa_l
    
    # Feedstub Crash: feed + short_circuit_stub + clearance must fit on board
    if ifa_fp-ifa_e-ifa_w1 < clearance:
        errors.append(f"Feedstub Crash: ifa_e + ifa_w1 - ifa_fp = {ifa_e+ifa_w1 - ifa_fp:.3g} < 0.3.")

    # Vertical fit: antenna height + top edge clearance must fit on board
    if ifa_h + ifa_te > board_hsub:
        errors.append(f"Antenna height exceeds board height: ifa_h + ifa_te = {ifa_h+ifa_te:.3g} > board_hsub={board_hsub:.3g}.")

    # Feed stub inside board
    if ifa_e + ifa_fp + ifa_wf > board_wsub:
        errors.append(f"Feed stub exceeds board width: ifa_e + ifa_fp + ifa_wf = {ifa_e+ifa_fp+ifa_wf:.3g} > board_wsub={board_wsub:.3g}.")

    # Manufacturing sanity
    if via_size > min(ifa_w1, ifa_w2, ifa_wf):
        warnings.append(f"via_size={via_size*1e3:.2f} mm is not smaller than trace widths (w1/w2/wf).")

    # --- Your specific meander rules ---

    # 1) "mifa_meander < 2*w2 will be too small"
    if mifa_meander < 2*ifa_w2 + clearance:
        errors.append(f"mifa_meander={mifa_meander*1e3:.2f} mm is < 2*w2 + clearance={(2*ifa_w2+clearance)*1e3:.4f} mm (too small for a useful meander).")

    # 2) "if mifa_meander_edge_distance or mifa_tipdistance is larger than ifa_h-w2 it won't meander"
    vertical_room = ifa_h - ifa_w2
    derived["vertical_room_for_meander"] = vertical_room
    if mifa_meander_edge_distance >= vertical_room:
        errors.append(f"mifa_meander_edge_distance={mifa_meander_edge_distance*1e3:.2f} mm > ifa_h - w2={vertical_room*1e3:.2f} mm.")
    if mifa_meander_tip_distance >= vertical_room:
        errors.append(f"mifa_tipdistance={mifa_meander_tip_distance*1e3:.2f} mm > ifa_h - w2={vertical_room*1e3:.2f} mm.")

    # 3) "Meander is grown backwards from the tip; last meander should be at least w2 from feedpoint+wf"
    # Horizontal positions:
    x_tip      = ifa_e + ifa_l                  # rightmost end of radiator
    x_feed_end = ifa_e + ifa_fp + ifa_wf                # right edge of feed stub region
    needed_gap = ifa_w2                          # clearance
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
        n_max = int(available_backspace // mifa_meander)

    derived["max_meanders_by_length"] = n_max
    if n_max <= 0:
        errors.append("Backwards-grown meander cannot fit even a single segment with the required clearance to the feed area.")

    # Vertical packing estimate (very rough): if meanders alternate up/down with edge clearance (medge) and a top/bottom usable band ~ (ifa_h - w2)
    # You can refine this when your exact geometry is fixed.
    # Here we just assert there is at least some vertical room beyond the edge/tip distances.
    if vertical_room <= max(mifa_meander_edge_distance, mifa_meander_tip_distance):
        errors.append("Vertical room for meander paths is exhausted by edge/tip distances.")

    # Nice-to-have: guard tiny copper features
    min_trace = min(ifa_w1, ifa_w2, ifa_wf)
    if min_trace < 0.15e-3:
        warnings.append(f"Very fine copper width detected ({min_trace*1e3:.2f} mm). Check fab capabilities.")
        
    max_num_meanders = (board_wsub-ifa_e-ifa_e2-ifa_fp-ifa_wf-ifa_w2) //(mifa_meander)
    max_num_meanders = max_num_meanders-max_num_meanders%2  # make even
    derived["estimated_number_of_meanders_fit"] = int(max_num_meanders)
    ant_stub = board_wsub-ifa_e-ifa_e2-max_num_meanders*(mifa_meander)-ifa_w2
    single_meander_length = ifa_h-mifa_meander_edge_distance
    tip_length = ifa_h - mifa_meander_tip_distance
    max_length = (max_num_meanders*(mifa_meander+single_meander_length-ifa_w2))+ tip_length + ant_stub
    derived["estimated_max_antenna_length_with_meanders"] = max_length
    if max_length < ifa_l:
        errors.append(f"Estimated max antenna length with meanders ({max_length*1e3:.2f} mm) is less than ifa_l ({ifa_l*1e3:.2f} mm).")
    return (errors, warnings, derived)

if __name__ == "__main__":
    #############################################################
    #|------------- board_wsub----- -------------------|
    # _______________________________________________     _ substrate_thickness
    #| A  ifa_e      |----------ifa_l(total length)-| |\  \-gndplane_position 
    #| V____          _______________     __________  | |  \_0 point
    #|               |    ___  ___   |___|  ______  | | |
    #|         ifa_h |   |   ||   |_________|    |  |_|_|_ mifa_meander_edge_distance 
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
        #"mifa_tipdistance": 0.003,
        "mifa_meander_edge_distance": 0.011,
        "f1": 700000000.0,
        "f0": 800000000.0,
        "f2": 900000000.0,
        "freq_points": 3,
        "mesh_boundry_size_divisor": 0.5,
        "mesh_wavelength_fraction": 0.5,
        "lambda_scale": 0.5,
        "clearance": 0.0003,
        }
    
    parameters = {
        "ifa_h": 0.026721961022660216,
        "ifa_l": 0.13516767907510804,
        "ifa_w1": 0.0007747970923122423,
        "ifa_w2": 0.0008121799266081413,
        "ifa_wf": 0.0012112860858232545,
        "ifa_fp": 0.007857536003039053,
        "ifa_e": 0.0005,
        "ifa_e2": 0.0005,
        "ifa_te": 0.0005,
        "via_size": 0.0005,
        "board_wsub": 0.03,
        "board_hsub": 0.11,
        "board_th": 0.0015,
        "mifa_meander": 1.9244e-03,
        "mifa_meander_edge_distance": 0.003,
        "f1": 791000000.0,
        "f0": 826000000.0,
        "f2": 862000000.0,
        "freq_points": 3.0,
        "mesh_boundry_size_divisor": 0.5,
        "mesh_wavelength_fraction": 0.5,
        "lambda_scale": 0.5,
        "clearance": 0.0003,
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