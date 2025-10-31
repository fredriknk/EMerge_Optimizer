"""
Refactored MIFA / Dual‑Freq MIFA builders for Emerge

Key goals:
- Single source of truth for geometry, meshing, and BCs
- Clean parameter handling (merge + defaults), optional validation
- Reusable helpers: assemble geometry, mesh, solve, postproc
- Deterministic far‑field at p['f0'] (not hardcoded)

Author: ChatGPT, Fredrik
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple, Optional, List
import numpy as np
import emerge as em
from dataclasses import asdict
from typing import List, Dict
import re
import ast

try:
    # Optional: your validator
    from ifa_validation import validate_ifa_params as _validate
except Exception:
    _validate = None

mm = 1e-3

# -----------------------------
# Parameter container
# -----------------------------
@dataclass
class AntennaParams:
    # Board / environment
    board_wsub: float = 21.0 * mm
    board_hsub: float = 90.0 * mm
    board_th: float = 1.5 * mm

    # Electrical setup
    f0: float = 2.45e9
    f1: float = 2.0e9
    f2: float = 3.6e9
    freq_points: int = 10
    lambda_scale: float = 1.0

    # Radiator geometry (top-level IFA)
    ifa_l: float = 20.0 * mm
    ifa_h: float = 8.0 * mm
    ifa_w1: float = 0.6 * mm
    ifa_w2: float = 1.0 * mm
    ifa_wf: float = 0.45 * mm
    ifa_fp: float = 3.8 * mm
    ifa_e: float = 0.5 * mm
    ifa_e2: float = 0.6 * mm
    ifa_te: float = 0.5 * mm

    # Meander / tip controls
    mifa_meander: float = 2.3 * mm  # step in x (incl. gap allowance)
    mifa_low_dist: float = 3.5 * mm
    mifa_tipdistance: Optional[float] = None  # defaults to meander_edge_distance

    # Feed / via
    via_size: float = 0.5 * mm
    
    #stub
    shunt: bool = True

    # Meshing
    mesh_boundary_size_divisor: float = 0.33
    mesh_wavelength_fraction: float = 0.2

    # Material (simple FR‑4 like; override if needed)
    eps_r: float = 3.38

    # Behavior
    validate: bool = True

    def merged_with(self, overrides: Dict) -> "AntennaParams":
        data = asdict(self)
        data.update(overrides or {})
        return AntennaParams(**data)

    @staticmethod
    def from_dict(d: Dict) -> "AntennaParams":
        return AntennaParams(**d)


# -----------------------------
# Geometry builders
# -----------------------------

def _add_box_xy(width: float, height: float, pos, name: str):
    if width <= 0 or height <= 0:
        return None
    return em.geo.XYPlate(width, height, position=pos, name=name)

def add_box_from_start_stop(start, stop, name: str):
        """Convert 'start/stop' corners into an XYPlate and append it."""
        x0, y0, _ = start
        x1, y1, _ = stop
        z = start[2]
        pos = (min(x0, x1), min(y0, y1), z)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        if w <= 0 or h <= 0:
            return None
        plate = em.geo.XYPlate(
            width=w, depth=h, position=pos, name=f"{name}"
        )
        return plate

def _build_mifa_plates(p: AntennaParams, tl) -> List[em.geo.XYPlate]:
    """Pure 2D radiator construction, returns list of XYPlate solids.
    Matches user's ASCII sketch logic with tip + optional meanders.
    """
    plates: List[em.geo.XYPlate] = []

    ifa_l = p.ifa_l
    ifa_h = p.ifa_h
    ifa_w2 = p.ifa_w2
    ifa_fp = p.ifa_fp
    ifa_e = p.ifa_e
    ifa_e2 = p.ifa_e2
    mifa_meander = p.mifa_meander

    mifa_low_dist = p.mifa_low_dist
    mifa_tipdistance = p.mifa_tipdistance or mifa_low_dist

    usable_x = p.board_wsub - ifa_e - ifa_e2

    # Main base run, referenced from left board edge
    # NOTE: origin choice = (−board_w/2, +board_h/2) + fp offset in caller
    start_main = np.array([-ifa_fp + ifa_e, ifa_h - ifa_w2, 0.0]) + tl

    if ifa_l <= usable_x:
        stop_main = start_main + np.array([ifa_l, ifa_w2, 0.0])
        plates.append(_add_box_xy(ifa_l, ifa_w2, start_main, "ifa_main"))
        return [p for p in plates if p is not None]

    # Need tip + meanders
    stop_main = start_main + np.array([usable_x, ifa_w2, 0.0])
    # Do NOT append the base yet; we may extend/attach to it later

    max_len_meander_y = ifa_h - mifa_low_dist - ifa_w2
    max_tip_y = ifa_h - mifa_tipdistance - ifa_w2

    length_diff = ifa_l - usable_x
    tip_anchor = stop_main.copy()

    if length_diff <= max_tip_y:
        # Partial tip only
        tip_h = length_diff + ifa_w2
        tip_start = tip_anchor
        tip_pos = tip_start + np.array([-ifa_w2, -tip_h, 0.0])
        plates.append(_add_box_xy(ifa_w2, tip_h, tip_pos, "ifa_tip_part"))
        plates.append(_add_box_xy(usable_x, ifa_w2, start_main, "ifa_base"))
        return [p for p in plates if p is not None]

    # Full tip
    tip_h = max_tip_y + ifa_w2
    tip_pos = tip_anchor + np.array([-ifa_w2, -tip_h, 0.0])
    plates.append(_add_box_xy(ifa_w2, tip_h, tip_pos, "ifa_tip_full"))
    length_diff -= max_tip_y
    #print(f"max_edgelength_tip={max_edgelength_tip*1e3:.2f} mm used, remaining length_diff={length_diff*1e3:.2f} mm")
    # --- Meanders for remaining length ---
    # We meander down/up in y by fractions of max_length_mifa, stepping in x by (mifa_meander+ifa_w2)
    #print(f"max_length_mifa={max_length_mifa*1e3:.2f} mm")
    if length_diff > 0:
        ldiff_ratio = length_diff / (max_len_meander_y * 2.0)  # each meander adds this much normalized length
        # Continue from tip bottom-left edge
        curr = tip_pos + np.array([0.0, tip_h, 0.0])

        while ldiff_ratio > 0:
            current_meander = min(1.0, ldiff_ratio)
            ldiff_ratio -= current_meander
            #print(f"  adding meander with ratio {current_meander:.3f}, remaining ldiff_ratio={ldiff_ratio:.3f}")
            if current_meander < 0.05*mm:
                break

            # Top horizontal of meander (leftwards)
            seg1_start = curr + np.array([0.0, 0.0, 0.0])
            seg1_stop  = seg1_start + np.array([-mifa_meander-ifa_w2, -ifa_w2, 0.0])
            plates.append(add_box_from_start_stop(seg1_start, seg1_stop, name="meander_top"))

            # Down leg
            seg2_start = seg1_stop + np.array([0.0, ifa_w2, 0.0])
            seg2_stop  = seg2_start + np.array([ifa_w2, -current_meander * max_len_meander_y-ifa_w2, 0.0])
            plates.append(add_box_from_start_stop(seg2_start, seg2_stop, name="meander_down"))

            # Bottom horizontal (rightwards)
            seg3_start = seg2_stop + np.array([0.0, ifa_w2, 0.0])
            seg3_stop  = seg3_start + np.array([-mifa_meander - ifa_w2, -ifa_w2, 0.0])
            plates.append(add_box_from_start_stop(seg3_start, seg3_stop, name="meander_bottom"))

            # Up leg
            seg4_start = seg3_stop + np.array([0, 0, 0.0])
            seg4_stop  = seg4_start + np.array([+ifa_w2, current_meander * max_len_meander_y+ifa_w2, 0.0])
            plates.append(add_box_from_start_stop(seg4_start, seg4_stop, name="meander_up"))

            curr = seg4_stop +  np.array([0.0, 0.0, 0.0])# tail for the next loop

        plates.append(add_box_from_start_stop(start_main, curr + np.array([0, 0, 0.0]), "base_link"))

        # Link base from start_main to latest curr
        base_len = (curr[0] - start_main[0])
        plates.append(_add_box_xy(base_len, ifa_w2, start_main, "ifa_base_link"))
    else:
        plates.append(_add_box_xy(usable_x, ifa_w2, start_main, "ifa_base"))

    return [p for p in plates if p is not None]


def _add_shunt_stub(p: AntennaParams, fp_origin):
    # Short circuit stub from feed towards left by (ifa_fp - ifa_e)
    ifa_stub = p.ifa_fp - p.ifa_e
    pos = fp_origin + np.array([-ifa_stub, -2 * p.via_size, 0.0])
    return em.geo.XYPlate(p.ifa_w1, p.ifa_h + 2 * p.via_size, position=pos, name="ifa_stub")


def _add_via(p: AntennaParams, fp_origin, name="via"):
    ifa_stub = p.ifa_fp - p.ifa_e
    via_cs = em.CoordinateSystem(
        xax=(1, 0, 0), yax=(0, 1, 0), zax=(0, 0, 1),
        origin=fp_origin + np.array([-ifa_stub + p.ifa_w1 / 2, -p.via_size, 0.0])
    )
    return em.geo.Cylinder(p.via_size / 2, -p.board_th, cs=via_cs, name=name)


# -----------------------------
# Meshing / BC / Solve helpers
# -----------------------------

def _make_air_and_dielectric(p: AntennaParams):
    # Dielectric centered in XY, thickness in -Z
    dielectric = em.geo.Box(p.board_wsub, p.board_hsub, p.board_th,
                            position=(-p.board_wsub / 2, -p.board_hsub / 2, -p.board_th))

    # Air box sized from lambda at f2
    lam2 = em.lib.C0 / (p.f2) * p.lambda_scale
    fwd, back = 0.50 * lam2, 0.30 * lam2
    side, top, bot = 0.30 * lam2, 0.30 * lam2, 0.30 * lam2

    airX = p.board_hsub + fwd + back
    airY = p.board_wsub + 2 * side
    airZ = top + bot + p.board_th
    x0, y0, z0 = -side - p.board_wsub / 2, -back - p.board_hsub / 2, -bot - p.board_th / 2
    air = em.geo.Box(airY, airX, airZ, position=(x0, y0, z0)).background()

    return dielectric, air


def _make_port(p: AntennaParams, fp_origin):
    return em.geo.Plate(
        fp_origin + np.array([0, -2 * p.via_size, 0]),
        np.array([p.ifa_wf, 0, 0]),
        np.array([0, 0, -p.board_th]),
    )


def _assign_materials(model: em.Simulation, p: AntennaParams, *, dielectric, metals: List[em.GeometryBase]):
    # Metals = PEC
    for m in metals:
        m.set_material(em.lib.PEC)
    # Board permittivity (color only cosmetic)
    dielectric.material = em.Material(p.eps_r, color="#207020", opacity=0.9)
    model.commit_geometry()


def _mesh_common(model: em.Simulation, p: AntennaParams, *, ifa_union, via_list, port):
    model.mw.set_resolution(p.mesh_wavelength_fraction)
    model.mw.set_frequency_range(p.f1, p.f2, p.freq_points)

    smallest_trace = min(p.ifa_w2, p.ifa_wf, p.ifa_w1)
    smallest_port = min(p.ifa_wf, p.board_th)
    model.mesher.set_boundary_size(ifa_union, smallest_trace * p.mesh_boundary_size_divisor)

    for v in via_list:
        model.mesher.set_boundary_size(v, min(p.via_size, p.board_th) * p.mesh_boundary_size_divisor)

    model.mesher.set_face_size(port, smallest_port * p.mesh_boundary_size_divisor)
    model.mesher.set_algorithm(em.Algorithm3D.HXT)
    model.generate_mesh()


def _solve(model: em.Simulation, p: AntennaParams, *, air, port):
    # BCs
    _ = model.mw.bc.AbsorbingBoundary(air.boundary())
    _ = model.mw.bc.LumpedPort(port, 1, width=p.ifa_wf, height=p.board_th, direction=em.ZAX, Z0=50)

    data = model.mw.run_sweep()
    freq_dense = np.linspace(p.f1, p.f2, 1001)
    S11 = data.scalar.grid.model_S(1, 1, freq_dense)

    return data, S11, freq_dense


# -----------------------------
# Public API
# -----------------------------
_LINK_RE = re.compile(r"\$\{\s*([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*\}")

def _resolve_linked_params(raw_list: List[Dict]) -> List[Dict]:
    """
    Resolve expressions with ${alias.key} placeholders and arithmetic (+ - * /, parentheses).
    Supports chained links and same-alias (p2 -> p2.other_key) references.
    Rejects true self-reference (key -> key).
    """

    # ---------- Safe arithmetic evaluator ----------
    ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

    def _safe_eval_expr(expr: str) -> float:
        tree = ast.parse(expr, mode="eval")
        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Expression): return _eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Num): return float(node.n)  # py<3.8 compat
            if isinstance(node, ast.BinOp) and isinstance(node.op, ALLOWED_BINOPS):
                l = _eval(node.left); r = _eval(node.right)
                if isinstance(node.op, ast.Add):  return l + r
                if isinstance(node.op, ast.Sub):  return l - r
                if isinstance(node.op, ast.Mult): return l * r
                if isinstance(node.op, ast.Div):  return l / r
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ALLOWED_UNARYOPS):
                v = _eval(node.operand)
                if isinstance(node.op, ast.UAdd): return +v
                if isinstance(node.op, ast.USub): return -v
            raise ValueError(f"Unsupported syntax in expression: {expr!r}")
        return _eval(tree)

    if not raw_list:
        return []

    # Effective dicts: base overlaid with each entry
    base = dict(raw_list[0])
    eff: List[Dict[str, Any]] = []
    for i, d in enumerate(raw_list):
        e = dict(base) if i else dict(base)
        if i: e.update(d)
        else: e.update(d)  # keep explicit
        eff.append(e)

    # Alias map: p, p2, p3, ...
    ctx: Dict[str, Dict[str, Any]] = {
        ("p" if i == 0 else f"p{i+1}"): e for i, e in enumerate(eff)
    }

    # Memo: fully resolved values
    memo: Dict[Tuple[str, str], Any] = {}

    def _resolve_value(alias: str, key: str, seen: Set[Tuple[str, str]], current_alias: Optional[str]=None) -> Any:
        """
        Resolve ctx[alias][key], following links recursively until concrete.
        - Uses memo to avoid recomputation.
        - Detects cycles (including key->key).
        - If alias == current_alias and key is already memoized, we reuse it (enables same-alias multi-key refs).
        """
        node = (alias, key)
        if node in memo:
            return memo[node]
        if node in seen:
            raise ValueError(f"Cyclic link detected at {alias}.{key}")

        if alias not in ctx:
            raise ValueError(f"Unknown alias '{alias}'")
        if key not in ctx[alias]:
            raise ValueError(f"Missing key '{key}' in alias '{alias}'")

        seen.add(node)
        raw_val = ctx[alias][key]

        # Expand placeholders if string
        if isinstance(raw_val, str):
            # First pass: substitute all ${a.b} with resolved numbers/strings
            def _sub_one(m: re.Match) -> str:
                a2, k2 = m.group(1), m.group(2)
                # Prevent true self-reference: alias/key equals the node we are resolving
                if a2 == alias and k2 == key:
                    raise ValueError(f"Self-reference detected at {alias}.{key}")
                v2 = _resolve_value(a2, k2, seen, current_alias=alias)  # recurse
                if isinstance(v2, (int, float)): return repr(float(v2))
                if isinstance(v2, bool):          return "1.0" if v2 else "0.0"
                return str(v2)

            expanded = _LINK_RE.sub(_sub_one, raw_val).strip()

            # Try to evaluate as arithmetic; if that fails, try numeric literal; else keep as string
            try:
                resolved = _safe_eval_expr(expanded)
            except Exception:
                try:
                    resolved = float(expanded)
                except Exception:
                    resolved = expanded
        else:
            resolved = raw_val

        memo[node] = resolved
        seen.remove(node)
        return resolved

    # Build fully-resolved dicts per alias
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(raw_list):
        alias = "p" if i == 0 else f"p{i+1}"
        rd: Dict[str, Any] = {}
        for k in d.keys():
            rd[k] = _resolve_value(alias, k, set(), current_alias=alias)
        out.append(rd)

    return out


# -----------------------------
# Public API
# -----------------------------

def _normalize_params_sequence(params_any) -> List[AntennaParams]:
    """Accepts:
      - dict with keys like {"p":..., "p2":..., ...}
      - list/tuple of dict or AntennaParams: [p, p2, ...]
      - single dict (one antenna)
      - single AntennaParams instance
    Returns a list of AntennaParams where entries > 0 inherit missing fields from entry 0.
    Supports links like "${p.ifa_fp}" and "${p2.ifa_h} + 0.0002".
    """
    # Single AntennaParams instance
    if isinstance(params_any, AntennaParams):
        return [params_any]

    # Dict input: either {"p":..., "p2":...} or a single param mapping
    if isinstance(params_any, dict):
        p_keys = [k for k in params_any.keys() if isinstance(k, str) and k.startswith("p")]
        if p_keys:
            # dict form {"p":..., "p2":..., ...}
            keys = sorted(p_keys, key=lambda s: (len(s), s))
            raw_list = [params_any[k] if isinstance(params_any[k], dict) else asdict(params_any[k]) for k in keys]
            # resolve links across aliases with base inheritance
            raw_list_resolved = _resolve_linked_params(raw_list)
            base = AntennaParams.from_dict(raw_list_resolved[0])
            out = [base]
            for d in raw_list_resolved[1:]:
                out.append(base.merged_with(d))
            return out
        else:
            # treat as a single-antenna dict
            return [AntennaParams.from_dict(params_any)]

    # Sequence input: [dict/Params, ...]
    if isinstance(params_any, (list, tuple)):
        if len(params_any) == 0:
            raise ValueError("Empty parameter list")
        raw_list: List[Dict] = []
        for v in params_any:
            if isinstance(v, AntennaParams):
                raw_list.append(asdict(v))
            elif isinstance(v, dict):
                raw_list.append(v)
            else:
                raise ValueError("List entries must be dict or AntennaParams")
        raw_list_resolved = _resolve_linked_params(raw_list)
        base = AntennaParams.from_dict(raw_list_resolved[0])
        out = [base]
        for d in raw_list_resolved[1:]:
            out.append(base.merged_with(d))
        return out

    raise ValueError("params must be dict {'p':..., 'p2':...}, a single dict/Params, or a list of dict/Params")

def build_mifa(
    params_any,
    *,
    model: Optional[em.Simulation] = None,
    view_mesh: bool = False,
    view_model: bool = False,
    run_simulation: bool = True,
    compute_farfield: bool = True,
    loglevel: str = "INFO",
    solver=em.EMSolver.CUDSS,
    ff_freq: Optional[float] = None,
) -> Tuple:
    """Build N IFAs that share the same feed/port location.

    Accepts either {"p":..., "p2":..., ...} or a list [p, p2, p3, ...].
    For entries > 0, missing fields are inherited from the first entry.

    Returns (model, S11, freq_dense, ff1, ff2, ff3d)
    """
    P_list = _normalize_params_sequence(params_any)
    P0 = P_list[0]

    if model is None:
        model = em.Simulation("MultiMIFA", loglevel=loglevel)
        model.set_solver(solver)
        model.check_version("1.1.0")

    dielectric, air = _make_air_and_dielectric(P0)

    # Single shared feed origin based on the base geometry
    fp_origin = np.array([
        -P0.board_wsub / 2 + P0.ifa_fp,
        P0.board_hsub / 2 - P0.ifa_h - P0.ifa_te,
        0.0,
    ])

    # Build all radiators and vias; add feed pad only for the first (shared port)
    ifa_union = None
    vias: List[em.GeometryBase] = []

    for idx, P in enumerate(P_list):
        plates = _build_mifa_plates(P, fp_origin)
        ifa = plates[0]
        for pl in plates[1:]:
            ifa = em.geo.add(ifa, pl)

        if P.shunt:
            shunt_stub = _add_shunt_stub(P, fp_origin)
            ifa = em.geo.add(ifa, shunt_stub)
            
            via = _add_via(P, fp_origin, name=f"via_{idx}")
            vias.append(via)
            
        if idx == 0:
            feed_pad = em.geo.XYPlate(P.ifa_wf, P.ifa_h + 2 * P.via_size, position=fp_origin + np.array([0.0, -2 * P.via_size, 0.0]), name="ifa_feedpad")
            ifa = em.geo.add(ifa, feed_pad)

        if ifa_union is None:
            ifa_union = ifa
        else:
            ifa_union = em.geo.add(ifa_union, ifa)

    # Ground and port from base params
    ground = em.geo.XYPlate(P0.board_wsub, fp_origin[1] + P0.board_hsub / 2, position=(-P0.board_wsub / 2, -P0.board_hsub / 2, -P0.board_th))
    port = _make_port(P0, fp_origin)

    # Materials
    _assign_materials(model, P0, dielectric=dielectric, metals=[ifa_union, *vias, ground])

    # Mesh using P0 strategy but with smallest features across all antennas
    smallest_trace = min(min(P.ifa_w2, P.ifa_wf, P.ifa_w1) for P in P_list)
    smallest_port = min(P0.ifa_wf, P0.board_th)

    model.mw.set_resolution(P0.mesh_wavelength_fraction)
    model.mw.set_frequency_range(P0.f1, P0.f2, P0.freq_points)
    model.mesher.set_boundary_size(ifa_union, smallest_trace * P0.mesh_boundary_size_divisor)

    for v, P in zip(vias, P_list):
        model.mesher.set_boundary_size(v, min(P.via_size, P.board_th) * P0.mesh_boundary_size_divisor)

    model.mesher.set_face_size(port, smallest_port * P0.mesh_boundary_size_divisor)
    model.mesher.set_algorithm(em.Algorithm3D.HXT)
    model.generate_mesh()

    if view_mesh:
        model.view(selections=[port], plot_mesh=True, volume_mesh=False)
    if view_model:
        model.view()

    if not run_simulation:
        return model, None, None, None, None, None

    data, S11, freq_dense = _solve(model, P0, air=air, port=port)

    if not compute_farfield:
        return model, S11, freq_dense, None, None, None

    fff = ff_freq or P0.f0
    ff1 = data.field.find(freq=fff).farfield_2d((0, 0, 1), (1, 0, 0), air.boundary())
    ff2 = data.field.find(freq=fff).farfield_2d((0, 0, 1), (0, 1, 0), air.boundary())
    ff3d = data.field.find(freq=fff).farfield_3d(air.boundary())

    # Display
    model.display.add_object(ifa_union)
    for v in vias:
        model.display.add_object(v)
    model.display.add_object(dielectric)

    return model, S11, freq_dense, ff1, ff2, ff3d

# -----------------------------
# Post‑proc helpers
# -----------------------------

def s11_rl_db(S11: np.ndarray) -> np.ndarray:
    return -20.0 * np.log10(np.abs(S11))


def s11_at_freq_db(S11: np.ndarray, freq_dense: np.ndarray, f: float) -> float:
    # Interpolate complex S11, then give RL in dB
    S = np.interp(f, freq_dense, S11)
    return -20.0 * np.log10(np.abs(S))


def resonant_freq(S11: np.ndarray, freq_dense: np.ndarray) -> float:
    RL = s11_rl_db(S11)
    return float(freq_dense[np.argmax(RL)])


def bandwidth(S11: np.ndarray, freq_dense: np.ndarray, rl_thresh_db: float = -10.0, f0: Optional[float] = None) -> Tuple[float, float]:
    RL = s11_rl_db(S11)
    mask = RL >= (-rl_thresh_db)
    if not mask.any():
        return (np.nan, np.nan)
    idx = np.where(mask)[0]
    f_lo, f_hi = freq_dense[idx[0]], freq_dense[idx[-1]]
    if f0 is not None and (f_lo > f0 or f_hi < f0):
        # ensure span includes f0 if desired
        return (np.nan, np.nan)
    return (float(f_lo), float(f_hi))
