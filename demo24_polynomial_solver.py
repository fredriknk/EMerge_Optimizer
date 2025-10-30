import numpy as np
import csv
import emerge as em
from ifalib import build_mifa, get_resonant_frequency
from optimize_lib import _fmt_params_singleline_raw

Z0 = 50.0

# ---------- utils ----------
def s_to_zin(S11): return Z0 * (1.0 + S11) / (1.0 - S11)

def eval_impedance_features(parameters):
    model, S11, freq, *_ = build_mifa(parameters,
        view_mesh=False, view_model=False, run_simulation=True,
        compute_farfield=False, loglevel="ERROR", solver=em.EMSolver.CUDSS)
    if S11 is None:
        return None
    f0 = parameters['f0']
    fr = float(get_resonant_frequency(S11=S11, freq_dense=freq))
    S  = np.interp(f0, freq, S11.real) + 1j*np.interp(f0, freq, S11.imag)
    Zin0 = s_to_zin(S)
    R0, X0 = float(np.real(Zin0)), float(np.imag(Zin0))
    return dict(fr=fr, R0=R0, X0=X0, freq=freq, S11=S11)

def _score(feat, f0):
    s_fr = (feat['fr'] - f0)**2 / (0.02*f0)**2
    s_R  = ((feat['R0'] - Z0)/Z0)**2
    s_X  = (feat['X0']/Z0)**2
    return s_fr + 0.5*s_R + 0.7*s_X

# ---------- quadratic surrogate (ridge) ----------
class QuadRidge:
    """
    y(x) ≈ c + a·x + x^T B x   (B symmetric).
    Fit via linear regression on features [1, x_i, x_i x_j (i<=j)] with L2 ridge.
    Also provides analytic gradient: ∇y = a + (B + B^T) x  = a + 2 B x (since we keep B symmetric).
    """
    def __init__(self, dim, ridge=1e-6, x_mean=None, x_scale=None):
        self.d = dim
        self.ridge = ridge
        self.x_mean = np.zeros(dim) if x_mean is None else np.array(x_mean, float)
        self.x_scale = np.ones(dim) if x_scale is None else np.array(x_scale, float)
        self.coef = None  # packed [c, a(=d), upper-tri(B)]
        # precompute index map for speed
        self.lin_idx = list(range(1, 1+self.d))
        self.quad_pairs = []
        for i in range(self.d):
            for j in range(i, self.d):
                self.quad_pairs.append((i, j))

    def _standardize(self, X):
        return (X - self.x_mean) / self.x_scale

    def _features(self, Xs):
        # Xs: (n, d) standardized
        n = Xs.shape[0]
        m = 1 + self.d + len(self.quad_pairs)
        Phi = np.empty((n, m), dtype=float)
        Phi[:,0] = 1.0
        Phi[:,1:1+self.d] = Xs
        k = 1 + self.d
        for (i,j) in self.quad_pairs:
            Phi[:,k] = Xs[:,i] * Xs[:,j]
            k += 1
        return Phi

    def fit(self, X, y):
        X = np.atleast_2d(X).astype(float)
        y = np.asarray(y, float).reshape(-1)
        Xs = self._standardize(X)
        Phi = self._features(Xs)
        # ridge: (Phi^T Phi + λI) w = Phi^T y
        PtP = Phi.T @ Phi
        lamI = self.ridge * np.eye(PtP.shape[0])
        w = np.linalg.solve(PtP + lamI, Phi.T @ y)
        self.coef = w
        return self

    def predict(self, X):
        X = np.atleast_2d(X).astype(float)
        Phi = self._features(self._standardize(X))
        return (Phi @ self.coef).reshape(-1)

    def grad(self, x):
        """Analytic gradient at single x (unstandardized). Returns d/dx in original units."""
        x = np.asarray(x, float).reshape(-1)
        xs = self._standardize(x)
        # unpack
        w = self.coef
        a = w[1:1+self.d].copy()          # coefficients for xs
        B = np.zeros((self.d, self.d))    # upper-tri from quad terms
        k = 1 + self.d
        for (i,j) in self.quad_pairs:
            B[i,j] = w[k]
            if i != j:
                B[j,i] = w[k]            # mirror (symmetric)
            k += 1
        # gradient wrt xs: a + (B + B^T) xs = a + 2B xs (symmetric)
        g_xs = a + B @ xs + B.T @ xs
        # chain rule: xs = (x - mean)/scale  => d/dx = g_xs / scale
        return g_xs / self.x_scale

# ---------- optimizer using polynomial surrogate ----------
def optimize_mifa_poly(
    initial_params: dict,
    bounds: dict,
    keys_to_tune: list,
    *,
    max_iters=60,
    init_samples=None,           # default: 2d + 4
    per_iter_samples=None,       # default: d//2 + 1
    trust_frac=0.10,
    ridge_model=1e-4,
    ridge_solve=1e-6,
    wR=0.8, wX=1.0,
    x_phase_thresh_ohm=5.0,
    verbose=1,
    csv_log_path=None,
    random_seed=0
):
    rng = np.random.default_rng(random_seed)
    p0 = dict(initial_params)
    order = list(keys_to_tune)
    d = len(order)

    lo = np.array([bounds[k][0] for k in order], float)
    hi = np.array([bounds[k][1] for k in order], float)
    span = hi - lo
    mid  = (hi + lo) / 2.0

    # --- represent params in normalized coords: x_n in [-0.5, +0.5] range roughly
    def to_norm(x_abs):   return (x_abs - mid) / np.where(span!=0, span, 1.0)
    def from_norm(x_norm): return np.clip(mid + x_norm*span, lo, hi)

    x_abs = np.array([p0[k] for k in order], float)
    x = to_norm(x_abs)                         # normalized working point
    trust = trust_frac * np.ones(d)           # trust in *normalized* units

    if init_samples is None: init_samples = 2*d + 4
    if per_iter_samples is None: per_iter_samples = max(d//2 + 1, 1)

    # data containers in normalized space
    Xd = []
    Yr_fr, Yr_R, Yr_X = [], [], []

    def pack_params_from_norm(xn):
        pd = dict(p0)
        for k, v in zip(order, from_norm(xn)):
            pd[k] = float(v)
        return pd

    def eval_and_store(xn):
        pd = pack_params_from_norm(xn)
        feat = eval_impedance_features(pd)
        if feat is None:
            return None
        Xd.append(xn.copy())
        Yr_fr.append(feat['fr'])
        Yr_R.append(feat['R0'])
        Yr_X.append(feat['X0'])
        return feat

    # seed center + random around it
    feat0 = eval_and_store(x)
    if feat0 is None:
        raise RuntimeError("Initial simulation failed.")
    f0 = p0['f0']

    for _ in range(init_samples):
        u = rng.uniform(-1, 1, size=d)
        cand = np.clip(x + u*trust, -0.6, +0.6)     # keep within bounds (normalized)
        eval_and_store(cand)

    # standardization for surrogate (on normalized X)
    Xmat = np.array(Xd)
    x_mean = np.mean(Xmat, axis=0)
    x_std  = np.std(Xmat, axis=0) + 1e-12

    class QuadRidge:
        def __init__(self, dim, ridge=1e-4, x_mean=None, x_std=None):
            self.d = dim; self.ridge = ridge
            self.x_mean = np.zeros(dim) if x_mean is None else np.array(x_mean, float)
            self.x_std  = np.ones(dim)  if x_std  is None else np.array(x_std,  float)
            self.coef = None
            self.quad_pairs = [(i,j) for i in range(dim) for j in range(i,dim)]

        def _std(self, X):
            return (X - self.x_mean) / self.x_std

        def _Phi(self, Xs):
            n = Xs.shape[0]
            m = 1 + self.d + len(self.quad_pairs)
            Phi = np.empty((n, m), float)
            Phi[:,0] = 1.0
            Phi[:,1:1+self.d] = Xs
            k = 1 + self.d
            for (i,j) in self.quad_pairs:
                Phi[:,k] = Xs[:,i]*Xs[:,j]; k += 1
            return Phi

        def fit(self, X, y):
            Xs = self._std(np.atleast_2d(X).astype(float))
            Phi = self._Phi(Xs)
            PtP = Phi.T @ Phi
            w = np.linalg.solve(PtP + self.ridge*np.eye(PtP.shape[0]),
                                Phi.T @ np.asarray(y, float).reshape(-1))
            self.coef = w
            return self

        def predict(self, X):
            Xs = self._std(np.atleast_2d(X).astype(float))
            return (self._Phi(Xs) @ self.coef).reshape(-1)

        def grad(self, x):
            x  = np.asarray(x, float).reshape(-1)
            xs = (x - self.x_mean) / self.x_std
            w = self.coef
            a = w[1:1+self.d].copy()
            # build symmetric B
            B = np.zeros((self.d, self.d))
            k = 1 + self.d
            for (i,j) in self.quad_pairs:
                B[i,j] = w[k]
                if i != j: B[j,i] = w[k]
                k += 1
            g_xs = a + (B + B.T) @ xs
            return g_xs / self.x_std   # chain rule

    # CSV
    csv_fh = None
    csv_writer = None
    if csv_log_path:
        csv_fh = open(csv_log_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["iter","fr_GHz","R_ohm","X_ohm","score", *[f"p:{k}" for k in order]])

    # --- scales to fix conditioning in the solve (key change) ---
    fr_scale = 10e6   # 10 MHz
    R_scale  = 10.0
    X_scale  = 10.0

    for it in range(1, max_iters+1):
        # fit surrogates in normalized param space
        fr_m = QuadRidge(d, ridge=ridge_model, x_mean=x_mean, x_std=x_std).fit(Xd, Yr_fr)
        R_m  = QuadRidge(d, ridge=ridge_model, x_mean=x_mean, x_std=x_std).fit(Xd, Yr_R)
        X_m  = QuadRidge(d, ridge=ridge_model, x_mean=x_mean, x_std=x_std).fit(Xd, Yr_X)

        # current true features
        feat_cur = eval_impedance_features(pack_params_from_norm(x))
        if feat_cur is None:
            trust *= 0.6
            continue

        fr, R0, X0 = feat_cur['fr'], feat_cur['R0'], feat_cur['X0']
        err_fr, err_R, err_X = (f0 - fr), (Z0 - R0), (-X0)

        # phase weights
        if abs(X0) > x_phase_thresh_ohm:
            wR_eff, wX_eff, phase = 0.3*wR, 1.4*wX, 1
        else:
            wR_eff, wX_eff, phase = 1.2*wR, 0.8*wX, 2

        if verbose:
            print(f"[it {it}] fr={fr/1e9:.4f} GHz  @f0: R={R0:.2f}Ω X={X0:.2f}Ω  "
                  f"targets: Δfr={err_fr/1e6:.2f} MHz, ΔR={err_R:.2f}, ΔX={err_X:.2f}  (phase {phase})")

        # surrogate Jacobian (in normalized param space)
        J_fr = fr_m.grad(x)
        J_R  =  R_m.grad(x)
        J_X  =  X_m.grad(x)

        # guard: if all rows are tiny, add exploration and continue
        row_norms = np.array([np.linalg.norm(J_fr), np.linalg.norm(J_R), np.linalg.norm(J_X)])
        if np.all(row_norms < 1e-10):
            trust = np.minimum(trust*1.4, 0.6)
            for _ in range(max(per_iter_samples, d)):
                u = rng.uniform(-1, 1, size=d)
                eval_and_store(np.clip(x + u*trust, -0.6, +0.6))
            Xmat = np.array(Xd); x_mean = Xmat.mean(0); x_std = Xmat.std(0)+1e-12
            if verbose: print("  Jacobian tiny → exploring & expanding dataset.\n")
            continue

        # --- scaled system (key change) ---
        A = np.vstack([J_fr/fr_scale, wR_eff*J_R/R_scale, wX_eff*J_X/X_scale])
        b = np.array([err_fr/fr_scale, wR_eff*err_R/R_scale, wX_eff*err_X/X_scale], float)

        H = A.T @ A + ridge_solve*np.eye(d)
        g = A.T @ b
        try:
            dx = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dx = np.zeros(d)

        # trust clip (normalized units)
        dx = np.clip(dx, -trust, trust)

        # --- surrogate line search (backtracking) ---
        def surrogate_score(xn):
            fr_p = fr_m.predict([xn])[0]; R_p = R_m.predict([xn])[0]; X_p = X_m.predict([xn])[0]
            return ((fr_p - f0)**2 / (0.02*f0)**2) + 0.5*((R_p - Z0)/Z0)**2 + 0.7*(X_p/Z0)**2

        sc_here = surrogate_score(x)
        alpha = 1.0
        for _ in range(6):
            x_try = np.clip(x + alpha*dx, -0.6, +0.6)
            if surrogate_score(x_try) < sc_here:
                break
            alpha *= 0.5
        dx *= alpha
        x_new = np.clip(x + dx, -0.6, +0.6)

        # log predicted changes (unscaled)
        pred_fr = fr_m.predict([x_new])[0] - fr_m.predict([x])[0]
        pred_R  =  R_m.predict([x_new])[0] -  R_m.predict([x])[0]
        pred_X  =  X_m.predict([x_new])[0] -  X_m.predict([x])[0]
        if verbose:
            print(f"  Predicted changes: Δfr={pred_fr:+.3e}  ΔR={pred_R:+.3e}  ΔX={pred_X:+.3e}")
            print("  Step (trust-clipped):")
            for k, dki, tr in zip(order, dx*span, trust*span):  # report in meters
                print(f"    {k:<28} Δp={dki:+.3e}  (trust={tr:.3e})")

        # evaluate true candidate
        feat_new = eval_and_store(x_new)
        if feat_new is None:
            trust *= 0.6
            if verbose: print("  -> candidate failed; shrinking trust.\n")
            continue

        sc_old = _score(feat_cur, f0)
        sc_new = _score(feat_new, f0)
        accept = sc_new < sc_old

        if verbose:
            print(f"  Score: old={sc_old:.4f} new={sc_new:.4f}  ({'accepted' if accept else 'rejected'})")

        if accept:
            x = x_new
            trust = np.minimum(trust*1.3, 0.6)
            if verbose: print("  -> accepted; expanding trust.\n")
        else:
            trust *= 0.7
            if verbose: print("  -> rejected; shrinking trust.\n")

        # exploration around new center
        for _ in range(per_iter_samples):
            u = rng.uniform(-1, 1, size=d)
            eval_and_store(np.clip(x + u*trust, -0.6, +0.6))

        # refresh surrogate standardization
        Xmat = np.array(Xd)
        x_mean = Xmat.mean(0); x_std = Xmat.std(0) + 1e-12

        if csv_writer:
            row = [it, feat_new['fr']/1e9, feat_new['R0'], feat_new['X0'], sc_new,
                   *from_norm(x)]
            csv_writer.writerow(row)

    if csv_fh: csv_fh.close()

    # return best-by-score from dataset
    best_idx = None; best_score = np.inf
    for xn, frv, Rv, Xv in zip(Xd, Yr_fr, Yr_R, Yr_X):
        sc = ((frv - f0)**2 / (0.02*f0)**2) + 0.5*((Rv - Z0)/Z0)**2 + 0.7*(Xv/Z0)**2
        if sc < best_score:
            best_score = sc; best_idx = xn
    out = dict(p0)
    for k, v in zip(order, from_norm(best_idx)):
        out[k] = float(v)
    return out


parameters = { 'ifa_h': 0.006,
    'ifa_l': 0.027-0.00075,
    'ifa_w1': 0.0015,
    'ifa_w2': 0.0005,
    'ifa_wf': 0.0005,
    'ifa_fp': 0.0025,
    'ifa_e': 0.0005, 'ifa_e2': 0.0005, 'ifa_te': 0.0005,
    'via_size': 0.0003, 'board_wsub': 0.014, 'board_hsub': 0.025, 'board_th': 0.0015,
    'mifa_meander': 0.0015, 'mifa_meander_edge_distance': 0.0005,
    'f1': 2.3e+09, 'f0': 2.45e+09, 'f2': 2.6e+09, 'freq_points': 5,
    'mesh_boundry_size_divisor': 0.33, 'mesh_wavelength_fraction': 0.2, 'lambda_scale': 0.7 }

keys = ['ifa_l','ifa_fp','ifa_w1','ifa_w2','ifa_wf']
bounds = {k: (0.8*parameters[k], 1.2*parameters[k]) for k in keys}
for k,v in parameters.items():
    bounds.setdefault(k, (v, v))

best = optimize_mifa_poly(
    initial_params=parameters,
    bounds=bounds,
    keys_to_tune=keys,
    max_iters=80,
    init_samples=None,        # defaults: 2d+4
    per_iter_samples=None,    # defaults: d//2+1
    trust_frac=0.02,          # start modest; surrogate will expand
    ridge_model=1e-4,         # surrogate ridge
    ridge_solve=1e-6,         # step solve ridge
    wR=0.8, wX=1.0,
    x_phase_thresh_ohm=5.0,
    verbose=2,
    random_seed=42
)

print("Best params:", _fmt_params_singleline_raw(best))
