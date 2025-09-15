import numpy as np

rng = np.random.default_rng(0)

def to_pm1(P01):
    """0/1 -> ±1"""
    return (2*P01 - 1).astype(float)

def hebb_symmetric(Xpm1):
    """W_sym = (1/N) sum_p x^p x^{pT}, with zero diagonal."""
    N = Xpm1.shape[1]
    W = (Xpm1.T @ Xpm1) / N
    np.fill_diagonal(W, 0.0)
    return W

def temporal_asymmetric(Xpm1, loop=True):
    """W_asym = (1/N) sum_p x^{p+1} x^{pT} (wrap if loop)."""
    N = Xpm1.shape[1]
    if loop:
        X_next = np.roll(Xpm1, -1, axis=0)   # p=P -> 1
        W = (X_next.T @ Xpm1) / N
    else:
        if Xpm1.shape[0] < 2:
            W = np.zeros((N, N))
        else:
            X_now  = Xpm1[:-1]
            X_next = Xpm1[1:]
            W = (X_next.T @ X_now) / N
    np.fill_diagonal(W, 0.0)
    return W

def build_W(Xpm1, alpha=0.2, loop=True):
    W_sym  = hebb_symmetric(Xpm1)
    W_asym = temporal_asymmetric(Xpm1, loop=loop)
    return (1 - alpha) * W_sym + alpha * W_asym

def sign(x):
    # tie-break to +1 on zero; you can randomize if you prefer
    y = np.where(x >= 0, 1.0, -1.0)
    return y

def run_sync(W, s0, steps=10):
    s = s0.copy()
    traj = [s.copy()]
    for _ in range(steps):
        s = sign(W @ s)
        traj.append(s.copy())
    return np.array(traj)

def run_async(W, s0, steps=10, seed=0):
    rng = np.random.default_rng(seed)
    s = s0.copy()
    traj = [s.copy()]
    N = len(s)
    for _ in range(steps):
        order = rng.permutation(N)
        for i in order:
            h_i = W[i] @ s
            s[i] = 1.0 if h_i >= 0 else -1.0
        traj.append(s.copy())
    return np.array(traj)

# --- Demo: make a simple 4-pattern sequence in 0/1, then convert to ±1.
P, N = 4, 24
P01 = rng.integers(0, 2, size=(P, N))       # random binary patterns (rows)
X   = to_pm1(P01)                            # convert to ±1

W = build_W(X, alpha=0.25, loop=True)       # try alpha in [0.1, 0.35]

# Start near the first pattern with noise
noise = rng.choice([-1.0, 1.0], size=N, p=[0.1, 0.9])
s0 = X[0] * noise

traj = run_sync(W, s0, steps=10)             # or run_async(W, s0, steps=10)

# Inspect how well each step matches the intended pattern index
def overlap(a, b):        # normalized correlation in ±1 space ∈ [-1,1]
    return (a @ b) / len(a)

for t in range(traj.shape[0]):
    overlaps = [overlap(traj[t], X[p]) for p in range(P)]
    best = int(np.argmax(overlaps))
    print(f"t={t:2d}  best match={best}  overlaps={np.round(overlaps,2)}")