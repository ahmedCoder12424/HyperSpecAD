"""
gods_wrapper.py  —  run GODS and KGODS from the merlresearch/GODS repo
                    as subprocesses, returning sklearn-style pred arrays.

Setup (one-time):
    git clone https://github.com/merlresearch/GODS /path/to/GODS
    pip install pymanopt==0.2.4 autograd
    # KGODS only — copy the GeneralizedStiefel manifold patch:
    PYMANOPT_MANIFOLDS=$(python -c \
        "import pymanopt, os; print(os.path.dirname(pymanopt.__file__))")/manifolds
    cp /path/to/GODS/pymanopt-patch/generalizedstiefel.py $PYMANOPT_MANIFOLDS/
    cp /path/to/GODS/pymanopt-patch/__init__.py           $PYMANOPT_MANIFOLDS/

Usage:
    Set GODS_REPO_PATH below, or pass gods_repo to run_gods() / run_kgods().
"""

import os
import sys
import pickle
import tempfile
import subprocess
import numpy as np
from pathlib import Path

# ── configure this once ───────────────────────────────────────────────────────
GODS_REPO_PATH = os.environ.get("GODS_REPO", "../GODS")


# ── patched runner scripts (written to tmpdir at call time) ───────────────────
# These are minimal wrappers around the original gods.py / kgods.py logic that
# (a) accept our pre-split pkl files,
# (b) dump per-sample predictions + scores to a second pkl instead of just
#     printing accuracy, so we can reconstruct the full pred array.

_GODS_RUNNER = '''\
# gods_run.py — patched GODS runner; do not edit manually.
import argparse, os, pickle, random
import autograd.numpy as np
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Product, Stiefel, Sphere, Euclidean
from pymanopt.solvers import ConjugateGradient
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser()
parser.add_argument("--embed_path",    type=str)
parser.add_argument("--split_num",     type=int,   default=0)
parser.add_argument("--num_subspaces", type=int,   default=5)
parser.add_argument("--eta",           type=float, default=0.01)
parser.add_argument("--L",             type=float, default=0.1)
parser.add_argument("--max_iter",      type=int,   default=1000)
parser.add_argument("--thresh",        type=float, default=0.0)
parser.add_argument("--unnormalize",   action="store_true")
parser.add_argument("--out_pkl",       type=str,   default="gods_preds.pkl")
args = parser.parse_args()

seed = 42
np.random.seed(seed)

with open(os.path.join(args.embed_path,
          f"data_train{args.split_num}.pkl"), "rb") as f:
    data_tr = pickle.load(f, encoding="latin1")
with open(os.path.join(args.embed_path,
          f"data_test{args.split_num}.pkl"), "rb") as f:
    data_te = pickle.load(f, encoding="latin1")

if not args.unnormalize:
    data_tr = normalize(data_tr, axis=1, norm="l2")
    data_te = normalize(data_te, axis=1, norm="l2")

d = data_tr.shape[1]
k = args.num_subspaces
eta = args.eta
Lambda = args.L
data = data_tr.T

if k > 1:
    manifold = Product((Stiefel(d,k), Stiefel(d,k), Euclidean(1,k), Euclidean(1,k)))
else:
    manifold = Product((Sphere(d), Sphere(d), Euclidean(1,k), Euclidean(1,k)))

def cost(M):
    w1,w2,b1,b2 = np.transpose(M[0]),np.transpose(M[1]),np.transpose(M[2]),np.transpose(M[3])
    ww1 = np.dot(w1,data) + b1*np.ones((data_tr.shape[0],))
    ww2 = np.dot(w2,data) + b2*np.ones((data_tr.shape[0],))
    lower = np.maximum(0, np.add(eta, -np.min(ww1, axis=0)))
    upper = np.maximum(0, np.add(eta,  np.max(ww2, axis=0)))
    return (np.sum(np.square(lower)) + np.sum(np.square(upper))
            + Lambda*(np.sum(np.square(ww1)) + np.sum(np.square(ww2))))

solver  = ConjugateGradient(maxiter=args.max_iter)
problem = Problem(manifold=manifold, cost=cost, verbosity=0)
Xopt    = solver.solve(problem)

w1,w2 = np.transpose(Xopt[0]), np.transpose(Xopt[1])
b1,b2 = np.transpose(Xopt[2]), np.transpose(Xopt[3])

# ── vectorized scoring ────────────────────────────────────────────────────
# w1, w2 : (k, d);  data_te : (n_te, d)
# WW1     : (k, n_te) → transpose to (n_te, k)
# b1, b2  : (k, 1)  — squeeze to (k,) for broadcasting
b1_vec = b1.squeeze()   # (k,)
b2_vec = b2.squeeze()   # (k,)
WW1 = (w1 @ data_te.T).T + b1_vec    # (n_te, k)
WW2 = (w2 @ data_te.T).T + b2_vec    # (n_te, k)
in_band = (WW1.min(axis=1) > args.thresh) & (WW2.max(axis=1) < args.thresh)
all_preds  = np.where(in_band, 1, -1)
all_scores = np.minimum(WW1.min(axis=1), -WW2.max(axis=1))

with open(args.out_pkl, "wb") as f:
    pickle.dump({"pred": all_preds.tolist(), "scores": all_scores.tolist()}, f)
print(f"GODS: saved {len(all_preds)} predictions → {args.out_pkl}")
'''

_KGODS_RUNNER = '''\
# kgods_run.py — patched KGODS runner; do not edit manually.
import argparse, os, pickle, random
import autograd.numpy as np
from pymanopt import Problem
from pymanopt.manifolds import GeneralizedStiefel, Product
from pymanopt.solvers import ConjugateGradient
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import chi2_kernel

parser = argparse.ArgumentParser()
parser.add_argument("--embed_path",    type=str)
parser.add_argument("--split_num",     type=int,   default=0)
parser.add_argument("--num_subspaces", type=int,   default=1)
parser.add_argument("--eta",           type=float, default=0.0001)
parser.add_argument("--sigma",         type=float, default=0.01)
parser.add_argument("--max_iter",      type=int,   default=100)
parser.add_argument("--thresh",        type=float, default=0.0)
parser.add_argument("--unnormalize",   action="store_true")
parser.add_argument("--kernel",        type=str,   default="rbf")
parser.add_argument("--out_pkl",       type=str,   default="kgods_preds.pkl")
args = parser.parse_args()

seed = 42
np.random.seed(seed)

with open(os.path.join(args.embed_path,
          f"data_train{args.split_num}.pkl"), "rb") as f:
    data_tr = pickle.load(f, encoding="latin1")
with open(os.path.join(args.embed_path,
          f"data_test{args.split_num}.pkl"), "rb") as f:
    data_te = pickle.load(f, encoding="latin1")

if not args.unnormalize:
    row_sums = data_tr.sum(1, keepdims=True) + 1e-10
    data_tr  = data_tr / row_sums
    row_sums = data_te.sum(1, keepdims=True) + 1e-10
    data_te  = data_te / row_sums

sigma   = args.sigma
n_tr    = data_tr.shape[0]
k       = args.num_subspaces
eta     = args.eta
one_nxk = np.ones((n_tr, k), dtype="float")
X       = data_tr

def compute_kernel(A, B, same=False):
    if args.kernel == "linear":
        K = np.matmul(A, B.T)
    elif args.kernel == "rbf":
        K = np.exp(-(cdist(A, B, "sqeuclidean")) / (2.0*sigma))
    elif args.kernel == "chisq":
        K = chi2_kernel(A, B, gamma=sigma)
    elif args.kernel == "min":
        K = np.zeros((A.shape[0], B.shape[0]))
        for c in range(A.shape[1]):
            K += np.minimum(A[:,c:c+1], B[:,c].T)
        K /= A.shape[1]
    if same:
        K = (K + K.T)/2.0
        K += np.eye(n_tr)*1e-7
    return K

K = compute_kernel(X, X, same=True)
manifold = Product((GeneralizedStiefel(n_tr,k,K), GeneralizedStiefel(n_tr,k,K)))

def cost(M):
    Y,Z = M[0],M[1]
    Y = Y*Y; Z = Z*Z
    obj = (0.5*np.matmul(Y.T,Y).sum()
           + np.trace(np.matmul(Y.T, np.matmul(K,Z)))
           - eta*np.trace(np.matmul((Y-Z).T, one_nxk)))
    obj += 0.1*np.linalg.norm(Y-Z)**2.0
    return obj

solver  = ConjugateGradient(maxiter=args.max_iter)
problem = Problem(manifold=manifold, cost=cost, verbosity=0)
Xopt    = solver.solve(problem)

Y_opt, Z_opt = Xopt[0]**2, Xopt[1]**2
b1 = (eta - K @ Z_opt).max(0)   # shape (k,)
b2 = (eta + K @ Y_opt).min(0)   # shape (k,)

# ── vectorized scoring over all n_test points at once ────────────────────
# K_test : (n_test, n_tr);  Z_opt/Y_opt : (n_tr, k)
# WW1    : (n_test, k)  — must all be > thresh for "in band"
# WW2    : (n_test, k)  — must all be < thresh for "in band"
CHUNK = 10_000   # process in chunks to cap peak memory
n_te  = len(data_te)
all_preds  = np.ones(n_te, dtype=int)
all_scores = np.zeros(n_te, dtype=float)

for start in range(0, n_te, CHUNK):
    end      = min(start + CHUNK, n_te)
    K_chunk  = compute_kernel(data_te[start:end], X)          # (chunk, n_tr)
    WW1      = K_chunk @ Z_opt + b1                           # (chunk, k)
    WW2      = -(K_chunk @ Y_opt) + b2                        # (chunk, k)
    in_band  = (WW1.min(axis=1) > args.thresh) & (WW2.max(axis=1) < args.thresh)
    all_preds[start:end]  = np.where(in_band, 1, -1)
    all_scores[start:end] = np.minimum(WW1.min(axis=1), -WW2.max(axis=1))

with open(args.out_pkl, "wb") as f:
    pickle.dump({"pred": all_preds.tolist(), "scores": all_scores.tolist()}, f)
print(f"KGODS: saved {n_te} predictions → {args.out_pkl}")
'''


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_split_pkls(X_train, X_test, tmpdir, split_num=0):
    """Serialise train/test matrices as the format gods.py/kgods.py expect."""
    tr_path = Path(tmpdir) / f"data_train{split_num}.pkl"
    te_path = Path(tmpdir) / f"data_test{split_num}.pkl"
    with open(tr_path, "wb") as f:
        pickle.dump(X_train.astype(np.float32), f)
    with open(te_path, "wb") as f:
        pickle.dump(X_test.astype(np.float32), f)


def _run_script(script_src, script_name, extra_args, tmpdir, timeout=3600):
    """
    Write script_src to tmpdir/<script_name>, run it as a subprocess,
    stream stdout/stderr live, return the output pkl path.
    """
    script_path = Path(tmpdir) / script_name
    out_pkl     = Path(tmpdir) / script_name.replace(".py", "_preds.pkl")

    script_path.write_text(script_src)

    cmd = [sys.executable, str(script_path),
           "--embed_path", str(tmpdir),
           "--out_pkl",    str(out_pkl),
           "--split_num",  "0"] + extra_args

    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, timeout=timeout)

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} exited with code {result.returncode}."
        )
    if not out_pkl.exists():
        raise RuntimeError(f"{script_name} finished but {out_pkl} was not created.")

    return out_pkl


def _load_preds(out_pkl, n_total):
    """Load pred/scores from the output pkl and validate length."""
    with open(out_pkl, "rb") as f:
        result = pickle.load(f)
    pred   = np.array(result["pred"],   dtype=int)
    scores = np.array(result["scores"], dtype=float)
    if len(pred) != n_total:
        raise RuntimeError(
            f"Expected {n_total} predictions, got {len(pred)}. "
            "Check that X_test passed to the script has all n rows."
        )
    return pred, scores


# ── public API ────────────────────────────────────────────────────────────────

def run_gods(X, contamination=0.05,
             num_subspaces=5, eta=0.01, lam=0.1,
             max_iter=1000, normalize=False,
             subsample=20_000, random_state=42,
             gods_repo=None, timeout=3600):
    """
    Run GODS via subprocess using the merlresearch/GODS repo scripts.

    X          : (n, d) float array — already preprocessed (log1p, no L2 norm).
    subsample  : rows drawn for training. All n rows are scored as test.
    gods_repo  : path to the cloned GODS repo (overrides GODS_REPO_PATH).
    normalize  : pass --unnormalize flag when False (our data must not be
                 L2-normalised — it erases the intensity gap).
    """
    rng    = np.random.default_rng(random_state)
    n      = X.shape[0]
    n_tr   = min(subsample, n)
    idx_tr = rng.choice(n, size=n_tr, replace=False)
    X_tr   = X[idx_tr]
    X_te   = X          # score all n points

    print(f"    GODS: training on {n_tr:,} / {n:,} samples")

    extra = [
        "--num_subspaces", str(num_subspaces),
        "--eta",           str(eta),
        "--L",             str(lam),
        "--max_iter",      str(max_iter),
    ]
    if not normalize:
        extra.append("--unnormalize")

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_split_pkls(X_tr, X_te, tmpdir)
        out_pkl = _run_script(_GODS_RUNNER, "gods_run.py", extra, tmpdir,
                              timeout=timeout)
        pred, scores = _load_preds(out_pkl, n)

    return pred, scores


def run_kgods(X, contamination=0.05,
              num_subspaces=1, eta=0.0001,
              kernel="rbf", sigma=0.01,
              max_iter=100, normalize=False,
              subsample=5_000, random_state=42,
              gods_repo=None, timeout=3600):
    """
    Run KGODS via subprocess using the merlresearch/GODS repo scripts.

    subsample : training set size. Keep ≤ 8,000 — builds an n_tr×n_tr
                kernel matrix (~512MB at float64 for 8K points).
    """
    rng    = np.random.default_rng(random_state)
    n      = X.shape[0]
    n_tr   = min(subsample, n)
    idx_tr = rng.choice(n, size=n_tr, replace=False)
    X_tr   = X[idx_tr]
    X_te   = X

    print(f"    KGODS: training on {n_tr:,} / {n:,} samples "
          f"(kernel={kernel}, sigma={sigma})")

    extra = [
        "--num_subspaces", str(num_subspaces),
        "--eta",           str(eta),
        "--sigma",         str(sigma),
        "--kernel",        kernel,
        "--max_iter",      str(max_iter),
    ]
    if not normalize:
        extra.append("--unnormalize")

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_split_pkls(X_tr, X_te, tmpdir)
        out_pkl = _run_script(_KGODS_RUNNER, "kgods_run.py", extra, tmpdir,
                              timeout=timeout)
        pred, scores = _load_preds(out_pkl, n)

    return pred, scores