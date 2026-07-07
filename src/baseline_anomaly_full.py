"""
baseline_anomaly.py  —  mass-spec anomaly detection (IF / LOF / OCSVM / OC-PCA / GODS)
                         + automatic result parsing/summary.

Runs the detection pipeline, writes the raw run log to --output-file, then
immediately parses that same file (or a glob of files, e.g. multiple
contamination levels) into a summary table written to --summary-file.

Usage (single run):
    python baseline_anomaly.py data/normal data/anomalies \
        --anomaly-files data/anomalies/foo.mgf \
        --output-file results_pct0.05.txt

Usage (skip one or more methods, e.g. GODS is slow / broken today):
    python baseline_anomaly.py data/normal data/anomalies \
        --anomaly-files data/anomalies/foo.mgf \
        --skip gods \
        --output-file results_pct0.05.txt

    python baseline_anomaly.py data/normal data/anomalies \
        --skip gods ocpca \
        --output-file results_pct0.05.txt

Usage (summarize across multiple prior runs instead of running the pipeline):
    python baseline_anomaly.py --summarize-only "results_pct*.txt" \
        --summary-file baseline_summary.txt
"""
import subprocess
import sys

# Automatically install autograd
def install_autograd():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "autograd"])
    subprocess.check_call([sys.executable, "pip", "install", "pymanopt"])

# Call the function before importing
try:
    import autograd
except ImportError:
    install_autograd()
    import autograd


import re
import sys
import io
import glob
import csv
import time
import resource
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score

from gods_wrapper import run_gods, run_kgods

_PROCESS_START_WALL   = time.perf_counter()
_PROCESS_START_RUSAGE = resource.getrusage(resource.RUSAGE_SELF)

# ── SKIPPABLE METHODS ─────────────────────────────────────────────────────────
# Keys accepted by --skip on the CLI / `skip=` on run_pipeline().
SKIP_CHOICES = ["if", "lof", "svm", "ocpca", "gods", "kgods"]


# ═════════════════════════════════════════════════════════════════════════
# PART 1 — PIPELINE (formerly baseline_anomaly.py)
# ═════════════════════════════════════════════════════════════════════════

# ── RAW RESULTS CSV ───────────────────────────────────────────────────────────

def write_raw_results_csv(spectra, if_pred, lof_pred, svm_pred, ocpca_pred,
                          gods_pred, contamination, output_file):
    """
    Any of the *_pred arrays may be None if that model was skipped via
    --skip; in that case the corresponding column is filled with
    'skipped' instead of raising.
    """
    out_dir  = Path(output_file).parent / f"contamination_{contamination}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "raw_outlier_results.csv"

    def _label(pred, i):
        if pred is None:
            return "skipped"
        return "outlier" if pred[i] == -1 else "normal"

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["identifier", "source_file",
                         "isolation_forest_outlier", "lof_outlier",
                         "ocsvm_outlier", "ocpca_outlier",
                         "gods_outlier"])
        for i, s in enumerate(spectra):
            identifier = s.get("title") or f"spectrum_{i}"
            writer.writerow([
                identifier,
                s.get("source_file"),
                _label(if_pred, i),
                _label(lof_pred, i),
                _label(svm_pred, i),
                _label(ocpca_pred, i),
                _label(gods_pred, i),
            ])

    print(f"  Raw outlier CSV saved → {csv_path}")


# ── LINUX time SNAPSHOT ───────────────────────────────────────────────────────

def _linux_time_snapshot():
    ru   = resource.getrusage(resource.RUSAGE_SELF)
    wall = time.perf_counter() - _PROCESS_START_WALL
    user = ru.ru_utime - _PROCESS_START_RUSAGE.ru_utime
    sys_ = ru.ru_stime - _PROCESS_START_RUSAGE.ru_stime
    return wall, user, sys_


# ── TIMING UTILITY ────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self._entries = []

    @contextmanager
    def measure(self, label):
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self._entries.append((label, elapsed))
        print(f"  ⏱  {label}: {elapsed:.3f}s")

    def entries(self):
        return list(self._entries)

    def total(self):
        return sum(e for _, e in self._entries)

    def linux_time(self):
        return _linux_time_snapshot()


# ── OUTPUT WRITER ─────────────────────────────────────────────────────────────

class ResultWriter:
    """Writes to both stdout and a txt file simultaneously."""
    def __init__(self, path):
        self._path = Path(path)
        self._buf  = []

    def write(self, text=""):
        print(text)
        self._buf.append(text)

    def flush(self):
        with open(self._path, "a") as fh:
            fh.write("\n".join(self._buf) + "\n")
        print(f"\n  Results saved → {self._path}")


# ── 1. MGF READER ─────────────────────────────────────────────────────────────

def read_mgf(filepath, label=0):
    spectra  = []
    current  = None
    filepath = Path(filepath)

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line == "BEGIN IONS":
                current = {"title": None, "pepmass": None, "charge": None,
                           "source_file": str(filepath), "label": label,
                           "mzs": [], "intensities": []}

            elif line == "END IONS":
                if current is not None:
                    current["mzs"]         = np.array(current["mzs"],        dtype=np.float32)
                    current["intensities"] = np.array(current["intensities"], dtype=np.float32)
                    spectra.append(current)
                current = None

            elif current is None:
                continue

            elif line.startswith("TITLE="):
                current["title"] = line[6:]
            elif line.startswith("PEPMASS="):
                current["pepmass"] = float(line[8:].split()[0])
            elif line.startswith("CHARGE="):
                current["charge"] = int(line[7:].replace("+", "").replace("-", ""))
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        current["mzs"].append(float(parts[0]))
                        current["intensities"].append(float(parts[1]))
                    except ValueError:
                        pass

    return spectra


# ── 2. MULTI-DIRECTORY / MULTI-FILE LOADER ───────────────────────────────────

def load_mgf_dirs(*paths, anomaly_paths=None, recursive=False):
    spectra       = []
    file_log      = []
    seen          = set()
    anomaly_paths = {Path(p).resolve() for p in (anomaly_paths or [])}

    def _load_file(mgf_path):
        mgf_path = Path(mgf_path).resolve()
        if mgf_path in seen:
            return
        seen.add(mgf_path)
        label = 1 if mgf_path in anomaly_paths else 0
        tag   = " [ANOMALY FILE]" if label else ""
        try:
            batch = read_mgf(mgf_path, label=label)
            spectra.extend(batch)
            file_log.append((mgf_path, len(batch), label))
            print(f"  ✓ {mgf_path}  ({len(batch)} spectra){tag}")
        except Exception as e:
            print(f"  [ERROR] {mgf_path}: {e}")

    for p in paths:
        p = Path(p)
        if p.is_dir():
            pattern   = "**/*.mgf" if recursive else "*.mgf"
            mgf_files = sorted(p.glob(pattern)) + sorted(p.glob(pattern.replace("mgf", "MGF")))
            if not mgf_files:
                print(f"  [WARN] No .mgf files found in: {p}")
            for f in mgf_files:
                _load_file(f)
        elif p.is_file():
            _load_file(p)
        else:
            print(f"  [WARN] Path not found, skipping: {p}")

    return spectra, file_log


# ── 3. BINNING ────────────────────────────────────────────────────────────────

def spectrum_to_bins(mzs, intensities, mz_min=0, mz_max=2000, bin_width=1.0):
    n_bins = int((mz_max - mz_min) / bin_width)
    vec    = np.zeros(n_bins, dtype=np.float32)
    idx    = ((mzs - mz_min) / bin_width).astype(int)
    valid  = (idx >= 0) & (idx < n_bins)
    for i, v in zip(idx[valid], intensities[valid]):
        vec[i] += v
    return vec


def build_matrix(spectra, mz_min=0, mz_max=2000, bin_width=1.0):
    rows = [spectrum_to_bins(s["mzs"], s["intensities"], mz_min, mz_max, bin_width)
            for s in spectra]
    return np.vstack(rows)


# ── 4. PREPROCESSING ──────────────────────────────────────────────────────────

def preprocess_if(X):
    """
    IsolationForest / GODS / KGODS: log1p only — NO normalisation.
    Anomalies have ~14,000x higher total intensity; any normalisation
    erases this signal by mapping every spectrum to the same scale.
    log1p compresses dynamic range while preserving the magnitude gap.
    """
    return np.log1p(X.copy().astype(np.float64))

def preprocess_lof(X):
    Xl  = np.log1p(X.copy().astype(np.float64))
    mag = np.log1p(X.sum(axis=1, keepdims=True))
    return np.hstack([Xl, mag])

def preprocess_svm(X):
    Xl  = np.log1p(X.copy().astype(np.float64))
    mag = np.log1p(X.sum(axis=1, keepdims=True))
    return np.hstack([Xl, mag])  # no L2 norm — it erases the magnitude signal

def preprocess_ocpca(X):
    """
    OC-PCA: log1p only, same as IF.
    Reconstruction error is dominated by high-intensity anomalies, so
    preserving the magnitude gap via log1p (without normalisation) is
    the right choice here too.
    """
    return np.log1p(X.copy().astype(np.float64))


# ── 5. PCA HELPER ─────────────────────────────────────────────────────────────

def apply_pca(X, n_pca, label, rw):
    if n_pca > 0:
        n_comp = min(n_pca, X.shape[0], X.shape[1])
        pca    = PCA(n_components=n_comp)
        Xp     = pca.fit_transform(X)
        rw.write(f"  [{label}] explained variance (top {n_comp} PCs): "
                 f"{pca.explained_variance_ratio_.sum():.1%}")
        return Xp
    else:
        rw.write(f"  [{label}] PCA skipped — using full {X.shape[1]}-dim feature matrix")
        return X


# ── 6. OUTLIER DETECTION ──────────────────────────────────────────────────────

def run_isolation_forest(X, contamination=0.05, n_estimators=100):
    clf    = IsolationForest(n_estimators=n_estimators,
                             contamination=contamination, random_state=42)
    pred   = clf.fit_predict(X)
    scores = clf.decision_function(X)
    return pred, scores


def run_lof(X, n_neighbors=20, contamination=0.05):
    """
    LOF with deduplication.

    Normal battery spectra are extremely sparse (~8 non-zero bins / 2000),
    causing many duplicate rows in PCA space. Duplicates cause distance=0
    between neighbours → degenerate LOF scores and sklearn warnings.

    Fix:
      1. Find unique rows (rounded to 6 dp to catch near-duplicates).
      2. Run LOF only on unique points with a safe n_neighbors.
      3. Broadcast predictions back to all original indices.
    """
    X_rounded = np.round(X, decimals=6)
    _, unique_idx, inverse_idx = np.unique(
        X_rounded, axis=0, return_index=True, return_inverse=True
    )

    n_dupes = X.shape[0] - len(unique_idx)
    if n_dupes > 0:
        print(f"    LOF: {n_dupes:,} duplicate / near-duplicate rows collapsed "
              f"→ running on {len(unique_idx):,} unique points")

    k = min(n_neighbors, len(unique_idx) - 1)
    if k != n_neighbors:
        print(f"    LOF: n_neighbors clamped {n_neighbors} → {k} "
              f"(only {len(unique_idx):,} unique points)")

    lof      = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
    pred_u   = lof.fit_predict(X[unique_idx])
    scores_u = lof.negative_outlier_factor_

    pred   = pred_u[inverse_idx]
    scores = scores_u[inverse_idx]
    return pred, scores


def run_ocsvm(X, contamination=0.05, nu=None,
              kernel="rbf", gamma="scale",
              subsample=10_000, random_state=42):
    """
    One-Class SVM with subsampling for scalability.

    OCSVM is O(n²)–O(n³) to train; infeasible on 260K points directly.
    Strategy:
      1. Draw `subsample` points uniformly at random for training.
      2. Call clf.predict() on all n points for scoring.
    """
    rng = np.random.default_rng(random_state)
    nu  = float(np.clip(nu if nu is not None else contamination, 1e-5, 1 - 1e-5))

    n = X.shape[0]
    if n > subsample:
        idx_train = rng.choice(n, size=subsample, replace=False)
        X_train   = X[idx_train]
        print(f"    OCSVM: training on {subsample:,} / {n:,} samples "
              f"(nu={nu:.5f}, kernel={kernel}, gamma={gamma})")
    else:
        X_train = X
        print(f"    OCSVM: training on all {n:,} samples "
              f"(nu={nu:.5f}, kernel={kernel}, gamma={gamma})")

    clf    = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    clf.fit(X_train)
    pred   = clf.predict(X)
    scores = clf.decision_function(X)
    return pred, scores


def run_ocpca(X, contamination=0.05, n_components=5):
    """
    OC-PCA anomaly detector based on reconstruction error.

    Fits a PCA model on the full dataset and flags spectra whose
    reconstruction error exceeds the (1 - contamination) percentile.
    Anomalies with atypical intensity patterns project poorly onto the
    normal subspace and therefore have high reconstruction error.

    scores : negative reconstruction error (higher = more normal),
             matching sklearn's decision_function sign convention.
    """
    n_comp    = min(n_components, X.shape[0], X.shape[1])
    pca       = PCA(n_components=n_comp)
    X_proj    = pca.fit_transform(X)
    X_recon   = pca.inverse_transform(X_proj)
    errors    = np.linalg.norm(X - X_recon, axis=1)
    threshold = np.percentile(errors, 100 * (1 - contamination))
    pred      = np.where(errors > threshold, -1, 1)
    scores    = -errors
    return pred, scores


# ── 7. EVALUATION ─────────────────────────────────────────────────────────────

def evaluate(model_name, pred, ground_truth, scores=None,
             higher_is_anomaly=False, rw=None):
    """
    scores : optional continuous anomaly score used for AUROC / AUPRC.
             By convention here, most of our detectors return a score
             where HIGHER = MORE NORMAL (sklearn decision_function style:
             IsolationForest, LOF's negative_outlier_factor_, OCSVM,
             and our OC-PCA's -errors all follow this).
             Set higher_is_anomaly=True for detectors whose raw score
             convention is flipped, e.g. if gods_wrapper returns scores
             that way — check before calling.
    """
    def w(text=""):
        if rw:
            rw.write(text)
        else:
            print(text)

    pred_anom = (pred == -1).astype(int)
    gt        = np.array(ground_truth)

    TP = int(((pred_anom == 1) & (gt == 1)).sum())
    FP = int(((pred_anom == 1) & (gt == 0)).sum())
    TN = int(((pred_anom == 0) & (gt == 0)).sum())
    FN = int(((pred_anom == 0) & (gt == 1)).sum())

    precision      = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall         = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1             = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
    fpr            = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    detection_rate = recall

    # ── AUROC / AUPRC (threshold-free, uses continuous scores) ─────────────
    auroc = auprc = None
    if scores is not None and len(np.unique(gt)) > 1:
        anomaly_score = np.asarray(scores) if higher_is_anomaly else -np.asarray(scores)
        try:
            auroc = roc_auc_score(gt, anomaly_score)
            auprc = average_precision_score(gt, anomaly_score)
        except ValueError:
            pass  # e.g. degenerate / constant scores

    w(f"")
    w(f"  ── {model_name} ──────────────────────────────")
    w(f"  {'':>4} {'Pred: normal':>14} {'Pred: anomaly':>14}")
    w(f"  {'GT: normal':>10}   {TN:>12}   {FP:>12}   ← False Positives")
    w(f"  {'GT: anomaly':>10}   {FN:>12}   {TP:>12}   ← True Positives")
    w(f"")
    w(f"  TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    w(f"  Precision      : {precision:.4f}  (of flagged, how many were real anomalies)")
    w(f"  Recall         : {recall:.4f}  (of real anomalies, how many were caught)")
    w(f"  F1             : {f1:.4f}")
    w(f"  Detection Rate : {detection_rate:.4f}  (TP / (TP+FN), same as Recall/TPR)")
    w(f"  FP Rate        : {fpr:.4f}  (false alarms among normal spectra)")
    w(f"  AUROC          : {f'{auroc:.4f}' if auroc is not None else 'N/A'}")
    w(f"  AUPRC          : {f'{auprc:.4f}' if auprc is not None else 'N/A'}")

    return dict(TP=TP, FP=FP, TN=TN, FN=FN,
                precision=precision, recall=recall, f1=f1,
                detection_rate=detection_rate, fpr=fpr,
                auroc=auroc, auprc=auprc)


# ── 8. SANITY CHECK ───────────────────────────────────────────────────────────

def print_anomaly_sanity(spectra, X_raw):
    gt     = np.array([s["label"] for s in spectra])
    n_anom = gt.sum()
    if n_anom == 0:
        return
    print("\n  ── Anomaly sanity check ──────────────────────────────")
    print(f"  Mean peaks   — normal : "
          f"{np.mean([len(s['mzs']) for s in spectra if s['label']==0]):.1f}")
    print(f"  Mean peaks   — anomaly: "
          f"{np.mean([len(s['mzs']) for s in spectra if s['label']==1]):.1f}")
    print(f"  Raw int sum  — normal : {X_raw[gt==0].sum(axis=1).mean():.2f}")
    print(f"  Raw int sum  — anomaly: {X_raw[gt==1].sum(axis=1).mean():.2f}")
    print(f"  Non-zero bins— normal : {(X_raw[gt==0] > 0).sum(axis=1).mean():.1f}")
    print(f"  Non-zero bins— anomaly: {(X_raw[gt==1] > 0).sum(axis=1).mean():.1f}")


# ── 9. MAIN PIPELINE ──────────────────────────────────────────────────────────

def run_pipeline(*paths,
                 anomaly_paths=None,
                 recursive=False,
                 mz_min=0, mz_max=2000, bin_width=1.0,
                 n_pca=50, contamination=0.05,
                 lof_neighbors=20,
                 svm_subsample=10_000,
                 svm_kernel="rbf",
                 svm_gamma="scale",
                 ocpca_components=10,
                 gods_subspaces=5,
                 gods_eta=0.01,
                 gods_lam=0.1,
                 gods_subsample=20_000,
                 gods_higher_is_anomaly=False,
                 kgods_kernel="rbf",
                 kgods_sigma=0.01,
                 kgods_subspaces=1,
                 kgods_subsample=5_000,
                 skip=None,
                 output_file="results.txt"):

    # Normalize/validate the skip set once up front.
    skip = set(skip or [])
    bad  = skip - set(SKIP_CHOICES)
    if bad:
        raise ValueError(f"Unknown --skip value(s): {sorted(bad)}. "
                         f"Valid choices: {SKIP_CHOICES}")

    timer = Timer()
    rw    = ResultWriter(output_file)

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rw.write(f"HyperSpec Anomaly Detection — {run_ts}")
    rw.write(f"Args: bin_width={bin_width}  mz=[{mz_min},{mz_max}]  "
             f"n_pca={n_pca}  contamination={contamination}  "
             f"lof_neighbors={lof_neighbors}  "
             f"svm_subsample={svm_subsample}  svm_kernel={svm_kernel}  "
             f"svm_gamma={svm_gamma}  ocpca_components={ocpca_components}  "
             f"gods_subspaces={gods_subspaces}  gods_eta={gods_eta}  "
             f"gods_lam={gods_lam}  gods_subsample={gods_subsample}  "
             f"kgods_kernel={kgods_kernel}  kgods_sigma={kgods_sigma}  "
             f"kgods_subspaces={kgods_subspaces}  kgods_subsample={kgods_subsample}")
    if skip:
        rw.write(f"Skipping: {sorted(skip)}")
    rw.write("=" * 72)

    # ── load ─────────────────────────────────────────────────────────────────
    rw.write(f"\nScanning {len(paths)} path(s) ...")
    with timer.measure("File loading"):
        spectra, file_log = load_mgf_dirs(*paths, anomaly_paths=anomaly_paths,
                                          recursive=recursive)

    if not spectra:
        raise RuntimeError("No spectra loaded — check your paths.")

    n_anom   = sum(s["label"] for s in spectra)
    n_normal = len(spectra) - n_anom
    rw.write(f"\n  Total   : {len(spectra)} spectra across {len(file_log)} file(s)")
    rw.write(f"  Normal  : {n_normal}  |  Known anomalies (ground truth): {n_anom}")

    ground_truth = np.array([s["label"] for s in spectra])

    # ── bin ──────────────────────────────────────────────────────────────────
    rw.write("\nBuilding bin matrix ...")
    with timer.measure("Binning"):
        X_raw = build_matrix(spectra, mz_min, mz_max, bin_width)

    print_anomaly_sanity(spectra, X_raw)

    # ── preprocess ───────────────────────────────────────────────────────────
    # Cheap relative to model fitting, so always compute — keeps things simple
    # and lets any of these be reused if you flip a --skip flag off later.
    rw.write("\nPreprocessing ...")
    with timer.measure("Preprocessing"):
        X_if    = preprocess_if(X_raw)   # shared by IF, GODS, KGODS
        print("preprocessed if")
        X_lof   = X_raw
        print("preprocessed lof")
        X_svm   = preprocess_svm(X_raw)
        print("preprocessed svm")
        X_ocpca = preprocess_ocpca(X_raw)
        print("preprocessed oc_pca")

    # ── PCA ──────────────────────────────────────────────────────────────────
    # Only run PCA for methods that are actually going to be fit — PCA on
    # 260K x 2000 can itself be slow, so a skipped method should skip its PCA too.
    rw.write("Running PCA ...")

    X_pca_if = X_pca_lof = X_pca_svm = X_pca_gods = None

    if "if" not in skip:
        with timer.measure("PCA (IF)"):
            X_pca_if = apply_pca(X_if, n_pca, "IF", rw)
    else:
        rw.write("  [IF]  PCA skipped (--skip if)")

    if "lof" not in skip:
        with timer.measure("PCA (LOF)"):
            X_pca_lof = apply_pca(X_lof, n_pca, "LOF", rw)
    else:
        rw.write("  [LOF] PCA skipped (--skip lof)")

    if "svm" not in skip:
        with timer.measure("PCA (SVM)"):
            X_pca_svm = apply_pca(X_svm, n_pca, "SVM", rw)
    else:
        rw.write("  [SVM] PCA skipped (--skip svm)")

    if "gods" not in skip:
        with timer.measure("PCA (GODS)"):
            # GODS reuses IF's PCA projection when available; otherwise
            # (e.g. IF was skipped but GODS wasn't) compute its own.
            X_pca_gods = X_pca_if if X_pca_if is not None else apply_pca(X_if, n_pca, "GODS", rw)
    else:
        rw.write("  [GODS] PCA skipped (--skip gods)")
    # OC-PCA / KGODS do their own internal dimensionality reduction.

    # ── models ───────────────────────────────────────────────────────────────
    if_pred = if_scores = None
    lof_pred = lof_scores = None
    svm_pred = svm_scores = None
    ocpca_pred = ocpca_scores = None
    gods_pred = gods_scores = None

    if "if" not in skip:
        rw.write("\nRunning Isolation Forest ...")
        with timer.measure("Isolation Forest"):
            if_pred, if_scores = run_isolation_forest(X_pca_if, contamination)
    else:
        rw.write("\nSkipping Isolation Forest (--skip if)")

    if "lof" not in skip:
        rw.write(f"Running LOF (n_neighbors={lof_neighbors}) ...")
        with timer.measure("LOF"):
            lof_pred, lof_scores = run_lof(X_pca_lof,
                                           n_neighbors=lof_neighbors,
                                           contamination=contamination)
    else:
        rw.write("Skipping LOF (--skip lof)")

    if "svm" not in skip:
        rw.write(f"Running One-Class SVM "
                 f"(subsample={svm_subsample:,}, kernel={svm_kernel}, "
                 f"gamma={svm_gamma}) ...")
        with timer.measure("One-Class SVM"):
            svm_pred, svm_scores = run_ocsvm(
                X_pca_svm,
                contamination=contamination,
                kernel=svm_kernel,
                gamma=svm_gamma,
                subsample=svm_subsample,
            )
    else:
        rw.write("Skipping One-Class SVM (--skip svm)")

    if "ocpca" not in skip:
        rw.write(f"Running OC-PCA (n_components={ocpca_components}) ...")
        with timer.measure("OC-PCA"):
            ocpca_pred, ocpca_scores = run_ocpca(
                X_ocpca,
                contamination=contamination,
                n_components=ocpca_components,
            )
    else:
        rw.write("Skipping OC-PCA (--skip ocpca)")

    if "gods" not in skip:
        rw.write(f"Running GODS "
                 f"(num_subspaces={gods_subspaces}, eta={gods_eta}, "
                 f"lam={gods_lam}, subsample={gods_subsample:,}) ...")
        with timer.measure("GODS"):
            gods_pred, gods_scores = run_gods(
                X_pca_gods,
                contamination=contamination,
                num_subspaces=gods_subspaces,
                eta=gods_eta,
                lam=gods_lam,
                subsample=gods_subsample,
                normalize=False,
            )
    else:
        rw.write("Skipping GODS (--skip gods)")

    if "kgods" not in skip:
        rw.write(f"Running KGODS "
                 f"(kernel={kgods_kernel}, sigma={kgods_sigma}, "
                 f"num_subspaces={kgods_subspaces}, subsample={kgods_subsample:,}) ...")
        # NOTE: kept as-is from the original script — run_kgods is imported
        # but not actually invoked here yet. This message is left in place
        # (guarded by --skip kgods) rather than wired up, since that's a
        # separate change from adding skip support.
    else:
        rw.write("Skipping KGODS (--skip kgods)")

    write_raw_results_csv(
        spectra=spectra,
        if_pred=if_pred,
        lof_pred=lof_pred,
        svm_pred=svm_pred,
        ocpca_pred=ocpca_pred,
        gods_pred=gods_pred,
        contamination=contamination,
        output_file=output_file,
    )

    # ── evaluation ───────────────────────────────────────────────────────────
    if_stats = lof_stats = svm_stats = ocpca_stats = None
    gods_stats = kgods_stats = None
    if n_anom > 0:
        rw.write("\n\n══ EVALUATION (ground truth available) ══════════════════════════════")

        if if_pred is not None:
            if_stats = evaluate("Isolation Forest", if_pred, ground_truth,
                                scores=if_scores, rw=rw)
        else:
            rw.write("\n  [SKIPPED] Isolation Forest — no stats (--skip if)")

        if lof_pred is not None:
            lof_stats = evaluate("LOF", lof_pred, ground_truth,
                                 scores=lof_scores, rw=rw)
        else:
            rw.write("\n  [SKIPPED] LOF — no stats (--skip lof)")

        if svm_pred is not None:
            svm_stats = evaluate("One-Class SVM", svm_pred, ground_truth,
                                 scores=svm_scores, rw=rw)
        else:
            rw.write("\n  [SKIPPED] One-Class SVM — no stats (--skip svm)")

        if ocpca_pred is not None:
            ocpca_stats = evaluate("OC-PCA", ocpca_pred, ground_truth,
                                   scores=ocpca_scores, rw=rw)
        else:
            rw.write("\n  [SKIPPED] OC-PCA — no stats (--skip ocpca)")

        if gods_pred is not None:
            gods_stats = evaluate("GODS", gods_pred, ground_truth,
                                  scores=gods_scores,
                                  higher_is_anomaly=gods_higher_is_anomaly, rw=rw)
        else:
            rw.write("\n  [SKIPPED] GODS — no stats (--skip gods)")
    else:
        rw.write("\n  [INFO] No anomaly files labelled — skipping TP/FP stats.")

    # ── timing summary ────────────────────────────────────────────────────────
    real, user, sys_ = timer.linux_time()

    rw.write("\n\n══ TIMING SUMMARY (perf_counter) ════════════════════════════════════")
    rw.write(f"  {'Stage':<40} {'Time (s)':>10}")
    rw.write(f"  {'-'*40} {'-'*10}")
    for label, elapsed in timer.entries():
        rw.write(f"  {label:<40} {elapsed:>10.3f}")
    rw.write(f"  {'─'*40} {'─'*10}")
    rw.write(f"  {'TOTAL (instrumented stages only)':<40} {timer.total():>10.3f}")
    rw.write(f"")
    rw.write(f"  Note: perf_counter total covers only the labeled stages above.")
    rw.write(f"  Uninstrumented overhead (I/O, Python internals) is in 'real - total'.")
    rw.write(f"")
    rw.write(f"══ LINUX time EQUIVALENT ════════════════════════════════════════════")
    rw.write(f"  {'Metric':<40} {'Time (s)':>10}   Notes")
    rw.write(f"  {'-'*40} {'-'*10}   {'-'*40}")
    rw.write(f"  {'real  (wall clock)':<40} {real:>10.3f}   ≈ `time cmd` real")
    rw.write(f"  {'user  (user-space CPU)':<40} {user:>10.3f}   ≈ `time cmd` user")
    rw.write(f"  {'sys   (kernel CPU)':<40} {sys_:>10.3f}   ≈ `time cmd` sys")
    rw.write(f"  {'overhead (real - perf total)':<40} {real - timer.total():>10.3f}   "
             f"imports, I/O, etc.")

    rw.flush()

    return {
        "spectra"       : spectra,
        "ground_truth"  : ground_truth,
        "X_raw"         : X_raw,
        "X_pca_if"      : X_pca_if,
        "X_pca_lof"     : X_pca_lof,
        "X_pca_svm"     : X_pca_svm,
        "X_pca_gods"    : X_pca_gods,
        "if_pred"       : if_pred, "if_scores"     : if_scores, "if_stats"      : if_stats,
        "lof_pred"      : lof_pred, "lof_scores"    : lof_scores, "lof_stats"     : lof_stats,
        "svm_pred"      : svm_pred, "svm_scores"    : svm_scores, "svm_stats"     : svm_stats,
        "ocpca_pred"    : ocpca_pred, "ocpca_scores"  : ocpca_scores, "ocpca_stats"   : ocpca_stats,
        "gods_pred"     : gods_pred, "gods_scores"   : gods_scores, "gods_stats"    : gods_stats,
        "timings"       : timer.entries(),
        "linux_time"    : {"real": real, "user": user, "sys": sys_},
        "output_file"   : output_file,
        "skip"          : sorted(skip),
    }


# ═════════════════════════════════════════════════════════════════════════
# PART 2 — SUMMARY PARSER (formerly parse_results.py)
# ═════════════════════════════════════════════════════════════════════════

MODELS = [
    ("Isolation Forest", "Isolation_Forest"),
    ("LOF",              "LOF"),
    ("One-Class SVM",    "One_Class_SVM"),
    ("OC-PCA",           "OC_PCA"),
    ("GODS",             "GODS"),
]

TIMING_STAGE_MAP = {
    "Preprocessing":      "preprocessing",
    "PCA (IF)":           "pca_IF",
    "PCA (LOF)":          "pca_LOF",
    "PCA (SVM)":          "pca_SVM",
    "PCA (GODS)":         "pca_GODS",
    "Isolation Forest":   "Isolation_Forest",
    "LOF":                "LOF",
    "One-Class SVM":      "One_Class_SVM",
    "OC-PCA":             "OC_PCA",
    "GODS":               "GODS",
}


def parse_block(block):
    """Extract one run's worth of metrics from a single result block."""
    run = {}

    m = re.search(r"contamination=([\d.eE+-]+)", block)
    run["contamination"] = float(m.group(1)) if m else None

    m = re.search(r"Total\s+:\s+(\d+)", block)
    run["total"] = int(m.group(1)) if m else None

    m = re.search(r"Normal\s+:\s+(\d+)", block)
    run["n_normal"] = int(m.group(1)) if m else None

    m = re.search(r"Known anomalies \(ground truth\):\s+(\d+)", block)
    run["n_anomaly_gt"] = int(m.group(1)) if m else None

    m = re.search(r"TOTAL \(instrumented stages only\)\s+([\d.]+)", block)
    run["total_time_s"] = float(m.group(1)) if m else None

    # ── per-stage timings ────────────────────────────────────────────────────
    timing = {}
    timing_block_m = re.search(
        r"══ TIMING SUMMARY.*?(?=\n══(?! TIMING)|\Z)", block, flags=re.S
    )
    if timing_block_m:
        tb = timing_block_m.group(0)
        for stage_name, tag in TIMING_STAGE_MAP.items():
            pat_name = re.escape(stage_name)
            tm = re.search(rf"^\s*{pat_name}\s+([\d.]+)", tb, flags=re.M)
            timing[tag] = float(tm.group(1)) if tm else None
    run["timing"] = timing

    # ── per-model classification metrics ────────────────────────────────────
    model_labels = [label for label, _ in MODELS]
    boundary_alts = "|".join(re.escape(l) for l in model_labels)
    boundary = rf"(?=^\s*──\s*(?:{boundary_alts})|^══|\Z)"

    for label, tag in MODELS:
        section_start = r"──\s*" + re.escape(label).replace(r"\ ", r"\s+").replace(r"\-", r"-")
        pat = section_start + r".*?" + boundary

        msec = re.search(pat, block, flags=re.S | re.M)

        if not msec:
            run[tag] = None
            continue

        model_block = msec.group(0)

        def _int(key, mb=model_block):
            m2 = re.search(rf"\b{key}\s*=\s*(\d+)", mb)
            return int(m2.group(1)) if m2 else None

        def _float_metric(key, mb=model_block):
            """Extract lines like 'AUROC          : 0.9231' or '... : N/A'."""
            m2 = re.search(rf"\b{key}\s*:\s*(N/A|[\d.]+)", mb)
            if not m2:
                return None
            val = m2.group(1)
            return None if val == "N/A" else float(val)

        TP = _int("TP"); FP = _int("FP")
        TN = _int("TN"); FN = _int("FN")

        if None in (TP, FP, TN, FN):
            run[tag] = None
            continue

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall    = TP / (TP + FN) if (TP + FN) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr       = FP / (FP + TN) if (FP + TN) else 0.0

        run[tag] = dict(
            TP=TP, FP=FP, TN=TN, FN=FN,
            precision=precision, recall=recall, f1=f1, fpr=fpr,
            auroc=_float_metric("AUROC"),
            auprc=_float_metric("AUPRC"),
        )

    return run


def parse_file(filepath):
    text   = Path(filepath).read_text()
    blocks = re.split(r"(?=HyperSpec Anomaly Detection —)", text)
    blocks = [b.strip() for b in blocks if b.strip()]
    return [parse_block(b) for b in blocks]


def parse_paths(pattern):
    paths = sorted(glob.glob(pattern)) or [pattern]
    runs  = []
    for p in paths:
        try:
            runs.extend(parse_file(p))
        except FileNotFoundError:
            print(f"[WARN] File not found, skipping: {p}", file=sys.stderr)
    runs.sort(key=lambda r: r["contamination"] or 0.0)
    return runs


def _t(run, key):
    return run["timing"].get(key)

def _fmt(v, fmt=".1f"):
    return f"{v:{fmt}}" if v is not None else "     —"


def print_metrics_table(runs):
    col_method = 20
    header = (
        f"{'Method':<{col_method}}  {'Contam%':>9}  {'#GT':>6}  {'#Total':>8}  "
        f"{'Prec%':>7}  {'Rec%':>7}  {'F1':>6}  {'FPR%':>7}  "
        f"{'AUROC':>7}  {'AUPRC':>7}  "
        f"{'TP':>6}  {'FP':>6}  {'TotalTime(s)':>12}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in runs:
        pct   = r["contamination"] * 100 if r["contamination"] is not None else float("nan")
        n_gt  = r["n_anomaly_gt"] or 0
        total = r["total"]        or 0
        t     = r["total_time_s"] or 0.0
        first = True

        for label, tag in MODELS:
            s = r.get(tag)
            if s is None:
                prec = rec = f1 = fpr_pct = float("nan")
                auroc = auprc = float("nan")
                tp = fp = 0
            else:
                prec    = s["precision"] * 100
                rec     = s["recall"]    * 100
                f1      = s["f1"]
                fpr_pct = s["fpr"]       * 100
                auroc   = s.get("auroc")
                auprc   = s.get("auprc")
                auroc   = auroc if auroc is not None else float("nan")
                auprc   = auprc if auprc is not None else float("nan")
                tp      = s["TP"]
                fp      = s["FP"]

            if first:
                shared = f"{pct:9.4f}  {n_gt:6d}  {total:8d}"
                time_s = f"{t:12.1f}"
                first  = False
            else:
                shared = f"{'':9}  {'':6}  {'':8}"
                time_s = f"{'':12}"

            print(
                f"{label:<{col_method}}  {shared}  "
                f"{prec:7.2f}  {rec:7.2f}  {f1:6.4f}  {fpr_pct:7.4f}  "
                f"{auroc:7.4f}  {auprc:7.4f}  "
                f"{tp:6d}  {fp:6d}  {time_s}"
            )
        print(sep)


def print_timing_table(runs):
    col_method = 20
    header = (
        f"{'Method':<{col_method}}  {'Contam%':>9}  "
        f"{'Preproc(s)':>10}  {'PCA(s)':>8}  {'Fit(s)':>8}  {'Subtotal(s)':>11}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    pca_key = {
        "Isolation_Forest": "pca_IF",
        "LOF":              "pca_LOF",
        "One_Class_SVM":    "pca_SVM",
        "OC_PCA":           None,
        "GODS":             "pca_GODS",
    }
    fit_key = {
        "Isolation_Forest": "Isolation_Forest",
        "LOF":              "LOF",
        "One_Class_SVM":    "One_Class_SVM",
        "OC_PCA":           "OC_PCA",
        "GODS":             "GODS",
    }

    for r in runs:
        pct     = r["contamination"] * 100 if r["contamination"] is not None else float("nan")
        preproc = _t(r, "preprocessing")
        first   = True

        for label, tag in MODELS:
            pk   = pca_key[tag]
            pca  = _t(r, pk) if pk else None
            fit  = _t(r, fit_key[tag])

            if first:
                pct_str = f"{pct:9.4f}"
                pre_str = _fmt(preproc, "10.3f")
                first   = False
            else:
                pct_str = f"{'':9}"
                pre_str = f"{'':10}"

            pca_str = _fmt(pca, "8.3f") if pca is not None else f"{'':8}"
            fit_str = _fmt(fit, "8.3f")

            if tag == "Isolation_Forest":
                sub_val = (preproc or 0) + (pca or 0) + (fit or 0)
                tot_str = f"{sub_val:11.3f}" if preproc is not None else f"{'—':>11}"
            elif tag == "OC_PCA":
                sub_val = fit or 0
                tot_str = f"{sub_val:11.3f}" if fit is not None else f"{'—':>11}"
            else:
                sub_val = (pca or 0) + (fit or 0)
                tot_str = f"{sub_val:11.3f}" if pca is not None else f"{'—':>11}"

            print(
                f"{label:<{col_method}}  {pct_str}  "
                f"{pre_str}  {pca_str}  {fit_str}  {tot_str}"
            )
        print(sep)


def write_summary(pattern, summary_file):
    """Parse `pattern` (a file path or glob) and write the summary table."""
    runs = parse_paths(pattern)
    if not runs:
        print("No runs found — check your file path / glob pattern.")
        return

    buf     = io.StringIO()
    _stdout = sys.stdout
    sys.stdout = buf

    print("═" * 80)
    print("  CLASSIFICATION METRICS")
    print("═" * 80)
    print_metrics_table(runs)

    print()
    print("═" * 80)
    print("  TIMING BREAKDOWN  (Preproc shared; PCA and Fit are per-method)")
    print("═" * 80)
    print_timing_table(runs)

    sys.stdout = _stdout
    output = buf.getvalue()
    print(output, end="")

    Path(summary_file).write_text(output)
    print(f"\nSummary saved → {summary_file}")


# ═════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mass spec binning + outlier detection with optional "
                    "ground-truth evaluation, followed by automatic summary parsing."
    )
    parser.add_argument("paths", nargs="*",
                        help="Directories or .mgf files to load (mix freely). "
                             "Not needed with --summarize-only.")
    parser.add_argument("--anomaly-files", nargs="*", default=[],
                        metavar="FILE",
                        help="Files containing known true anomalies "
                             "(used only for TP/FP scoring).")
    parser.add_argument("--recursive",          action="store_true")
    parser.add_argument("--bin-width",          type=float, default=1.0)
    parser.add_argument("--mz-min",             type=float, default=0.0)
    parser.add_argument("--mz-max",             type=float, default=2000.0)
    parser.add_argument("--n-pca",              type=int,   default=50,
                        help="PCA components for IF/LOF/SVM/GODS. 0 = skip.")
    parser.add_argument("--contamination",      type=float, default=0.05)
    parser.add_argument("--lof-neighbors",      type=int,   default=20)
    parser.add_argument("--svm-subsample",      type=int,   default=10_000)
    parser.add_argument("--svm-kernel",         type=str,   default="rbf",
                        choices=["rbf", "linear", "poly", "sigmoid"])
    parser.add_argument("--svm-gamma",          type=str,   default="scale")
    parser.add_argument("--ocpca-components",   type=int,   default=10,
                        help="PCA components for OC-PCA normal subspace.")
    # GODS
    parser.add_argument("--gods-subspaces",     type=int,   default=5)
    parser.add_argument("--gods-eta",           type=float, default=0.01)
    parser.add_argument("--gods-lam",           type=float, default=0.1)
    parser.add_argument("--gods-subsample",     type=int,   default=20_000)
    parser.add_argument("--gods-higher-is-anomaly", action="store_true",
                        help="Set if gods_wrapper.run_gods returns scores "
                             "where HIGHER = more anomalous. Check before using.")
    # KGODS
    parser.add_argument("--kgods-kernel",       type=str,   default="rbf",
                        choices=["rbf", "linear", "chisq", "min"])
    parser.add_argument("--kgods-sigma",        type=float, default=0.01)
    parser.add_argument("--kgods-subspaces",    type=int,   default=1)
    parser.add_argument("--kgods-subsample",    type=int,   default=5_000)

    # ── SKIP ────────────────────────────────────────────────────────────────
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=SKIP_CHOICES, metavar="METHOD",
                        help=f"Skip one or more methods entirely (no PCA, no "
                             f"fit, no eval, CSV column shows 'skipped'). "
                             f"Choices: {SKIP_CHOICES}. "
                             f"e.g. --skip gods  or  --skip gods ocpca")

    parser.add_argument("--output-file",        type=str,   default="results.txt",
                        help="Raw run log (same file(s) fed into the summary parser).")
    parser.add_argument("--summary-file",       type=str,   default="baseline_summary.txt",
                        help="Parsed summary table output path.")
    parser.add_argument("--summarize-only",     type=str,   default=None,
                        metavar="PATTERN",
                        help="Skip the pipeline; parse an existing file or glob "
                             "(e.g. 'results_pct*.txt') straight to --summary-file.")

    args = parser.parse_args()

    if args.summarize_only:
        write_summary(args.summarize_only, args.summary_file)
        sys.exit(0)

    if not args.paths:
        parser.error("paths are required unless --summarize-only is used")

    svm_gamma = args.svm_gamma
    try:
        svm_gamma = float(svm_gamma)
    except ValueError:
        pass  # keep as string ('scale' / 'auto')

    result = run_pipeline(
        *args.paths,
        anomaly_paths=args.anomaly_files,
        recursive=args.recursive,
        bin_width=args.bin_width,
        mz_min=args.mz_min,
        mz_max=args.mz_max,
        n_pca=args.n_pca,
        contamination=args.contamination,
        lof_neighbors=args.lof_neighbors,
        svm_subsample=args.svm_subsample,
        svm_kernel=args.svm_kernel,
        svm_gamma=svm_gamma,
        ocpca_components=args.ocpca_components,
        gods_subspaces=args.gods_subspaces,
        gods_eta=args.gods_eta,
        gods_lam=args.gods_lam,
        gods_subsample=args.gods_subsample,
        gods_higher_is_anomaly=args.gods_higher_is_anomaly,
        kgods_kernel=args.kgods_kernel,
        kgods_sigma=args.kgods_sigma,
        kgods_subspaces=args.kgods_subspaces,
        kgods_subsample=args.kgods_subsample,
        skip=args.skip,
        output_file=args.output_file,
    )

    # ── auto-summarize the run we just produced ─────────────────────────────
    print("\n" + "=" * 72)
    print("  Parsing results into summary table ...")
    print("=" * 72)
    write_summary(result["output_file"], args.summary_file)