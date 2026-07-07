import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import resource
from datetime import datetime
from contextlib import contextmanager
import csv
from gods_wrapper import run_gods, run_kgods

_PROCESS_START_WALL   = time.perf_counter()
_PROCESS_START_RUSAGE = resource.getrusage(resource.RUSAGE_SELF)


# ── RAW RESULTS CSV ───────────────────────────────────────────────────────────

def write_raw_results_csv(spectra, if_pred, lof_pred, svm_pred, ocpca_pred,
                          gods_pred, contamination, output_file):
    out_dir  = Path(output_file).parent / f"contamination_{contamination}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "raw_outlier_results.csv"

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["identifier", "source_file",
                         "isolation_forest_outlier", "lof_outlier",
                         "ocsvm_outlier", "ocpca_outlier",
                         "gods_outlier"])
                        #   "kgods_outlier"])
        for i, s in enumerate(spectra):
            identifier = s.get("title") or f"spectrum_{i}"
            writer.writerow([
                identifier,
                s.get("source_file"),
                "outlier" if if_pred[i]    == -1 else "normal",
                "outlier" if lof_pred[i]   == -1 else "normal",
                "outlier" if svm_pred[i]   == -1 else "normal",
                "outlier" if ocpca_pred[i] == -1 else "normal",
                "outlier" if gods_pred[i]  == -1 else "normal",
                # "outlier" if kgods_pred[i] == -1 else "normal",
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
             convention is flipped (higher = more anomalous), e.g. if
             gods_wrapper returns scores that way — check before calling.
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
                 output_file="results.txt"):

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
    # Preprocessing strategy per model:
    #
    #   IF / GODS / KGODS : log1p only
    #       Preserves the ~14,000x intensity gap. No normalisation.
    #
    #   LOF    : log1p only (L2 norm collapses sparse spectra to near-identical
    #            unit vectors → degenerate distance → bad LOF scores).
    #
    #   SVM    : log1p + magnitude feature appended (no L2 norm).
    #
    #   OC-PCA : log1p only — reconstruction error relies on magnitude gap.
    #
    #   GODS / KGODS pass normalize=False in run_gods/run_kgods for the same
    #   reason; the wrapper's internal normalisation flag is left off.
    rw.write("\nPreprocessing ...")
    with timer.measure("Preprocessing"):
        X_if    = preprocess_if(X_raw)   # shared by IF, GODS, KGODS, OC-PCA
        X_lof   = X_raw
        X_svm   = preprocess_svm(X_raw)
        X_ocpca = preprocess_ocpca(X_raw)

    # ── PCA ──────────────────────────────────────────────────────────────────
    rw.write("Running PCA ...")
    with timer.measure("PCA (IF)"):
        X_pca_if  = apply_pca(X_if,  n_pca, "IF",    rw)
    with timer.measure("PCA (LOF)"):
        X_pca_lof = apply_pca(X_lof, n_pca, "LOF",   rw)
    with timer.measure("PCA (SVM)"):
        X_pca_svm = apply_pca(X_svm, n_pca, "SVM",   rw)
    with timer.measure("PCA (GODS)"):
        X_pca_gods = apply_pca(X_if, n_pca, "GODS",  rw)
    # OC-PCA / KGODS do their own internal dimensionality reduction.

    # ── models ───────────────────────────────────────────────────────────────
    rw.write("\nRunning Isolation Forest ...")
    with timer.measure("Isolation Forest"):
        if_pred, if_scores = run_isolation_forest(X_pca_if, contamination)

    rw.write(f"Running LOF (n_neighbors={lof_neighbors}) ...")
    with timer.measure("LOF"):
        lof_pred, lof_scores = run_lof(X_pca_lof,
                                       n_neighbors=lof_neighbors,
                                       contamination=contamination)

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

    rw.write(f"Running OC-PCA (n_components={ocpca_components}) ...")
    with timer.measure("OC-PCA"):
        ocpca_pred, ocpca_scores = run_ocpca(
            X_ocpca,
            contamination=contamination,
            n_components=ocpca_components,
        )

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

    rw.write(f"Running KGODS "
             f"(kernel={kgods_kernel}, sigma={kgods_sigma}, "
             f"num_subspaces={kgods_subspaces}, subsample={kgods_subsample:,}) ...")
    


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
        if_stats    = evaluate("Isolation Forest", if_pred,    ground_truth,
                                scores=if_scores,    rw=rw)
        lof_stats   = evaluate("LOF",              lof_pred,   ground_truth,
                                scores=lof_scores,   rw=rw)
        svm_stats   = evaluate("One-Class SVM",    svm_pred,   ground_truth,
                                scores=svm_scores,   rw=rw)
        ocpca_stats = evaluate("OC-PCA",           ocpca_pred, ground_truth,
                                scores=ocpca_scores, rw=rw)
        gods_stats  = evaluate("GODS",             gods_pred,  ground_truth,
                                scores=gods_scores,
                                higher_is_anomaly=gods_higher_is_anomaly, rw=rw)
        # kgods_stats = evaluate("KGODS",            kgods_pred, ground_truth,
        #                         scores=kgods_scores, rw=rw)
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
        "if_pred"       : if_pred,
        "if_scores"     : if_scores,
        "if_stats"      : if_stats,
        "lof_pred"      : lof_pred,
        "lof_scores"    : lof_scores,
        "lof_stats"     : lof_stats,
        "svm_pred"      : svm_pred,
        "svm_scores"    : svm_scores,
        "svm_stats"     : svm_stats,
        "ocpca_pred"    : ocpca_pred,
        "ocpca_scores"  : ocpca_scores,
        "ocpca_stats"   : ocpca_stats,
        "gods_pred"     : gods_pred,
        "gods_scores"   : gods_scores,
        "gods_stats"    : gods_stats,
        # "kgods_pred"    : kgods_pred,
        # "kgods_scores"  : kgods_scores,
        # "kgods_stats"   : kgods_stats,
        "timings"       : timer.entries(),
        "linux_time"    : {"real": real, "user": user, "sys": sys_},
    }


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mass spec binning + outlier detection with optional "
                    "ground-truth evaluation."
    )
    parser.add_argument("paths", nargs="+",
                        help="Directories or .mgf files to load (mix freely).")
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
    parser.add_argument("--gods-subspaces",     type=int,   default=5,
                        help="Number of hyperplane pairs for GODS (default 5).")
    parser.add_argument("--gods-eta",           type=float, default=0.01,
                        help="GODS margin parameter eta (default 0.01).")
    parser.add_argument("--gods-lam",           type=float, default=0.1,
                        help="GODS regularisation lambda (default 0.1).")
    parser.add_argument("--gods-subsample",     type=int,   default=20_000,
                        help="Training set size for GODS (default 20,000).")
    parser.add_argument("--gods-higher-is-anomaly", action="store_true",
                        help="Set if gods_wrapper.run_gods returns scores "
                             "where HIGHER = more anomalous (flip the default "
                             "'higher = more normal' convention used for "
                             "AUROC/AUPRC). Check gods_wrapper before using.")
    # KGODS
    parser.add_argument("--kgods-kernel",       type=str,   default="rbf",
                        choices=["rbf", "linear", "chisq", "min"],
                        help="Kernel for KGODS (default rbf).")
    parser.add_argument("--kgods-sigma",        type=float, default=0.01,
                        help="Kernel bandwidth sigma for KGODS (default 0.01).")
    parser.add_argument("--kgods-subspaces",    type=int,   default=1,
                        help="Number of subspaces for KGODS (default 1).")
    parser.add_argument("--kgods-subsample",    type=int,   default=5_000,
                        help="Training set size for KGODS (default 5,000). "
                             "Keep ≤8,000 — builds an n×n kernel matrix.")
    parser.add_argument("--output-file",        type=str,   default="results.txt")

    args = parser.parse_args()

    svm_gamma = args.svm_gamma
    try:
        svm_gamma = float(svm_gamma)
    except ValueError:
        pass  # keep as string ('scale' / 'auto')

    run_pipeline(
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
        output_file=args.output_file,
    )