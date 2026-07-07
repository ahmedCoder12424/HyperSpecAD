import re
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


METHODS = ["Isolation Forest", "LOF", "One-Class SVM", "OC-PCA", "GODS"]

COLORS = {
    "Hyper-Spec": "tab:blue",
    "Isolation Forest": "tab:orange",
    "LOF": "tab:green",
    "One-Class SVM": "tab:red",
    "OC-PCA": "tab:purple",
    "GODS": "tab:brown",
}


HYPER_COLUMNS = [
    "param",
    "tp_pct",
    "fp_pct",
    "detection_rate",
    "tp",
    "fp",
    "total",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
]


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def parse_baselines(text):
    results = {m: [] for m in METHODS}

    section = None
    current_contam = np.nan
    current_gt = np.nan
    current_total = np.nan

    def match_method(line):
        for m in METHODS:
            if line.startswith(m):
                return m
        return None

    # First pass: classification metrics
    for line in text.splitlines():
        line = line.strip()

        if "CLASSIFICATION METRICS" in line:
            section = "classification"
            continue
        if "TIMING BREAKDOWN" in line:
            section = "timing"
            continue

        if section != "classification":
            continue

        if (
            not line
            or line.startswith("═")
            or line.startswith("-")
            or line.startswith("Method")
        ):
            continue

        method = match_method(line)
        if method is None:
            continue

        parts = line.split()
        rest = parts[len(method.split()):]
        
        if method == "Isolation Forest":
            if len(rest) < 11:
                continue

            current_contam = to_float(rest[0])
            current_gt = to_float(rest[1])
            current_total = to_float(rest[2])

            precision = to_float(rest[3])
            recall = to_float(rest[4])
            f1 = to_float(rest[5])
            fpr = to_float(rest[6])
            auroc = to_float(rest[7])
            auprc = to_float(rest[8])
            tp = to_float(rest[9])
            fp = to_float(rest[10])

        else:
            if len(rest) < 8:
                continue

            precision = to_float(rest[0])
            recall = to_float(rest[1])
            f1 = to_float(rest[2])
            fpr = to_float(rest[3])
            auroc = to_float(rest[4])
            auprc = to_float(rest[5])
            tp = to_float(rest[6])
            fp = to_float(rest[7])

        total = current_gt
        tp_pct = (tp / total * 100) if total and not np.isnan(tp) else np.nan
        fp_pct = (fp / total * 100) if total and not np.isnan(fp) else np.nan

        results[method].append({
            "contamination": current_contam,
            "gt": current_gt,
            "total": total,
            "precision": precision,
            "recall": recall,
            "detection_rate": recall,
            "f1": f1,
            "fpr": fpr,
            "auroc": auroc,
            "auprc": auprc,
            "tp": tp,
            "fp": fp,
            "tp_pct": tp_pct,
            "fp_pct": fp_pct,
            "time": np.nan,
        })

    # Second pass: timing breakdown
    section = None
    timing_idx = {m: 0 for m in METHODS}
    current_contam = np.nan

    for line in text.splitlines():
        line = line.strip()

        if "TIMING BREAKDOWN" in line:
            section = "timing"
            continue

        if section != "timing":
            continue

        if (
            not line
            or line.startswith("═")
            or line.startswith("-")
            or line.startswith("Method")
        ):
            continue

        method = match_method(line)
        if method is None:
            continue

        parts = line.split()
        rest = parts[len(method.split()):]

        if method == "Isolation Forest":
            if len(rest) < 5:
                continue

            current_contam = to_float(rest[0])
            subtotal = to_float(rest[-1])

        else:
            subtotal = to_float(rest[-1])

        idx = timing_idx[method]

        if idx < len(results[method]):
            results[method][idx]["time"] = subtotal
            timing_idx[method] += 1

    return results

def parse_timing(text):
    thresholds = []
    times = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = re.match(r"s?([\d.]+)\s*,\s*([\d.]+)", line)
        if m:
            thresholds.append(float(m.group(1)))
            times.append(float(m.group(2)))

    return np.array(thresholds), np.array(times)


def parse_accuracy(text):
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if "=" in line:
            continue

        if line.lower().startswith("param"):
            continue

        parts = [p.strip() for p in line.split(",")]

        vals = [to_float(p) for p in parts]

        if len(vals) < 2:
            continue

        rows.append(vals)

    if not rows:
        print("Warning: no valid Hyper-Spec accuracy rows found.")
        return {"threshold": np.array([])}

    max_cols = max(len(r) for r in rows)
    arr = np.full((len(rows), max_cols), np.nan)

    for i, row in enumerate(rows):
        arr[i, :len(row)] = row

    result = {}

    for idx, col in enumerate(HYPER_COLUMNS):
        if idx < arr.shape[1]:
            result[col] = arr[:, idx]

    if "param" in result:
        result["threshold"] = result["param"]
    else:
        result["threshold"] = np.array([])

    # Scale Hyper-Spec fraction metrics to percentages
    for col in ["tp_pct", "fp_pct", "detection_rate", "precision", "recall"]:
        if col in result:
            valid = result[col][~np.isnan(result[col])]
            if len(valid) > 0 and np.nanmax(valid) <= 1.5:
                result[col] = result[col] * 100

    return result


def best_by_contamination(records, key):
    groups = {}

    for r in records:
        c = r.get("contamination", np.nan)
        v = r.get(key, np.nan)

        if np.isnan(c) or np.isnan(v):
            continue

        if c not in groups:
            groups[c] = r
        elif v > groups[c].get(key, np.nan):
            groups[c] = r

    return dict(sorted(groups.items()))


def has_valid_hyper_metric(hyper_acc, metric_key):
    if "threshold" not in hyper_acc or metric_key not in hyper_acc:
        return False

    x = hyper_acc["threshold"]
    y = hyper_acc[metric_key]

    return len(x) > 0 and len(y) > 0 and np.any(~np.isnan(y))


def plot_metric(
    baselines,
    hyper_acc,
    metric_key,
    ylabel,
    title,
    outfile,
    outdir,
    include_hyper=True,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_anything = False
    print(metric_key)
    if include_hyper and has_valid_hyper_metric(hyper_acc, metric_key):
        x = hyper_acc["threshold"]
        y = hyper_acc[metric_key]
        print("valid", y)

        mask = ~np.isnan(x) & ~np.isnan(y)

        if np.any(mask):
            ax.plot(
                x[mask],
                y[mask],
                "o-",
                label="Hyper-Spec",
                color=COLORS["Hyper-Spec"],
            )
            plotted_anything = True

    for method in METHODS:
        groups = best_by_contamination(baselines[method], metric_key)
        if not groups:
            continue

        xs = np.array(list(groups.keys()))
        ys = np.array([groups[x][metric_key] for x in xs])

        mask = ~np.isnan(xs) & ~np.isnan(ys)

        if np.any(mask):
            ax.plot(
                xs[mask],
                ys[mask],
                "o--",
                label=method,
                color=COLORS[method],
            )
            plotted_anything = True

    if not plotted_anything:
        plt.close(fig)
        print(f"Skipped {outfile}: no valid data.")
        return

    ax.set_xlabel("Threshold / contamination level")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(outdir, outfile)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_timing(baselines, hyper_thresh, hyper_time, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_anything = False

    if len(hyper_thresh) > 0 and len(hyper_time) > 0:
        mask = ~np.isnan(hyper_thresh) & ~np.isnan(hyper_time)

        if np.any(mask):
            ax.plot(
                hyper_thresh[mask],
                hyper_time[mask],
                "o-",
                label="Hyper-Spec",
                color=COLORS["Hyper-Spec"],
            )
            plotted_anything = True

    for method in METHODS:
        xs = []
        ys = []

        for r in baselines[method]:
            c = r.get("contamination", np.nan)
            t = r.get("time", np.nan)

            if np.isnan(c) or np.isnan(t):
                continue

            xs.append(c)
            ys.append(t)

        if xs:
            ax.plot(
                xs,
                ys,
                "o--",
                label=method,
                color=COLORS[method],
            )
            plotted_anything = True

    if not plotted_anything:
        plt.close(fig)
        print("Skipped timing_comparison.png: no valid timing data.")
        return

    ax.set_xlabel("Threshold / contamination level")
    ax.set_ylabel("Time (s)")
    ax.set_title("Timing Comparison: Hyper-Spec vs Baselines")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(outdir, "timing_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate metric plots from baseline and Hyper-Spec result files."
    )
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--timing", required=True)
    parser.add_argument("--accuracy", required=True)
    parser.add_argument("--outdir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.baselines) as f:
        baseline_text = f.read()

    with open(args.timing) as f:
        timing_text = f.read()

    with open(args.accuracy) as f:
        accuracy_text = f.read()

    baselines = parse_baselines(baseline_text)
    hyper_thresh, hyper_time = parse_timing(timing_text)
    hyper_acc = parse_accuracy(accuracy_text)

    plot_timing(baselines, hyper_thresh, hyper_time, args.outdir)

    metric_specs = [
        ("precision", "Precision (%)", "Precision: Hyper-Spec vs Baselines", "precision_pct.png"),
        ("recall", "Recall (%)", "Recall: Hyper-Spec vs Baselines", "recall_pct.png"),
        ("detection_rate", "Detection rate (%)", "Detection Rate: Hyper-Spec vs Baselines", "detection_rate_pct.png"),
        ("f1", "F1 Score", "F1 Score: Hyper-Spec vs Baselines", "f1_score.png"),
        ("tp_pct", "True positive (%)", "True Positive Percentage: Hyper-Spec vs Baselines", "true_positive_pct.png"),
        ("fp_pct", "False positive (%)", "False Positive Percentage: Hyper-Spec vs Baselines", "false_positive_pct.png"),
        ("auroc", "AUROC", "AUROC: Hyper-Spec vs Baselines", "auroc.png"),
        ("auprc", "AUPRC", "AUPRC: Hyper-Spec vs Baselines", "auprc.png"),
    ]

    for metric_key, ylabel, title, outfile in metric_specs:
        plot_metric(
            baselines=baselines,
            hyper_acc=hyper_acc,
            metric_key=metric_key,
            ylabel=ylabel,
            title=title,
            outfile=outfile,
            outdir=args.outdir,
            include_hyper=True,
        )

    print(f"\nDone. Plots saved to: {args.outdir}")


if __name__ == "__main__":
    main()