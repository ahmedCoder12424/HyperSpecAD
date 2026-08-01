
"""
generate_plots_with_tables.py
===============
This scripts generates plots and a summary table for the different baselines and hyper-spec results. It creates graphs comparing
timing, True Positive percentage, False Postive Percentage, Detection Rate, F1 Score, Recall, Precision.
"""
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
        f1_pct = f1 * 100 if not np.isnan(f1) else np.nan

        results[method].append({
            "contamination": current_contam,
            "gt": current_gt,
            "total": total,
            "precision": precision,
            "recall": recall,
            "detection_rate": recall,
            "f1": f1_pct,
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
        # Strip whitespace and any trailing line-continuation backslash
        line = line.strip().rstrip("\\").strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue

        thresh_str = parts[0]
        if thresh_str.lower().startswith("s"):
            thresh_str = thresh_str[1:]

        threshold = to_float(thresh_str)
        # New file structure has multiple timing columns per row
        # (e.g. "threshold,time1,time2,time3"); only use the first one.
        time = to_float(parts[1])

        if np.isnan(threshold) or np.isnan(time):
            continue

        thresholds.append(threshold)
        times.append(time)

    thresholds = np.array(thresholds)
    times = np.array(times)

    if len(thresholds) > 0:
        order = np.argsort(thresholds)
        thresholds = thresholds[order]
        times = times[order]

    return thresholds, times


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
    for col in ["tp_pct", "fp_pct", "detection_rate", "precision", "recall", "f1"]:
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

    ax.set_xlabel("Anomaly Contamination level")
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
    ax.set_yscale('log')
    ax.set_xlabel("Anomaly Contamination level")
    ax.set_ylabel("Time (s)")
    ax.set_title("Timing Comparison: Hyper-Spec vs Baselines")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(outdir, "timing_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


TABLE_METRIC_KEYS = ["precision", "recall", "f1"]

TABLE_COL_LABELS = [
    "Method",
    "Precision (%)",
    "Recall (%)",
    "F1 (%)",
    "Time (s)",
]

# Whether a higher value is "better" for each metric column (used for bolding
# the best entry per column). Time and FP% are lower-is-better.
LOWER_IS_BETTER = {"fp_pct", "time"}

# Methods excluded from the summary table specifically (still used elsewhere,
# e.g. the per-metric line plots).
TABLE_EXCLUDE_METHODS = {"GODS"}


def _nanmean(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(np.mean(values)) if len(values) else np.nan


def summarize_baseline_records(records, keys):
    """Average each metric in `keys` across all rows for one baseline method."""
    return {k: _nanmean([r.get(k, np.nan) for r in records]) for k in keys}


def summarize_hyper_acc(hyper_acc, hyper_time, keys):
    """Average each metric in `keys` across all Hyper-Spec threshold rows."""
    out = {}
    for k in keys:
        out[k] = _nanmean(hyper_acc[k]) if k in hyper_acc else np.nan
    out["time"] = _nanmean(hyper_time)
    return out


def format_cell(value, key):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if key in ("tp_pct", "fp_pct", "precision", "recall", "f1"):
        return f"{value:.1f}"
    if key == "time":
        return f"{value:.1f}"
    return f"{value:.3f}"


def build_summary_rows(baselines, hyper_acc, hyper_time):
    """Returns list of (method_name, {metric_key: value}) tuples, Hyper-Spec first."""
    keys = TABLE_METRIC_KEYS
    rows = [("Hyper-Spec", summarize_hyper_acc(hyper_acc, hyper_time, keys))]

    for method in METHODS:
        if method in TABLE_EXCLUDE_METHODS:
            continue
        if not baselines.get(method):
            continue
        rows.append((method, summarize_baseline_records(baselines[method], keys + ["time"])))

    return rows


def plot_summary_table(baselines, hyper_acc, hyper_time, outdir, outfile="summary_table.png"):
    rows = build_summary_rows(baselines, hyper_acc, hyper_time)

    if len(rows) <= 1:
        print(f"Skipped {outfile}: no baseline data to compare.")
        return

    all_keys = TABLE_METRIC_KEYS + ["time"]

    # Determine best value per column for bolding/highlighting.
    best_per_col = {}
    for key in all_keys:
        vals = [r[1].get(key, np.nan) for r in rows]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            best_per_col[key] = None
            continue
        best_per_col[key] = min(vals) if key in LOWER_IS_BETTER else max(vals)

    table_data = []
    for name, metrics in rows:
        row_cells = [name] + [format_cell(metrics.get(k, np.nan), k) for k in all_keys]
        table_data.append(row_cells)

    n_rows = len(table_data)
    fig_height = 0.55 * (n_rows + 1) + 0.35
    fig, ax = plt.subplots(figsize=(11, fig_height))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.axis("off")
  

    table = ax.table(
        cellText=table_data,
        colLabels=TABLE_COL_LABELS,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.auto_set_column_width(col=list(range(len(TABLE_COL_LABELS))))

    for cell in table.get_celld().values():
        cell.PAD = 0.05

    header_color = "#2c3e50"
    hyper_row_color = "#aed4f5"
    hyper_border_color = "#1f6fb2"
    alt_row_color = "#f4f4f4"

    n_cols = len(TABLE_COL_LABELS)

    for col in range(n_cols):
        cell = table[(0, col)]
        cell.set_facecolor(header_color)
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")

    for row_idx, (name, metrics) in enumerate(rows, start=1):
        is_hyper = name == "Hyper-Spec"
        row_color = hyper_row_color if is_hyper else (
            alt_row_color if row_idx % 2 == 0 else "white"
        )

        for col in range(n_cols):
            cell = table[(row_idx, col)]
            cell.set_facecolor(row_color)

            if is_hyper:
                cell.set_edgecolor(hyper_border_color)
                cell.set_linewidth(2.2)
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_fontsize(15)

            if col == 0:
                cell.get_text().set_fontweight("bold")
                continue

            key = all_keys[col - 1]
            value = metrics.get(key, np.nan)
            best = best_per_col.get(key)

            if best is not None and not np.isnan(value) and np.isclose(value, best):
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_color("#1a7a1a")

    fig.tight_layout(pad=0.2)
    path = os.path.join(outdir, outfile)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.03, transparent=True)
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
        ("f1", "F1 Score (%)", "F1 Score: Hyper-Spec vs Baselines", "f1_score.png"),
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

    plot_summary_table(baselines, hyper_acc, hyper_time, args.outdir)

    print(f"\nDone. Plots saved to: {args.outdir}")


if __name__ == "__main__":
    main()