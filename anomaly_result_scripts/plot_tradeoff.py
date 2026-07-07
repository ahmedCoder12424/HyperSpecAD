#!/usr/bin/env python3
"""
Bucket width (mz_interval) tradeoff plots:
  - Accuracy vs bucket width
  - Timing vs bucket width
  - Combined (dual-axis) view

Reads:
  accuracy_raw.txt  -> blocks of "mz_interval=X, k=..." followed by rows:
                       contamination, recall, precision, ACCURACY, tp, fp, total
  timing_raw.txt    -> blocks of "mz_interval=X, k=..." followed by rows:
                       sCONTAMINATION,seconds

Outputs (PNG, saved next to this script):
  accuracy_vs_bucket_width.png
  timing_vs_bucket_width.png
  combined_tradeoff.png
"""

import re
import os
from collections import defaultdict

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ACC_FILE = os.path.join(HERE, "accuracy_raw.txt")
TIME_FILE = os.path.join(HERE, "timing_raw.txt")

HEADER_RE = re.compile(r"mz_interval\s*=\s*([0-9.]+)")


def parse_accuracy(path):
    """Returns dict: contamination_level -> list of (mz_interval, accuracy) sorted by mz_interval."""
    data = defaultdict(list)
    current_mz = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = HEADER_RE.search(line)
            if m:
                current_mz = float(m.group(1))
                continue
            parts = [p.strip() for p in line.split(",")]
            if current_mz is None or len(parts) < 4:
                continue
            contamination = float(parts[0])
            accuracy = float(parts[3])
            data[contamination].append((current_mz, accuracy))
    for k in data:
        data[k].sort(key=lambda t: t[0])
    return data


def parse_timing(path):
    """Returns dict: contamination_level -> list of (mz_interval, seconds) sorted by mz_interval."""
    data = defaultdict(list)
    current_mz = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = HEADER_RE.search(line)
            if m:
                current_mz = float(m.group(1))
                continue
            m2 = re.match(r"s([0-9.]+)\s*,\s*([0-9.]+)", line)
            if m2 and current_mz is not None:
                contamination = float(m2.group(1))
                seconds = float(m2.group(2))
                data[contamination].append((current_mz, seconds))
    for k in data:
        data[k].sort(key=lambda t: t[0])
    return data


def get_colors(n):
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def plot_accuracy(acc_data, outpath):
    fig, ax = plt.subplots(figsize=(9, 6))
    levels = sorted(acc_data.keys())
    colors = get_colors(len(levels))
    for color, level in zip(colors, levels):
        xs, ys = zip(*acc_data[level])
        ax.plot(xs, ys, marker="o", markersize=4, label=f"contamination={level}", color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Bucket width (mz_interval, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Bucket Width")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_timing(time_data, outpath):
    fig, ax = plt.subplots(figsize=(9, 6))
    levels = sorted(time_data.keys())
    colors = get_colors(len(levels))
    for color, level in zip(colors, levels):
        xs, ys = zip(*time_data[level])
        ax.plot(xs, ys, marker="o", markersize=4, label=f"contamination={level}", color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Bucket width (mz_interval, log scale)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Timing vs. Bucket Width")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def average_over_levels(data):
    """Given dict[level] -> [(mz, val), ...], average val across levels for each mz."""
    by_mz = defaultdict(list)
    for level, points in data.items():
        for mz, val in points:
            by_mz[mz].append(val)
    mzs = sorted(by_mz.keys())
    avg = [sum(by_mz[mz]) / len(by_mz[mz]) for mz in mzs]
    return mzs, avg


def plot_combined(acc_data, time_data, outpath):
    mz_acc, avg_acc = average_over_levels(acc_data)
    mz_time, avg_time = average_over_levels(time_data)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color_acc = "tab:blue"
    ax1.set_xlabel("Bucket width (mz_interval, log scale)")
    ax1.set_ylabel("Accuracy (avg across contamination levels)", color=color_acc)
    ax1.plot(mz_acc, avg_acc, marker="o", color=color_acc, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xscale("log")
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    color_time = "tab:red"
    ax2.set_ylabel("Runtime in seconds (avg across contamination levels)", color=color_time)
    ax2.plot(mz_time, avg_time, marker="s", color=color_time, label="Timing")
    ax2.tick_params(axis="y", labelcolor=color_time)

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Accuracy vs. Timing Tradeoff Across Bucket Width")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    acc_data = parse_accuracy(ACC_FILE)
    time_data = parse_timing(TIME_FILE)

    plot_accuracy(acc_data, os.path.join(HERE, "accuracy_vs_bucket_width.png"))
    plot_timing(time_data, os.path.join(HERE, "timing_vs_bucket_width.png"))
    plot_combined(acc_data, time_data, os.path.join(HERE, "combined_tradeoff.png"))

    print("Saved:")
    print(" -", os.path.join(HERE, "accuracy_vs_bucket_width.png"))
    print(" -", os.path.join(HERE, "timing_vs_bucket_width.png"))
    print(" -", os.path.join(HERE, "combined_tradeoff.png"))


if __name__ == "__main__":
    main()