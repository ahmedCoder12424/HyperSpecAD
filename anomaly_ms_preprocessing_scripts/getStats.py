import argparse
from pathlib import Path
from statistics import mean, median
import matplotlib.pyplot as plt
import math


def parse_mgf(filepath):
    spectra = []
    current_peaks = []
    in_spectrum = False

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line == "BEGIN IONS":
                in_spectrum = True
                current_peaks = []
                continue

            if line == "END IONS":
                if in_spectrum:
                    spectra.append(current_peaks)
                in_spectrum = False
                continue

            if in_spectrum and "=" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        current_peaks.append((mz, intensity))
                    except ValueError:
                        pass

    return spectra


def compute_dataset_stats(spectra):
    peak_counts = [len(spectrum) for spectrum in spectra]
    all_intensities = [intensity for spectrum in spectra for _, intensity in spectrum]

    if not spectra:
        return None

    return {
        "num_spectra": len(spectra),
        "total_peaks": sum(peak_counts),

        "avg_peaks_per_spectrum": mean(peak_counts),
        "median_peaks_per_spectrum": median(peak_counts),
        "min_peaks_per_spectrum": min(peak_counts),
        "max_peaks_per_spectrum": max(peak_counts),

        "avg_peak_intensity": mean(all_intensities) if all_intensities else 0,
        "median_peak_intensity": median(all_intensities) if all_intensities else 0,
        "min_peak_intensity": min(all_intensities) if all_intensities else 0,
        "max_peak_intensity": max(all_intensities) if all_intensities else 0,
    }


def plot_distributions(spectra, prefix="mgf_stats"):
    peak_counts = [len(spectrum) for spectrum in spectra]
    all_intensities = [intensity for spectrum in spectra for _, intensity in spectrum]

    if not peak_counts:
        print("No spectra available for plotting.")
        return

    # Peak count distribution
    plt.figure(figsize=(8, 5))
    plt.hist(peak_counts, bins=30, edgecolor="black")
    plt.title("Distribution of Peak Counts per Spectrum")
    plt.xlabel("Number of Peaks")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{prefix}_peak_counts.png", dpi=300)
    plt.show()
    plt.close()

    # Peak intensity distribution
    if all_intensities:
        plt.figure(figsize=(8, 5))
        plt.hist(all_intensities, bins=50, edgecolor="black")
        plt.title("Distribution of Peak Intensities")
        plt.xlabel("Peak Intensity")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{prefix}_peak_intensities.png", dpi=300)
        plt.show()
        plt.close()


        log_intensities = [math.log10(x) for x in all_intensities if x > 0]

        plt.figure(figsize=(8, 5))
        plt.hist(log_intensities, bins=50, edgecolor="black")
        plt.title("Distribution of Log10 Peak Intensities")
        plt.xlabel("log10(Peak Intensity)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{prefix}_log_peak_intensities.png", dpi=300)
        plt.show()
        plt.close()


def process_file(filepath: Path):
    spectra = parse_mgf(filepath)
    stats = compute_dataset_stats(spectra)
    return spectra, stats


def main():
    parser = argparse.ArgumentParser(description="Compute peak statistics for one or more MGF files.")
    parser.add_argument("inputs", nargs="+", help="Input MGF file(s)")
    args = parser.parse_args()

    all_spectra = []

    for name in args.inputs:
        path = Path(name)
        try:
            spectra, stats = process_file(path)

            print(f"\n=== {path} ===")
            if stats is None:
                print("No spectra found.")
                continue

            for key, value in stats.items():
                print(f"{key}: {value}")

            all_spectra.extend(spectra)

        except Exception as e:
            print(f"\n=== {path} ===")
            print(f"Skipping file due to error: {e}")

    combined_stats = compute_dataset_stats(all_spectra)
    print("\n=== Combined Stats Across All Files ===")
    if combined_stats is None:
        print("No spectra found in any input files.")
    else:
        for key, value in combined_stats.items():
            print(f"{key}: {value}")

        plot_distributions(all_spectra, prefix="combined")


if __name__ == "__main__":
    main()