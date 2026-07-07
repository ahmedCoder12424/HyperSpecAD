#!/bin/bash
# run_anomalies.sh
# Runs src/main.py for each anomaly file paired with the normal battery data.

NORMAL_DIR="/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_new_encoding_trunc"
ANOMALY_DIR="/hdd/data/fahmed/battery_mgf_files/anomalies_out_pct_2"
OUTPUT_DIR="/hdd/data/anomaly_inorganic_outputs_Gr_HC_Si_split_mgf_new_enc_trunc"

mkdir -p "$OUTPUT_DIR"

for anomaly_file in "$ANOMALY_DIR"/*.mgf; do
    # Extract just the filename stem (e.g. anomalies_n3023_pct1.0)
    stem=$(basename "$anomaly_file" .mgf)

    echo "=================================================="
    echo "Running: $stem"
    echo "=================================================="

    python src/main.py \
        --input_paths \
            "$NORMAL_DIR" \
            "$anomaly_file" \
        --output_filename "${OUTPUT_DIR}/output_battery_${stem}.csv" \
        --cpu_core_preprocess=8 \
        --cluster_alg dbscan \
        --use_gpu_cluster \
        --cluster_charges 1 2 3\
	--mz_interval=0.01 \
        --eps=0.26 \
        --min_mz_range 5 \
        --min_peaks=1

    echo "Done: ${OUTPUT_DIR}/output_battery_${stem}.csv"
    echo ""
done

echo "All runs complete."
