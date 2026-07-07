#!/bin/bash

ks=(1 2 3 4 6 8 10 15 20 25)

for k in "${ks[@]}"; do
    echo "Running k_${k}"

    python src/main2.py /hdd/data/fahmed/battery_mgf_files/anomaly_test_mod output_1511.csv \
        --cpu_core_preprocess=8 \
        --cluster_alg dbscan \
        --use_gpu_cluster \
        --cluster_charges 1 2 3 \
        --eps=0.26 \
        --anomaly_file "k${k}" \
	--min_mz_range 5 \
	--min_peaks=1 \
	--checkpoint "Gr_Si_mod_2"

  #  python src/analyze_anomaly_results.py \
   #     "anomaly_results_k${k}.csv" \
    #    excluded_spectra_smaller.csv

    python src/analyze_anomaly_results.py \
        "anomaly_results_k${k}.csv" \
        "inorganic_anomaly_split_files_mod/anomaly_spectra_s3.csv" \
        "${k}"
done
