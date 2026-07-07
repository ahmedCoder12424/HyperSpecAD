#!/bin/bash

#ython create_anomaly_datasets.py
#ks=(1 2 3 4 6 8 10 15 20 25 30 35)
#ks=(1 2 3 4 5 6 15 23)

ks=(0.01 0.05 0.1 0.2 0.4 0.6 0.8 1)
ks=(1)
#/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_split_mgf \
# #nomalies_n*_pct${k}.mgf \
for k in "${ks[@]}"; do
    echo "Running k_${k}"

    start=$(date +%s.%N)
    python src/main2.py \
	 --input_paths \
    "/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc" \
    /hdd/data/fahmed/battery_mgf_files/anomalies_proteomics_Gr_HC_Si_trunc/anomalies_n*_pct${k}_pmadj.mgf \
  --output_filename output_battery.csv \
  --cpu_core_preprocess=8 \
  --cluster_alg dbscan \
  --use_gpu_cluster \
  --cluster_charges 1 2 3 \
  --eps=0.26 \
  --min_mz_range 5 \
  --min_peaks=1 \
  --mz_interval=80 \
  --anomaly_file "s${k}"\
  --anomaly_path "/hdd/data/fahmed/anomaly_var_size_files_Gr_HC_Si_trunc_prot_2"\
  --anomaly_eps_percentile=0.90
  end=$(date +%s.%N)
  runtime=$(echo "$end - $start" | bc)

   python src/analyze_anomaly_results.py \
    "cluster_result/anomaly_results_s${k}.csv" \
    "/hdd/data/fahmed/anomaly_var_size_files_Gr_HC_Si_trunc_prot_2/anomaly_spectra_s${k}.csv" \
    "cluster_result/cluster_results_s${k}.csv" \
    "anomaly_validation_results/sweep_anomaly_threshold_results_Gr_HC_Si_trunc_new_metrics_proteomics.txt" \
    "${k}"
done





