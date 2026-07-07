ks=(0.01 0.05 0.1 0.2 0.4 0.6 0.8 1)

ps=(80 85 90 95 99 100)

for p in "${ps[@]}"; do
    python src/sweep_anomaly_percentile.py "${p}"

    echo "anomaly_percentile=${p}" >> anomaly_validation_results/sweep_anomaly_threshold_results.txt
    echo "anomaly_percentile=${p}" >> anomaly_validation_results/anomaly_percentile_sweep_timing.csv
    echo "Running anomaly percentile ${p}"

    for k in "${ks[@]}"; do
        echo "Running k_${k} with anomaly_percentile=${p}"

        python src/main2.py \
            --input_paths \
            /hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc \
            /hdd/data/fahmed/battery_mgf_files/anomalies_inorganic/anomalies_n*_pct${k}.mgf \
            --output_filename output_battery.csv \
            --cpu_core_preprocess=8 \
            --cluster_alg dbscan \
            --use_gpu_cluster \
            --cluster_charges 1 2 3 \
            --eps=0.26 \
            --min_mz_range 5 \
            --min_peaks=1 \
            --mz_interval=80 \
            --anomaly_file "s${k}"

        python src/analyze_anomaly_results.py \
            "anomaly_results_s${k}.csv" \
            "anomaly_var_size_files_inorganic_trunc/anomaly_spectra_s${k}.csv" \
            "${k}" 
    done
done





# ks=(0.01 0.05 0.1 0.2 0.4 0.6 0.8 1)

# ps=(80 85 90 95 99 100)

# for p in "${ps[@]}"; do
#     python src/sweep_anomaly_percentile.py "${p}"

#     echo "anomaly_percentile=${p}" >> sweep_anomaly_threshold_results.txt
#     echo "anomaly_percentile=${p}" >> anomaly_percentile_sweep_timing.csv
#     echo "Running anomaly percentile ${p}"

#     for k in "${ks[@]}"; do
#         echo "Running k_${k} with anomaly_percentile=${p}"

#         python src/main2.py \
#             --input_paths \
#             /hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc \
#             /hdd/data/fahmed/battery_mgf_files/anomalies_inorganic/anomalies_n*_pct${k}.mgf \
#             --output_filename output_battery.csv \
#             --cpu_core_preprocess=8 \
#             --cluster_alg dbscan \
#             --use_gpu_cluster \
#             --cluster_charges 1 2 3 \
#             --eps=0.26 \
#             --min_mz_range 5 \
#             --min_peaks=1 \
#             --mz_interval=80 \
#             --anomaly_file "s${k}"

#         python src/analyze_anomaly_results.py \
#             "anomaly_results_s${k}.csv" \
#             "anomaly_var_size_files_inorganic_trunc/anomaly_spectra_s${k}.csv" \
#             "${k}" 
#     done
# done