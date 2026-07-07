#!/bin/bash

python create_anomaly_datasets.py
#ks=(1 2 3 4 6 8 10 15 20 25 30 35)
ks=(1 2 3 4 5 6 15 23)

for k in "${ks[@]}"; do
    echo "Running k_${k}"

    python src/main2.py /hdd/data/sumukh/raw-ms-dataset/PXD001468/   output_1511.csv \
        --cpu_core_preprocess=8 \
        --cluster_alg dbscan \
        --use_gpu_cluster \
        --cluster_charges 2 3 \
        --eps=0.05 \
        --anomaly_file "s${k}" \
	--checkpoint 1468
        

    python src/analyze_anomaly_results.py \
        "anomaly_results_s${k}.csv" \
        "anomaly_size_files/anomaly_spectra_s${k}.csv" \
        "anomaly_validation_results/sweep_anomaly_threshold_results_Gr_HC_Si_trunc_new_metrics_all.txt" \
	"${k}"
done
