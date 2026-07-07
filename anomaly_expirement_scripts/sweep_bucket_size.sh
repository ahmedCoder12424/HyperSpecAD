
ks=(0.01 0.05 0.1 0.2 0.4 0.6 0.8 1)
#/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_split_mgf \
# #nomalies_n*_pct${k}.mgf \
is=(0.0001 0.001 0.01 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 500 1000)
for i in "${is[@]}"; do
    echo "mz_interval=${i}, k=${k}" >> big_bucket_sweep_accuracy_results_opt.txt
    echo "mz_interval=${i}, k=${k}" >> inorganic_timing_big_bucket_sweep_opt.csv
    echo "running bucket interval width ${i}"
    for k in "${ks[@]}"; do
    	echo "Running k_${k} with bucket interval width ${i}"

    	python src/main2.py \
        	 --input_paths \
    	/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc\
    	/hdd/data/fahmed/battery_mgf_files/anomalies_inorganic/anomalies_n*_pct${k}.mgf \
  	--output_filename output_battery.csv \
  	--cpu_core_preprocess=8 \
  	--cluster_alg dbscan \
  	--use_gpu_cluster \
  	--cluster_charges 1 2 3 \
  	--eps=0.26 \
  	--min_mz_range 5 \
  	--min_peaks=1 \
  	--mz_interval=${i} \
  	--anomaly_file "s${k}" 


    	python src/analyze_anomaly_results.py \
        	"anomaly_results_s${k}.csv" \
        	"anomaly_var_size_files_inorganic_trunc/anomaly_spectra_s${k}.csv" \
        	"${k}"
	done
done
