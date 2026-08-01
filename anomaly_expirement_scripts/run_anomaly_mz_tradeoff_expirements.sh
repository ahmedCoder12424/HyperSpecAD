ks=(0.01 0.05 0.1 0.2 0.4 0.6 0.8 1) #0.05 0.1 0.2 0.4 0.6 0.8 1
# --- Run configs ---



INPUT_DIR_run1="/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc"
ANOMALY_DIR_run1="/hdd/data/fahmed/battery_mgf_files/anomalies_inorganic"
ANOMALY_SPECTRA_DIR_run1="/hdd/data/fahmed/anomaly_var_size_files_trunc"
RUN_TAG_run1="inorganic_anomalies"

# INPUT_DIR_run1="/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc"
# ANOMALY_DIR_run1="/hdd/data/fahmed/battery_mgf_files/anomalies_inorganic_Gr_HC_Si_trunc_correct_percent"
# ANOMALY_SPECTRA_DIR_run1="/hdd/data/fahmed/anomaly_var_size_files_Gr_HC_Si_inorganic_correct_pc"
# RUN_TAG_run1="inorganic_anomalies"

INPUT_DIR_run2="/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc"
ANOMALY_DIR_run2="/hdd/data/fahmed/battery_mgf_files/anomalies_proteomics_Gr_HC_Si_trunc"
ANOMALY_SPECTRA_DIR_run2="/hdd/data/fahmed/anomaly_var_size_files_Gr_HC_Si_trunc_prot_2"
RUN_TAG_run2="proteomics_anomalies"



INPUT_DIR_run3="/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc"
ANOMALY_DIR_run3="/hdd/data/fahmed/battery_mgf_files/anomalies_inorganic_Gr_HC_Si_trunc_correct_percent"
ANOMALY_SPECTRA_DIR_run3="/hdd/data/fahmed/anomaly_var_size_files_Gr_HC_Si_inorganic_correct_pc"
RUN_TAG_run3="inorganic_anomalies correct percentages"


INPUT_DIR_run4="/hdd/data/fahmed/pigments_mgf"
ANOMALY_DIR_run4="/hdd/data/fahmed/battery_mgf_files/anomalies_organic_in_pigments"
ANOMALY_SPECTRA_DIR_run4="/hdd/data/fahmed/anomaly_var_size_files_pigments"
RUN_TAG_run4="pigments"


INPUT_DIR_run4="/hdd/data/fahmed/PXD001468"
ANOMALY_DIR_run4="/hdd/data/fahmed/battery_mgf_files/pigment_anom_in_proteomics_trunc"
ANOMALY_SPECTRA_DIR_run4="/hdd/data/fahmed/anomaly_var_size_files_pigments_proteomics_normal_trunc"
RUN_TAG_run4="proteomics"


ACCURACY_RESULT_FILE="anomaly_validation_results/proteomics_acc_3.txt"
TIMING_RESULT_FILE="anomaly_validation_results/proteomics_timing_3.csv"




# INPUT_DIR_run2="/hdd/data/fahmed/battery_mgf_files/Pre_30_Ref_DE_250_2040_5shots_1frame_FIBlong_FIBpolish_1_mgf_split"
# ANOMALY_DIR_run2="/hdd/data/fahmed/battery_mgf_files/anomalies_inorganic_Pre_30_Ref_DE_split_mgf"
# ANOMALY_SPECTRA_DIR_run2="anomaly_var_size_files_Pre_30_Ref"
# RUN_TAG_run2="Pre_30_Ref_DE_"

# INPUT_DIR_run3="/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_one layer_negative_DE_2048pixels_rotated_after FIB_2_mgf_split"
# ANOMALY_DIR_run3="/hdd/data/fahmed/battery_mgf_files/anomalies_inorganic_Gr_HC_Si_split_mgf_clean"
# ANOMALY_SPECTRA_DIR_run3="anomaly_var_size_files_Gr_HC_Si"
# RUN_TAG_run3="Gr_HC_Si"
# -----------------------

mzs=(1.0 10.0 80.0 100.0 1000.0)

for run in run4; do
    for mz in "${mzs[@]}"; do     
        input_var="INPUT_DIR_${run}"
        anomaly_var="ANOMALY_DIR_${run}"
        spectra_var="ANOMALY_SPECTRA_DIR_${run}"
        tag_var="RUN_TAG_${run}"

        INPUT_DIR="${!input_var}"
        ANOMALY_DIR="${!anomaly_var}"
        ANOMALY_SPECTRA_DIR="${!spectra_var}"
        RUN_TAG="${!tag_var}_${mz}"
        echo $RUN_TAG >> $ACCURACY_RESULT_FILE
        echo $RUN_TAG >> $TIMING_RESULT_FILE
        echo "=== Run: ${RUN_TAG} ==="
        for k in "${ks[@]}"; do
            echo "  k=${k}"
            python src/main_anomaly_detection_2.py \
                --input_paths \
                "${INPUT_DIR}" \
                ${ANOMALY_DIR}/anomalies_n*_pct${k}*.mgf \
                --output_filename "output_${RUN_TAG}.csv" \
                --cpu_core_preprocess=8 \
                --cluster_alg dbscan \
                --use_gpu_cluster \
                --cluster_charges 1 2 3 \
                --eps=0.26 \
                --min_mz_range 10 \
                --min_peaks=1 \
                --mz_interval="${mz}" \
                --anomaly_file "s${k}" \
            --anomaly_path "$ANOMALY_SPECTRA_DIR" \
                --anomaly_eps_percentile=0.9
    


            python src/analyze_anomaly_results.py \
                "cluster_result/anomaly_results_pig_s${k}.csv" \
                "${ANOMALY_SPECTRA_DIR}/anomaly_spectra_s${k}.csv" \
                "cluster_result/cluster_results_pig_s${k}.csv" \
                "${ACCURACY_RESULT_FILE}" \
                "${k}" 
        done
    done
done

