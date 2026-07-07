#!/bin/bash
ks=(0.01 0.05 0.1 0.2 0.4 0.6 0.8 1.0)


# --- Run configs ---

INPUT_DIR_run1="/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc"
ANOMALY_DIR_run1="/hdd/data/fahmed/battery_mgf_files/anomalies_inorganic"
OUTPUT_FILE_run1="baseline_results/baseline_results_inorganic_Gr_HC_Si_trunc_metrics.txt"
RUN_TAG_run1="Gr_HC_Si_trunc_inorganic_anom"


INPUT_DIR_run2="/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc"
ANOMALY_DIR_run2="/hdd/data/fahmed/battery_mgf_files/anomalies_proteomics_Gr_HC_Si_trunc"
OUTPUT_FILE_run2="baseline_results/baseline_results_organic_Gr_HC_Si_trunc_metrics.txt"
RUN_TAG_run2="Gr_HC_Si_trunc_organic_anom"

#INPUT_DIR_run1="/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_one layer_negative_DE_2048pixels_rotated_after FIB_2_mgf_split"
#ANOMALY_DIR_run1="/hdd/data/fahmed/battery_mgf_files/anomalies_inorganic_Gr_HC_Si_split_mgf_clean"
#OUTPUT_FILE_run1="baseline_results_inorganic_Gr_HC_Si_full.txt"
#RUN_TAG_run1="Gr_HC_Si"

# INPUT_DIR_run2="/hdd/data/fahmed/battery_mgf_files/Pre_30_Ref_DE_250_2040_5shots_1frame_FIBlong_FIBpolish_1_mgf_split"
# ANOMALY_DIR_run2="/hdd/data/fahmed/battery_mgf_files/anomalies_inorganic_Pre_30_Ref_DE_split_mgf"
# OUTPUT_FILE_run2="baseline_results_inorganic_Pre_30.txt"
# RUN_TAG_run2="Pre_30_Ref"



# INPUT_DIR_run2="/hdd/data/fahmed/battery_mgf_files/Cycled_graphite_9014_DE_2048_250_after SE cleaning_1_mgf_split"
# ANOMALY_DIR_run2="/hdd/data/fahmed/battery_mgf_files/anomalies_inorganic_Cycled_graphite_split_mgf"
# OUTPUT_FILE_run2="baseline_results_inorganic_Cycled_graphite.txt"
# RUN_TAG_run2="Cycled_graphite"

# -----------------------

for run in run1; do
    input_var="INPUT_DIR_${run}"
    anomaly_var="ANOMALY_DIR_${run}"
    output_var="OUTPUT_FILE_${run}"
    tag_var="RUN_TAG_${run}"

    INPUT_DIR="${!input_var}"
    ANOMALY_DIR="${!anomaly_var}"
    OUTPUT_FILE="${!output_var}"
    RUN_TAG="${!tag_var}"

    echo "=== Run: ${RUN_TAG} ==="
    for k in "${ks[@]}"; do
        contamination=$(python -c "print(f'{float(\"${k}\") / 100:.6f}')")
        k_str=$(python -c "v=float('${k}'); print(int(v) if v == int(v) else '${k}')")
        echo "Running baseline k=${k}%  contamination=${contamination}"

        anomaly_files=( ${ANOMALY_DIR}/anomalies_n*_pct${k_str}.mgf )
        if [ ! -f "${anomaly_files[0]}" ]; then
            echo "  [WARN] No anomaly files matched for k=${k} (pattern pct${k_str}), skipping"
            continue
        fi
        echo "  Matched ${#anomaly_files[@]} anomaly file(s):"
        for f in "${anomaly_files[@]}"; do echo "    $f"; done

        python src/baseline_anomaly_full.py \
            "${INPUT_DIR}" \
            "${anomaly_files[@]}" \
            --anomaly-files "${anomaly_files[@]}" \
            --contamination "${contamination}" \
             --skip gods kgods \
            --output-file "${OUTPUT_FILE}"
    done
done
