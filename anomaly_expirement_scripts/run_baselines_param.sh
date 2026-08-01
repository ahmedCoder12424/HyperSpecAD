#!/bin/bash
ks=(0.01 0.05 0.1 0.2 0.4 0.6 0.8 1.0)


# --- Run configs ---
INPUT_DIR_run1="/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc"
ANOMALY_DIR_run1="/hdd/data/fahmed/battery_mgf_files/anomalies_inorganic_Gr_HC_Si_trunc_correct_percent"
OUTPUT_FILE_run1="baseline_results/baseline_results_inorganic_Gr_HC_Si_trunc_metrics.txt"
RUN_TAG_run1="Gr_HC_Si_trunc_inorganic_anom"


INPUT_DIR_run2="/hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc"
ANOMALY_DIR_run2="/hdd/data/fahmed/battery_mgf_files/anomalies_proteomics_Gr_HC_Si_trunc"
OUTPUT_FILE_run2="baseline_results/baseline_results_organic_Gr_HC_Si_trunc_metrics.txt"
RUN_TAG_run2="Gr_HC_Si_trunc_organic_anom"


INPUT_DIR_run3="/hdd/data/fahmed/pigments_mgf"
ANOMALY_DIR_run3="/hdd/data/fahmed/battery_mgf_files/anomalies_organic_in_pigments"
OUTPUT_FILE_run3="baseline_results/sanity_check.txt"
RUN_TAG_run3="baselines_sanity_check"


INPUT_DIR_run4="/hdd/data/fahmed/pigments_mgf"
ANOMALY_DIR_run4="/hdd/data/fahmed/battery_mgf_files/anomalies_organic_in_pigments"
OUTPUT_FILE_run4="baseline_results/sanity_check.txt"
RUN_TAG_run4="proteomics baseline"


INPUT_DIR_run5="/hdd/data/fahmed/PXD001468"
ANOMALY_DIR_run5="/hdd/data/fahmed/battery_mgf_files/pigment_anom_in_proteomics_trunc"
OUTPUT_FILE_run5="baseline_results/proteomics_redone_correct_2_full.txt"
RUN_TAG_run5="proteomics_redone_correct_2_full"


# -----------------------

for run in run5; do
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

        anomaly_files=( ${ANOMALY_DIR}/anomalies_n*_pct${k_str}*.mgf )
        if [ ! -f "${anomaly_files[0]}" ]; then
            echo "  [WARN] No anomaly files matched for k=${k} (pattern pct${k_str}), skipping"
            continue
        fi
        echo "  Matched ${#anomaly_files[@]} anomaly file(s):"
        for f in "${anomaly_files[@]}"; do echo "    $f"; done
        echo "contamiantion"
        echo $contamination
        python src/baseline_anomaly_complete.py \
            "${INPUT_DIR}" \
            "${anomaly_files[@]}" \
            --anomaly-files "${anomaly_files[@]}" \
            --contamination "${contamination}" \
            --output-file "${OUTPUT_FILE}" \
            --skip if ocpca lof gods kgods\
            --summary-file "baseline_results/${RUN_TAG}_baseline_summary.txt"

    done
done






