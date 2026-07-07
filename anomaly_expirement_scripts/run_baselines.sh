#!/bin/bash
ks=(0.01 0.05 0.1 0.2 0.4 0.6 0.8 1.0)

for k in "${ks[@]}"; do
    contamination=$(python -c "print(f'{float(\"${k}\") / 100:.6f}')")

    # strip trailing .0 so 1.0 → 1 to match filenames like anomalies_n1300_pct1.mgf
    k_str=$(python -c "v=float('${k}'); print(int(v) if v == int(v) else '${k}')")

    echo "Running baseline k=${k}%  contamination=${contamination}"

    anomaly_files=( /hdd/data/fahmed/battery_mgf_files/anomalies_inorganic/anomalies_n*_pct${k_str}.mgf )

    if [ ! -f "${anomaly_files[0]}" ]; then
        echo "  [WARN] No anomaly files matched for k=${k} (pattern pct${k_str}), skipping"
        continue
    fi

    echo "  Matched ${#anomaly_files[@]} anomaly file(s):"
    for f in "${anomaly_files[@]}"; do echo "    $f"; done

    python src/baseline_anomaly.py \
        /hdd/data/fahmed/battery_mgf_files/Gr_HC_Si_trunc \
        "${anomaly_files[@]}" \
        --anomaly-files "${anomaly_files[@]}" \
        --contamination "${contamination}" \
        --output-file "baseline_results_inorganic_lof_debug.txt"
done
