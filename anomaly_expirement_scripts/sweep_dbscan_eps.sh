


#!/bin/bash



#for i in 0.2 0.25 0.3 0.35 0.4 0.5 0.55 0.6 0.65 0.7 0.75 0.80 0.85 0.9 0.95; do
for i in 0.39 0.391 0.392 0.393 0.394 0.395 0.396 0.397 0.398 0.399 0.4; do
  echo "Iteration $i"
  # Replace 'another_command' with the actual command
  python src/main.py 1468dataset/ louvain_dbscan_stabilization/dbscan_baseline_$i.csv     --cpu_core_preprocess=8     --cluster_alg dbscan     --use_gpu_cluster    --cluster_charges 2 3  --eps=$i
done
