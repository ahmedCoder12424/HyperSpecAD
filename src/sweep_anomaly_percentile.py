# set_anomaly_percentile.py

import sys
import re

if len(sys.argv) != 2:
    print("Usage: python set_anomaly_percentile.py <percentile>")
    sys.exit(1)

percentile = sys.argv[1]

with open("src/hd_cluster.py") as f:
    text = f.read()

text, n = re.subn(
    r"ANOMALY_EPS_PERCENTILE\s*=\s*[\d.]+",
    f"ANOMALY_EPS_PERCENTILE = {percentile}",
    text
)

if n == 0:
    raise RuntimeError("Couldn't find ANOMALY_EPS_PERCENTILE")

with open("src/hd_cluster.py", "w") as f:
    f.write(text)

print(f"Set ANOMALY_EPS_PERCENTILE = {percentile}")