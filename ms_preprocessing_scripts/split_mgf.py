import os

input_mgf = "battery_mgf_files/Pre_30_Ref_DE_250_2040_5shots_1frame_FIBlong_FIBpolish_1.ImzML.mgf"
out_dir = "battery_mgf_files/Pre_30_Ref_mgf_med_files"
max_bytes = 100 * 1024 * 1024  # 100 MB

os.makedirs(out_dir, exist_ok=True)

part = 0
current_bytes = 0
out = open(f"{out_dir}/split_{part:04d}.mgf", "w")

with open(input_mgf, "r") as f:
    block = []
    for line in f:
        block.append(line)

        if line.strip() == "END IONS":
            block_text = "".join(block)
            block_size = len(block_text.encode("utf-8"))

            if current_bytes + block_size > max_bytes and current_bytes > 0:
                out.close()
                part += 1
                current_bytes = 0
                out = open(f"{out_dir}/split_{part:04d}.mgf", "w")

            out.write(block_text)
            current_bytes += block_size
            block = []

out.close()
print(f"Wrote {part + 1} files")
