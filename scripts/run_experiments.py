import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import subprocess
import re

biases = [-3.1, -3.15, -3.2, -3.25, -3.3]
lambda_keys = [0.02]
results = []

file_path = "src/spine.py"

# Originalinhalt lesen
with open(file_path, "r") as f:
    original_content = f.read()

print(f"{'Bias':<10} | {'Lambda_K':<10} | {'MSE (Train)':<15} | {'Delta g':<10}")
print("-" * 55)

try:
    for bias in biases:
        # Datei anpassen für Bias
        new_content = re.sub(
            r"self\.gate_bias = nn\.Parameter\(torch\.tensor\(-?\d+\.?\d*\)\)",
            f"self.gate_bias = nn.Parameter(torch.tensor({bias}))",
            original_content
        )
        with open(file_path, "w") as f:
            f.write(new_content)
        
        for lk in lambda_keys:
            # Training ausführen mit aktuellem lambda_key
            train_cmd = ["python3", "scripts/train_phase_1.py", "--lambda_key", str(lk)]
            train_out = subprocess.check_output(train_cmd, stderr=subprocess.STDOUT, text=True)
            mse_match = re.search(r"Finaler Rekonstruktions-Fehler \(MSE\): ([\d\.]+)", train_out)
            mse = mse_match.group(1) if mse_match else "N/A"
            
            # Diagnose ausführen
            diag_out = subprocess.check_output(["python3", "scripts/diagnose_spine.py"], stderr=subprocess.STDOUT, text=True)
            dg_match = re.search(r"Gate-Trennschärfe \(Δg\): ([\d\.\-]+)", diag_out)
            dg = dg_match.group(1) if dg_match else "N/A"
            
            print(f"{bias:<10} | {lk:<10} | {mse:<15} | {dg:<10}")
            results.append((bias, lk, mse, dg))

finally:
    # Original wiederherstellen
    with open(file_path, "w") as f:
        f.write(original_content)

print("\nGrid Search abgeschlossen.")
