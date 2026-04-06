import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
import os
from src.spine import TheSpine
from train_phase_1 import get_batch

def diagnose():
    d_model = 32
    d_state = 64
    seq_len = 20
    model_path = "models/spine_model.pth"
    
    model = TheSpine(d_model, d_state)
    
    if os.path.exists(model_path):
        print(f"Lade trainiertes Modell von {model_path}...")
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"WARNUNG: Keine Modelldatei gefunden ({model_path}). Benutze untrainierte Gewichte!")

    model.eval()
    
    # 2. Diagnose-Lauf (Ein einzelnes Sample)
    print("\n--- DIAGNOSE-PROTOKOLL (Phase 1.6: Energy-Aware) ---")
    print(f"{'Schritt':<8} | {'Gate (g_t)':<12} | {'Skip-Gate':<12} | {'State Norm':<12}")
    print("-" * 55)
    
    with torch.no_grad():
        x_test, target_test = get_batch(1, seq_len, d_model)
        h = None
        g_key = 0
        g_noise_sum = 0
        
        for t in range(seq_len):
            y, h, gate = model(x_test[t], h)
            
            # Gating-Werte sammeln
            g_val = gate.item()
            # Der Skip-Gate-Logit wird in der spine.py über model.skip_gate(x) + model.skip_gate.bias berechnet (bzw. Linear Layer)
            # Da in forward() das skip_gate genutzt wird (oder sein sollte):
            skip_val = torch.sigmoid(model.skip_gate(x_test[t])).item()
            s_norm = torch.norm(h).item()
            
            if t == 0:
                g_key = g_val
            else:
                g_noise_sum += g_val
                
            label = "KEY" if t == 0 else "NOISE"
            print(f"{t:02d} ({label:<5}) | {g_val:.4f}       | {skip_val:.4f}       | {s_norm:.4f}")

    g_noise_avg = g_noise_sum / (seq_len - 1)
    delta_g = g_key - g_noise_avg
    error = F.mse_loss(y, target_test).item()
    
    print("-" * 55)
    print(f"Gate-Trennschärfe (Δg): {delta_g:.4f}")
    print(f"Finaler Rekonstruktions-Fehler: {error:.6f}")
    
    if delta_g > 0.5:
        print("STATUS: Starke Signal-Selektion aktiv.")
    else:
        print("STATUS: Selektion noch zu schwach.")

if __name__ == "__main__":
    diagnose()
