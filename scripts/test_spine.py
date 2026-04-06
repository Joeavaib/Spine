import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.spine import TheSpine

def run_stability_test():
    d_model = 64
    d_state = 128
    seq_len = 50
    
    model = TheSpine(d_model, d_state)
    h = None
    
    print(f"Starte Stabilitätstest über {seq_len} Schritte...")
    
    for i in range(seq_len):
        x = torch.randn(1, d_model)
        y, h, _ = model(x, h)
        
        # Check auf Instabilität
        if torch.isnan(y).any() or torch.isnan(h).any():
            print(f"KATASTROPHE: NaNs in Schritt {i} detektiert!")
            return
            
        if i % 10 == 0:
            print(f"Schritt {i:02d}: Output Mean={y.mean().item():.4f}, State Max={h.max().item():.4f}")

    print("--- TEST ERFOLGREICH ---")
    print("Die Wirbelsäule ist stabil und trägt den Zustand kohärent weiter.")

if __name__ == "__main__":
    run_stability_test()
