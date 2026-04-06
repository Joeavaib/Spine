import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from src.spine import TheSpine

def stress_test(seq_lens=[20, 100, 500, 1000]):
    d_model = 32
    d_state = 64
    
    model = TheSpine(d_model, d_state)
    # Lade das trainierte Modell (wir nutzen das Joker-Modell als Basis)
    model.load_state_dict(torch.load("models/spine_v2_joker.pth"), strict=False)
    model.eval()
    
    print(f"--- STRESSTEST: Binary State Lock ---")
    
    for seq_len in seq_lens:
        x = torch.randn(seq_len, 1, d_model) * 0.1
        key = torch.randn(1, d_model)
        target_pos = torch.randint(0, seq_len, (1,)).item()
        x[target_pos] = key
        
        h = None
        drift = 0.0
        with torch.no_grad():
            h_init = model.h_system.clone()
            for t in range(seq_len):
                y, h, gate = model(x[t], h)
                # Messung der Drift bei Noise (gate sollte 0 sein)
                if t > 0: 
                    drift += torch.norm(h - h_init).item()
                    h_init = h.clone()
                
        mse = F.mse_loss(y, key).item()
        print(f"SeqLen: {seq_len:<5} | MSE nach Abruf: {mse:.6f} | State-Drift: {drift:.8f}")

if __name__ == "__main__":
    stress_test()
