import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.spine import TheSpine
from scripts.train_two_part_key import get_two_part_key_batch

def diagnose_vault():
    d_model, d_state = 16, 32
    seq_len = 1000
    model = TheSpine(d_model, d_state)
    
    # Versuche das Modell zu laden, falls es existiert
    if os.path.exists("models/spine_vault_key.pth"):
        model.load_state_dict(torch.load("models/spine_vault_key.pth", weights_only=True))
        print("Modell geladen.")
    else:
        print("Teste frisches Modell.")

    x, target, xA, xB = get_two_part_key_batch(1, seq_len, d_model)
    h = None
    
    gates = []
    with torch.no_grad():
        for t in range(seq_len):
            y, h, gate = model(x[t], h)
            gates.append(gate.item())
            
    gates = torch.tensor(gates)
    print(f"Gate Statistik - Max: {gates.max():.4f}, Mean: {gates.mean():.4f}, Min: {gates.min():.4f}")
    
    # Finde heraus, wo xA und xB waren
    # Da wir wissen, wie get_two_part_key_batch arbeitet, suchen wir nach den Peaks in x
    x_norms = torch.norm(x, dim=-1).squeeze()
    indices = torch.where(x_norms > 0.5)[0] # Unsere Keys haben std 1.0, Noise std 0.1
    
    print(f"Key Indizes gefunden bei: {indices.tolist()}")
    for idx in indices:
        print(f"Gate bei Index {idx}: {gates[idx]:.4f}")

    if gates.max() < 0.1:
        print("\nDIAGNOSE: Das Gate öffnet sich nie! (Dead Gate Problem)")
        print("Mögliche Lösung: Höhere h_system Initialisierung oder temporär positiver gate_bias.")
    elif gates[indices].mean() < 0.1:
        print("\nDIAGNOSE: Das Gate öffnet sich, aber NICHT bei den Keys.")
    else:
        print("\nDIAGNOSE: Das Gate reagiert auf die Keys. Training sollte theoretisch möglich sein.")

if __name__ == "__main__":
    diagnose_vault()
