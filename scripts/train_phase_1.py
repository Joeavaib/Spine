import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.spine import TheSpine

def get_batch(batch_size, seq_len, d_model):
    """
    Erzeugt einen Batch für den Identity Task:
    - Step 0: Ein zufälliger Key (Target)
    - Steps 1 bis seq_len-1: Rauschen (Noise)
    - Target: Rekonstruktion des Keys am Ende
    """
    # Key am Anfang der Sequenz
    key = torch.randn(batch_size, d_model)
    
    # Rauschen für die restliche Sequenz (Noise-Scale 0.5)
    noise = torch.randn(seq_len - 1, batch_size, d_model) * 0.5
    
    # Sequenz zusammenfügen: [seq_len, batch_size, d_model]
    x = torch.cat([key.unsqueeze(0), noise], dim=0)
    
    return x, key

def compute_gate_supervision_loss(gates, lambda_key=0.02, lambda_noise=0.02):
    """
    Bestraft die Lazy-Gate-Pathologie:
    - g_0 (Key) soll 1.0 sein
    - g_{t>0} (Noise) soll 0.0 sein
    """
    # gates: [seq_len, batch_size, 1]
    loss_key = (1.0 - gates[0]).pow(2).mean()
    loss_noise = gates[1:].pow(2).mean()
    
    return lambda_key * loss_key + lambda_noise * loss_noise

import argparse

def train(lambda_key=0.02):
    d_model = 32
    d_state = 64
    seq_len = 20
    batch_size = 32
    epochs = 3000

    model = TheSpine(d_model, d_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print(f"Starte Phase 1.6: Energy-Aware Training (SeqLen={seq_len}, lambda_key={lambda_key})")
    print("-" * 50)

    for epoch in range(epochs):
        x, target = get_batch(batch_size, seq_len, d_model)
        h = None
        gates_list = []

        # Sequentieller Forward-Pass durch die Wirbelsäule
        for t in range(seq_len):
            _, h, gate = model(x[t], h)
            gates_list.append(gate)

        # Letzter Output y ist für den MSE relevant
        y, _, _ = model(x[-1], h) 

        # Gates für Aux-Loss sammeln: [seq_len, batch, 1]
        all_gates = torch.stack(gates_list)

        # 1. Haupt-Loss: Rekonstruktion am Ende der Sequenz
        loss_mse = F.mse_loss(y, target)

        # 2. Auxiliary Loss: Gate-Supervision (lambda_key dynamisch)
        loss_gate = compute_gate_supervision_loss(all_gates, lambda_key=lambda_key)

        loss = loss_mse + loss_gate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d} | MSE: {loss_mse:.6f} | Gate-Loss: {loss_gate:.6f}")
            
    print("-" * 50)
    print("Training abgeschlossen.")
    
    # Letzte Evaluation
    with torch.no_grad():
        x_val, target_val = get_batch(100, seq_len, d_model)
        h_val = None
        for t in range(seq_len):
            y_val, h_val, _ = model(x_val[t], h_val)
        final_mse = F.mse_loss(y_val, target_val).item()
        print(f"Finaler Rekonstruktions-Fehler (MSE): {final_mse:.8f}")
        torch.save(model.state_dict(), "models/spine_model.pth")
        
        if final_mse < 0.01:
            print("MISSION ERFÜLLT: Die Wirbelsäule hält den Key!")
        else:
            print("DIAGNOSE: Fokus reicht noch nicht aus.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_key", type=float, default=0.02)
    args = parser.parse_args()
    train(lambda_key=args.lambda_key)
