import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from src.spine import TheSpine

def get_vault_batch(batch_size, seq_len, d_model):
    x = torch.randn(seq_len, batch_size, d_model) * 0.05 # Reduziertes Rauschen
    xA = torch.randn(batch_size, d_model)
    xB = torch.randn(batch_size, d_model)
    target = xA + xB
    
    a_indices = []
    b_indices = []
    
    for b in range(batch_size):
        idx_a = random.randint(50, 250)
        idx_b = random.randint(750, 950)
        x[idx_a, b] = xA[b]
        x[idx_b, b] = xB[b]
        a_indices.append(idx_a)
        b_indices.append(idx_b)
        
    return x, target, a_indices, b_indices

def train_vault_advanced():
    d_model, d_state = 16, 32
    seq_len = 1000
    batch_size = 16
    epochs = 2000
    
    model = TheSpine(d_model, d_state)
    
    # 1. Warm-up: h_system verstärken, um Gate zu öffnen
    with torch.no_grad():
        nn.init.normal_(model.h_system, std=0.5)
        model.gate_bias.data.fill_(-1.0) # Etwas offener starten
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("--- ADVANCED VAULT TRAINING ---")
    
    for epoch in range(epochs):
        x, target, a_idxs, b_idxs = get_vault_batch(batch_size, seq_len, d_model)
        h = None
        gates = []
        
        for t in range(seq_len):
            y, h, gate = model(x[t], h)
            gates.append(gate)
            
        gates = torch.stack(gates) # [seq_len, batch, 1]
        
        # 1. MSE Loss (Final step)
        loss_mse = F.mse_loss(y, target)
        
        # 2. Gate Supervision (Warm-up / Auxiliary)
        # Wir erzwingen, dass das Gate bei den Key-Indizes offen ist
        # und sonst (Sparsity) eher geschlossen.
        key_mask = torch.zeros(seq_len, batch_size, 1, device=x.device)
        for b in range(batch_size):
            key_mask[a_idxs[b], b] = 1.0
            key_mask[b_idxs[b], b] = 1.0
            
        loss_gate = F.binary_cross_entropy(gates, key_mask)
        
        # Gewichtung: Am Anfang viel Gate-Supervision, später mehr MSE
        alpha = max(0.1, 1.0 - epoch / 1000.0)
        loss = (1-alpha) * loss_mse + alpha * loss_gate
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Total: {loss:.4f} | MSE: {loss_mse:.4f} | Gate-BCE: {loss_gate:.4f}")
            
        if loss_mse < 0.05:
            print("!!! VAULT SOLVED !!!")
            break

    torch.save(model.state_dict(), "models/spine_vault_key.pth")

if __name__ == "__main__":
    train_vault_advanced()
