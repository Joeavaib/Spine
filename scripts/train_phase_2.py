import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.spine import TheSpine

def get_shuffled_batch(batch_size, seq_len, d_model):
    # Alle Vektoren als Rauschen initialisieren
    x = torch.randn(seq_len, batch_size, d_model) * 0.1
    
    # Key an zufälliger Position einfügen
    key = torch.randn(batch_size, d_model)
    target_pos = torch.randint(0, seq_len, (1,)).item()
    x[target_pos] = key
    
    return x, key, target_pos

def train():
    d_model = 32
    d_state = 64
    seq_len = 20
    batch_size = 32
    epochs = 2000
    
    model = TheSpine(d_model, d_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    print(f"Starte Phase 2: Der Joker (Dynamisches Shuffling, SeqLen={seq_len})")
    
    for epoch in range(epochs):
        x, target, target_pos = get_shuffled_batch(batch_size, seq_len, d_model)
        h = None
        gates_list = []
        
        # Sequentieller Forward-Pass
        for t in range(seq_len):
            y, h, gate = model(x[t], h)
            gates_list.append(gate)
            
        all_gates = torch.stack(gates_list) # [seq_len, batch, 1]
        
        # Loss: Rekonstruktion (y ist nach dem letzten Step 't=seq_len-1')
        # Wir müssen den State h zum Zeitpunkt t=seq_len-1 nutzen
        loss_mse = F.mse_loss(y, target)
        
        # Joker-Gate-Loss
        loss_key_gate = (1.0 - all_gates[target_pos]).pow(2).mean()
        
        mask = torch.ones(seq_len, dtype=torch.bool)
        mask[target_pos] = False
        loss_noise_gate = all_gates[mask].pow(2).mean()
        
        loss = loss_mse + 0.05 * (loss_key_gate + loss_noise_gate)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:04d} | MSE: {loss_mse:.6f} | Gate-Loss: {loss_key_gate + loss_noise_gate:.6f}")

    torch.save(model.state_dict(), "models/spine_v2_joker.pth")
    print("Training Joker-Phase abgeschlossen.")

if __name__ == "__main__":
    train()
