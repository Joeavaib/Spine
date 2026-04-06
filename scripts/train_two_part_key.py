import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from src.spine import TheSpine

def get_two_part_key_batch(batch_size, seq_len, d_model):
    """
    Erzeugt einen Batch für den Two-Part Key Task:
    - xA (Anchor): Zufällig in den ersten 300 Schritten.
    - xB (Trigger): Zufällig in den letzten 300 Schritten.
    - Ngap (Noise): Alles andere ist Rauschen.
    - ytarget: xA + xB (Logische Fusion).
    - Loss: Nur am letzten Schritt berechnet.
    """
    # x: [seq_len, batch_size, d_model]
    x = torch.randn(seq_len, batch_size, d_model) * 0.1 # Grundrauschen
    
    xA = torch.randn(batch_size, d_model)
    xB = torch.randn(batch_size, d_model)
    
    # Fusion Target (was am Ende rauskommen soll)
    # Wir nehmen eine einfache Addition als Repräsentant für die Fusion
    y_target = xA + xB
    
    for b in range(batch_size):
        # xA Platzierung (0 bis 299)
        idx_a = random.randint(0, 299)
        x[idx_a, b] = xA[b]
        
        # xB Platzierung (700 bis 999)
        idx_b = random.randint(700, seq_len - 1)
        x[idx_b, b] = xB[b]
        
    return x, y_target, xA, xB

def train_vault():
    d_model = 16
    d_state = 32
    seq_len = 1000
    batch_size = 8 # Kleiner Batch wegen seq_len=1000
    epochs = 1000

    model = TheSpine(d_model, d_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    print(f"--- TRAINING: DER ZWEITEILIGE SCHLÜSSEL ---")
    print(f"SeqLen: {seq_len} | d_model: {d_model} | d_state: {d_state}")
    print(f"xA (0-299) -> Noise -> xB (700-999) -> ytarget (Final Step)")
    print("-" * 50)

    for epoch in range(epochs):
        x, target, xA_batch, xB_batch = get_two_part_key_batch(batch_size, seq_len, d_model)
        h = None
        
        # Wir müssen h_prev für das Gate mitschleifen
        # Da TheSpine.forward(x, h_prev) erwartet:
        for t in range(seq_len):
            y, h, gate = model(x[t], h)
            
        # Wir berechnen den Loss NUR am letzten Output y (nach Schritt 999)
        loss_mse = F.mse_loss(y, target)
        
        # Optional: Sparsity-Regularisierung für das Gate
        # (Das Modell soll das Gate nicht dauerhaft offen lassen)
        # Wir tracken das Gate hier nicht über alle 1000 Schritte für den Loss (Speicher),
        # aber wir könnten es stichprobenartig tun.
        
        optimizer.zero_grad()
        loss_mse.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Vault-Loss (MSE): {loss_mse:.6f}")
            
        if loss_mse < 0.001:
            print("!!! TRESOR GEKNACKT !!!")
            break

    print("-" * 50)
    print("Training abgeschlossen.")
    torch.save(model.state_dict(), "models/spine_vault_key.pth")

if __name__ == "__main__":
    train_vault()
