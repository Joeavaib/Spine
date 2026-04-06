import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.spine import TheSpine
from train_phase_1 import get_batch, compute_gate_supervision_loss

def finetune():
    d_model = 32
    d_state = 64
    seq_len = 20
    batch_size = 32
    epochs = 500
    
    model = TheSpine(d_model, d_state)
    # Lade das beste Modell vom letzten Training
    try:
        model.load_state_dict(torch.load("models/spine_model.pth"))
        print("Modell geladen.")
    except:
        print("Konnte Modell nicht laden, starte mit Initialisierung.")

    # Chirurgische Fixierung: Selektion einfrieren
    for param in model.gate_mlp.parameters():
        param.requires_grad = False
    for param in model.skip_gate.parameters():
        param.requires_grad = False
    model.gate_bias.requires_grad = False

    # Nur output_linear und B_proj (Feinanpassung) lernen
    optimizer = torch.optim.Adam(
        list(model.output_linear.parameters()) + list(model.B_proj.parameters()), 
        lr=0.0005
    )
    
    print(f"Starte Finetuning (Surgical Mode) über {epochs} Epochen...")
    
    for epoch in range(epochs):
        x, target = get_batch(batch_size, seq_len, d_model)
        h = None
        
        for t in range(seq_len):
            y, h, _ = model(x[t], h)
        
        loss = F.mse_loss(y, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d} | MSE: {loss:.6f}")
            
    torch.save(model.state_dict(), "models/spine_model_finetuned.pth")
    print("Finetuning abgeschlossen.")

if __name__ == "__main__":
    finetune()
