import torch
import torch.nn as nn
import torch.nn.functional as F

class TheSpine(nn.Module):
    """
    Die Wirbelsäule (The Spine) - Ein hybrider State-Compressor auf SSM-Basis.
    Eliminiert den KV-Cache durch einen rekursiven, dynamischen Zustandsvektor.
    """
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 1. Symmetriebrechung: Multi-Scale Zeitkonstanten
        # Wir initialisieren A logarithmisch, um verschiedene Gedächtnisspannen abzudecken.
        A_init = torch.log(torch.arange(1, d_state + 1).float())
        self.A_log = nn.Parameter(A_init)
        
        # 2. Selektive Projektionen (Datenabhängigkeit)
        # B: Wie stark beeinflusst der Input den Zustand?
        # C: Wie lesen wir den Zustand für den aktuellen Kontext aus?
        # Gate: Wie viel 'vergessen' wir im aktuellen Schritt?
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state) # C ist oft input-abhängig
        self.gate_proj = nn.Linear(d_model, 1)
        # Initialisierung für Langzeitgedächtnis: Gate standardmäßig fast geschlossen (0.1)
        nn.init.constant_(self.gate_proj.bias, -2.0)
        
        # 3. Skip-Connection & Norm (Harness)
        self.D = nn.Parameter(torch.ones(d_model))
        self.output_linear = nn.Linear(d_state, d_model) # Korrigiert: d_state -> d_model
        
        # System-DNA (Lernbarer Initialzustand)
        self.h_system = nn.Parameter(torch.randn(1, d_state) * 0.02)

    def forward(self, x, h_prev=None):
        """
        x: [batch, d_model] - Aktuelles Token-Embedding
        h_prev: [batch, d_state] - Vorheriger Zustand der Wirbelsäule
        """
        if h_prev is None:
            # Starte mit der System-DNA, falls kein Zustand übergeben wurde
            h_prev = self.h_system.expand(x.shape[0], -1)

        # Stabilität: A im Bereich (0, 1) halten
        A = torch.exp(-torch.exp(self.A_log))
        
        # Selektives Gating (Input-Dependent)
        gate = torch.sigmoid(self.gate_proj(x))
        
        # Input-Injektion (B)
        B_x = self.B_proj(x)
        
        # Rekursives Update (Das Herzstück)
        # h_t = (1-gate) * h_{t-1} + gate * (A * h_{t-1} + B_x)
        # Wir nutzen eine stabilisierte Form des Updates:
        h_t = (1 - gate) * h_prev + gate * (A * h_prev + torch.tanh(B_x))
        
        # Harness: Verhindere Drift durch RMSNorm auf dem Zustand
        h_t = F.rms_norm(h_t, (self.d_state,))
        
        # Semantischer Abruf (C)
        # C fungiert als "Aufmerksamkeits-Filter" für den internen Zustand
        C = torch.sigmoid(self.C_proj(x))
        y_state = C * h_t
        
        # Fusion: Zustand + Direkter Input (Skip-Connection)
        # Wir projizieren den Zustand zurück in den d_model Space
        # (Hier vereinfacht über Summation/Linear, in S4/Mamba komplexer)
        y = self.output_linear(y_state.to(x.dtype)) + (x * self.D)
        
        return y, h_t

if __name__ == "__main__":
    # Schneller Sanity-Check
    model = TheSpine(d_model=128, d_state=256)
    x_dummy = torch.randn(1, 128)
    y, h = model(x_dummy)
    print(f"Output Shape: {y.shape}") # Erwartet: [1, 128]
    print(f"State Shape: {h.shape}")   # Erwartet: [1, 256]
    print("Spine initialisiert und funktionsfähig.")
