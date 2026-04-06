import torch
import torch.nn as nn
import torch.nn.functional as F

class TheSpine(nn.Module):
    """
    Die Wirbelsäule (The Spine) - Ein hybrider State-Compressor auf SSM-Basis.
    Optimiert für extreme Signal-Rausch-Trennung (Energy-Aware Gating).
    """
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 2. Hadamard-Attention Projektionen
        self.W_q = nn.Linear(d_model, d_state, bias=False) # Query (from Input)
        self.W_k = nn.Linear(d_state, d_state, bias=False) # Key (from State)
        self.W_v = nn.Linear(d_model, d_state, bias=False) # Value (from Input)

        # Gate-Mechanik mit dynamischem Schwellenwert tau
        self.gate_mlp = nn.Linear(d_state, 1) 
        self.W_tau = nn.Linear(d_state, 1) # Erzeugt den dynamischen Widerstand tau_t
        self.skip_gate = nn.Linear(d_model, 1)

        # Initialisierung
        nn.init.xavier_uniform_(self.gate_mlp.weight, gain=1.0)
        nn.init.constant_(self.W_tau.bias, 3.0) # Start-Widerstand (tau ≈ 3.0)
        nn.init.constant_(self.skip_gate.bias, -3.0) 

        # 3. Skip-Connection & Norm
        self.D = nn.Parameter(torch.ones(d_model))
        self.output_linear = nn.Linear(d_state, d_model)
        self.h_norm = nn.RMSNorm(d_state)

        self.h_system = nn.Parameter(torch.zeros(1, d_state))
        nn.init.normal_(self.h_system, std=0.02)

    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = self.h_system.expand(x.shape[0], -1)

        # 1. Q, K, V Projektionen
        q = self.W_q(x)
        k = self.W_k(h_prev)
        v = self.W_v(x)
        
        # 2. Hadamard-Resonanz (Q ∘ K)
        # Elementweise Multiplikation formt den harten Logik-Filter
        R_t = q * k 
        
        # 3. Dynamisches Schwellenwert-Gating (State-Conditioned)
        # Widerstand tau wird aus dem Vorwissen h_prev berechnet
        tau_t = F.softplus(self.W_tau(h_prev))
        
        # Gate-Logits minus Schwellenwert
        logits = self.gate_mlp(R_t) - tau_t
        gate = torch.clamp(logits, min=0.0, max=1.0)
        
        # 4. Synthese (Hadamard Attention Application)
        # Die Resonanz-Maske wird auf die Nutzlast V angewendet
        x_refined = R_t * v 
        
        # 5. Drift-freies State-Update (Binary Lock bleibt erhalten)
        h_t = (1.0 - gate) * h_prev + gate * x_refined

        # 6. Skip-Gating (binär) & Output
        skip_gate = torch.clamp(self.skip_gate(x), 0.0, 1.0)
        y = self.output_linear(self.h_norm(h_t).to(x.dtype)) + (x * self.D * skip_gate)

        return y, h_t, gate
if __name__ == "__main__":
    model = TheSpine(d_model=32, d_state=64)
    x = torch.randn(1, 32)
    y, h, g = model(x)
    print(f"Output: {y.shape}, State: {h.shape}, Gate: {g.shape}")
