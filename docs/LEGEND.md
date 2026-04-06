# LEGEND: The Spine Architektur

## 1. Variablen-Index
*   `d_model` : Dimensionalität des Token-Embeddings.
*   `d_state` : Dimensionalität der "Wirbelsäule" (SSM-State).
*   `h_prev`/`h_t`: Zustand des SSM zu Zeitpunkt `t-1` bzw `t`.
*   `W_q`, `W_k`, `W_v`: Hadamard-Projektionen (Query from Input, Key from State, Value from Input).
*   `R_t`: Hadamard-Resonanz (Q ∘ K), dient als logischer Filter.
*   `W_tau`, `b_tau`: Parameter zur Berechnung des dynamischen Schwellenwerts $\tau_t$ aus $h_{t-1}$.
*   `tau_t`: Dynamischer Widerstand (Skalar), berechnet via `Softplus(W_tau * h_prev + b_tau)`.
*   `gate_mlp`: Lineare Projektion (`d_state` -> `1`), die auf der Resonanz operiert.
*   `gate`: Binärer Selektionsfaktor (via `clamp(MLP_gate(R_t) - tau_t, 0, 1)`).

## 2. Formel-Referenz (SSM-Hybrid)

### Hadamard-Attention & State-Conditioned Gating
1. $q_t = W_q x_t, \quad k_t = W_k h_{t-1}, \quad v_t = W_v x_t$
2. $R_t = q_t \circ k_t \quad \text{(Hadamard-Resonanz)}$
3. $\tau_t = \text{Softplus}(W_{\tau} h_{t-1} + b_{\tau}) \quad \text{(Dynamische Hürde)}$
4. $g_t = \text{clamp}(MLP_{gate}(R_t) - \tau_t, 0.0, 1.0)$
5. $x_{\text{refined}, t} = R_t \circ v_t$
6. $h_t = (1 - g_t) \cdot h_{t-1} + g_t \cdot x_{\text{refined}, t}$

*   Wenn $g_t=0$: $h_t = h_{t-1}$ (Identität/Lock).
*   Wenn $g_t=1$: $h_t = x_{\text{refined}, t}$ (Resonante Injektion).

### Resonanz-Aware Selection
*   Das Gate entscheidet basierend auf der Resonanz $R_t$, ob der Kontext ein "Knotting" (Zusammenführung) von Input und State erlaubt.


### Output-Fusion
$$y_t = \text{RMSNorm}(h_t) \cdot W_{out} + (x_t \cdot D \cdot \sigma(\text{skip\_gate}(x_t)))$$
*   Der State wird über `h_norm` (RMSNorm) für den Abruf normalisiert.
*   Der Skip-Bypass filtert Rauschen über das `skip_gate`.
