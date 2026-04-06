# LEGEND: The Spine Architektur

## 1. Variablen-Index
*   `d_model` : Dimensionalität des Token-Embeddings.
*   `d_state` : Dimensionalität der "Wirbelsäule" (SSM-State).
*   `h_prev`/`h_t`: Zustand des SSM zu Zeitpunkt `t-1` bzw `t`.
*   `W_q`, `W_k`, `W_v`: Hadamard-Projektionen (Query from Input, Key from State, Value from Input).
*   `R_t`: Hadamard-Resonanz (Q ∘ K), dient als logischer Filter.
*   `gate_mlp`: Der "Gatekeeper" (Linear: `d_state` -> `1`), operiert auf der Resonanz.
*   `gate`: Binärer Selektionsfaktor (via `clamp` auf `[0, 1]`).
*   `skip_gate`: Gating der Skip-Connection (Noise-Filter).
*   `D`: Learnable Skalierung der Skip-Connection.
*   `output_linear`: Projektion vom State `h_t` auf das Output-Embedding.

## 2. Formel-Referenz (SSM-Hybrid)

### Hadamard-Attention & State-Update
1. $q_t = W_q x_t, \quad k_t = W_k h_{t-1}, \quad v_t = W_v x_t$
2. $R_t = q_t \circ k_t \quad \text{(Resonanz-Filter)}$
3. $g_t = \text{clamp}(MLP_{gate}(R_t) + \text{bias}, 0, 1)$
4. $x_{\text{refined}, t} = R_t \circ v_t$
5. $h_t = (1 - g_t) \cdot h_{t-1} + g_t \cdot x_{\text{refined}, t}$

*   Wenn $g_t=0$: $h_t = h_{t-1}$ (Identität/Lock).
*   Wenn $g_t=1$: $h_t = x_{\text{refined}, t}$ (Resonante Injektion).

### Resonanz-Aware Selection
*   Das Gate entscheidet basierend auf der Resonanz $R_t$, ob der Kontext ein "Knotting" (Zusammenführung) von Input und State erlaubt.


### Output-Fusion
$$y_t = \text{RMSNorm}(h_t) \cdot W_{out} + (x_t \cdot D \cdot \sigma(\text{skip\_gate}(x_t)))$$
*   Der State wird über `h_norm` (RMSNorm) für den Abruf normalisiert.
*   Der Skip-Bypass filtert Rauschen über das `skip_gate`.
