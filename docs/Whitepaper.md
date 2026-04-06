# Whitepaper: The Spine SSM

## Phase 1.6 & 2: Inhalts-Adressierung und Signal-Isolierung

### Zusammenfassung
In der Joker-Phase wurde bewiesen, dass The Spine keine temporale Metronom-Logik verwendet, sondern eine semantische Energie-Selektion. Durch die Injektion von $\|x\|_2^2$ in das Gate-MLP wurde die Symmetrie zwischen Key und Noise gebrochen.

### Architektonische Erkenntnisse
1. **Energy-Aware Gating**: Das Gate reagiert auf die Informationsdichte.
   $$g(x_t) = \sigma(MLP_{gate}(\text{concat}(x_t, \mathbb{E}[x_t^2]))) + \text{bias}_{\text{init}}$$
2. **Tabula Rasa**: Durch $\text{h}_{\text{system}} = 0$ werden bias-induzierte Signal-Verzerrungen am Sequenzanfang eliminiert.
3. **Joker-Robustheit**: Das Modell extrahiert Keys unabhängig von ihrer zeitlichen Position durch "Abwarten" (Gate $\approx$ 0) bis zum Energy-Spike.

### Mathematische Herleitung der Stabilität
Die Stabilität des Zustands $h_t$ während Rausch-Phasen ($x_t \approx \epsilon$) wird durch die Bedingung erzwungen:
$$\lim_{\|x_t\| \to 0} g(x_t) \approx \sigma(-3.0) \to 0$$
Dies garantiert $h_t \approx h_{t-1}$ und verhindert eine State-Drift-Pathologie.
