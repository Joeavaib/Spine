# Die Wirbelsäule (The Spine) – SSM-basiertes State-Management

## Übersicht
Die „Wirbelsäule“ (The Spine) ist eine hocheffiziente, hybride Architektur zur Sequenzmodellierung, die auf **State Space Models (SSM)** basiert. Sie ersetzt den speicherhungrigen KV-Cache traditioneller Transformer durch einen komprimierten, rekursiven Zustandsvektor. 

Der Fokus liegt auf extremer **Signal-Rausch-Trennung** durch ein energieabhängiges Gating-System, das aktiv entscheidet, welche Informationen im „Eisernen Speicher“ erhalten bleiben und welches Rauschen ignoriert wird.

## Kern-Features
- **Selective State Space (SSM):** Lineare Zustandsfortführung mit selektiven Projektionen für konstante Inferenzzeit $O(1)$ und minimalen Speicherbedarf.
- **Energy-Aware Gating:** Ein dediziertes Gate-MLP analysiert die Signalenergie des Inputs, um zwischen relevanten Informationen (Keys) und Rauschen (Noise) zu unterscheiden.
- **Binary Locking:** Das Modell ist darauf optimiert, bei Rauschen den Zustand („State“) einzufrieren und bei relevantem Input präzise zu aktualisieren.
- **Skip-Gating & RMS-Harness:** Kombiniert Residual-Verbindungen mit Root Mean Square Normalization für maximale numerische Stabilität über extrem lange Sequenzen.

## Projektstruktur
Das Projekt ist nun modular organisiert, um eine klare Trennung zwischen Kernlogik, Experimenten und Modellen zu gewährleisten:

```text
.
├── docs/               # Dokumentation und Whitepaper
├── models/             # Trainierte .pth Modelldateien
├── scripts/            # Trainings-, Test- und Diagnose-Skripte
├── src/                # Kern-Quellcode (TheSpine Modul)
└── README.md
```

## Nutzung & Experimente

Alle Skripte werden vom Hauptverzeichnis aus gestartet:

### 1. Training
- **Phase 1 (Basis-Training):** `python3 scripts/train_phase_1.py`
  Trainiert die Wirbelsäule auf die Rekonstruktion von Informationen nach langen Rausch-Sequenzen.
- **Phase 2 (Joker-Training):** `python3 scripts/train_phase_2.py`
  Optimiert das Gating-Verhalten für eine noch schärfere Trennung von Nutzsignal und Rauschen.
- **Vault-Key-Training (Zweiteiliger Schlüssel):** `python3 scripts/train_two_part_key.py`
  Stresstest für die Hadamard-Attention. Verknüpft zwei zeitlich isolierte Keys über 1000 Schritte Rauschen hinweg.

### 2. Validierung & Tests
- **Stabilitätstest:** `python3 scripts/test_spine.py` (Überprüft die numerische Konsistenz).
- **Gedächtnis-Check:** `python3 scripts/test_memory.py` (Misst die Kosinus-Ähnlichkeit des Zustands über Zeit).
- **Diagnose:** `python3 scripts/diagnose_spine.py` (Visualisiert die Gate-Aktivierung und Trennschärfe).

### 3. Experimente
- **Grid Search:** `python3 scripts/run_experiments.py` führt automatisierte Tests über verschiedene Bias-Werte und Hyperparameter durch, um die optimale Konfiguration der Wirbelsäule zu finden.

## Technische Spezifikationen & Legende

### Kern-Variablen (TheSpine)
| Variable | Beschreibung | Rolle im Modell |
| :--- | :--- | :--- |
| `d_model` | Modell-Dimension | Die Breite des Input- und Output-Vektors. |
| `d_state` | Zustands-Dimension | Die Größe des internen komprimierten Speichers (H-System). |
| `h_t` / `h_prev` | Hidden State | Der aktuelle bzw. vorherige Zustand des "Eisernen Speichers". |
| `W_q`, `W_k`, `W_v` | Hadamard-Projektionen | Query (Input), Key (State), Value (Input) für die Resonanz. |
| `R_t` | Hadamard-Resonanz | Das Produkt $Q \circ K$, das als logischer Kontext-Filter dient. |
| `gate` | Gate-Aktivierung | Binäres Signal (0.0 bis 1.0), das entscheidet, ob Information gespeichert wird. |
| `gate_bias` | Gate-Vorspannung | Steuert die Empfindlichkeit des Modells gegenüber Rauschen. |
| `skip_gate` | Residual-Gate | Reguliert den direkten Signalfluss vom Input zum Output. |
| `D` | Skip-Parameter | Lernbarer Parameter für die gewichtete Residual-Verbindung. |
| `h_norm` | RMS-Norm | Stabilisiert den Zustand vor der Projektion in den Output-Raum. |

### Architektur-Parameter
- **Architektur:** Hybrides Hadamard-Attention SSM.
- **Zustands-Persistenz:** Verifiziert über 50+ Schritte Rauschen (Binary Lock).
- **Komplexität:** Konstante Latenz pro Token, unabhängig von der Historienlänge.
