import torch
import torch.nn.functional as F
from spine import TheSpine

def run_memory_persistence_test():
    d_model = 64
    d_state = 128
    seq_len = 50
    
    model = TheSpine(d_model, d_state)
    
    # 1. Schritt 0: Den Marker setzen (Alles Einsen)
    marker = torch.ones(1, d_model)
    _, h_initial = model(marker, None)
    
    # Wir speichern den Zustand direkt nach dem Marker als Referenz
    h_reference = h_initial.clone().detach()
    
    print(f"Marker gesetzt. Initialer State-Norm: {torch.norm(h_reference).item():.4f}")
    
    # 2. Schritte 1-49: Nur Rauschen füttern
    h_current = h_initial
    for i in range(1, seq_len):
        noise = torch.randn(1, d_model) * 0.1 # Rauschen mit geringer Amplitude
        _, h_current = model(noise, h_current)
        
        if i % 10 == 0:
            # Messung der Ähnlichkeit zum ursprünglichen Marker-Zustand
            similarity = F.cosine_similarity(h_current, h_reference)
            print(f"Schritt {i:02d}: Kosinus-Ähnlichkeit zum Marker = {similarity.item():.4f}")

    # 3. Schritt 50: Finaler Check
    final_similarity = F.cosine_similarity(h_current, h_reference).item()
    
    print("\n--- ERGEBNIS ---")
    print(f"Finale Ähnlichkeit nach {seq_len} Schritten Rauschen: {final_similarity:.4f}")
    
    if final_similarity > 0.5:
        print("DIAGNOSE: Die Wirbelsäule trägt die Information erfolgreich!")
    else:
        print("DIAGNOSE: Signalverlust. Die Wirbelsäule 'vergisst' zu schnell oder das Rauschen dominiert.")

if __name__ == "__main__":
    run_memory_persistence_test()
