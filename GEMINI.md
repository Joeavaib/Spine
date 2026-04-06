# Gemini Instructions: Projekt Wirbelsäule

Dieses Dokument definiert die spezifischen Arbeitsanweisungen für die KI-Assistenz in diesem Projekt.

## Mandate & Workflows

1. **Autonome Dokumentationspflege:**
   - Bei jeder strukturellen Änderung am Code (neue Module, geänderte Schnittstellen, neue Skripte) muss die `README.md` automatisch und ohne explizite Aufforderung aktualisiert werden.
   - Die Projektstruktur-Übersicht in der `README.md` ist stets synchron zum Dateisystem zu halten.

2. **Variablen-Konsistenz:**
   - Alle neu eingeführten Kern-Variablen in `src/spine.py` oder den Trainingsskripten müssen in der Variablen-Legende der `README.md` dokumentiert werden.

3. **Pfad-Integrität:**
   - Skripte in `scripts/` müssen so konzipiert sein, dass sie vom Projekt-Root aus ausführbar sind.
   - Modell-Artefakte sind strikt im Ordner `models/` zu verwalten.

4. **Code-Qualität:**
   - Änderungen an `src/spine.py` erfordern eine anschließende Validierung durch `scripts/test_spine.py`.
