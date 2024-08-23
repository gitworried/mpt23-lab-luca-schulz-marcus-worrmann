# MPT Nueronales Netzwerk für den MNIST-Datensatz

Dieses Projekt implementiert ein einfaches neuronales Netzwerk zur Klassifikation des MNIST-Datensatzes. Das Netzwerk unterstützt verschiedene Ausführungsmodi (sequentiell, parallel und SIMD) und kann über die Kommandozeile konfiguriert werden.

## Installation

1. **Voraussetzungen:**
   - GCC-Compiler mit OpenMP-Unterstützung
   - GNU Make
   - MNIST-Datensatz (`train-images.idx3-ubyte` und `train-labels.idx1-ubyte`) im Projektverzeichnis

2. **Projekt kompilieren:**

Das Netzwerk kann einfach mittels make gebaut werden:
   ```bash
   make
   ```

3. **Starten des Programmes:**

Wenn das Netzwerk erfolgreich gebaut wurde, kann es einfach über die Kommandozeile gestartet werden.

```bash
./out/mpt_nn <modus> <numTrainingSets> <numInputs> <numHiddenNodes> <numOutputs> <epochs> <learningRate>
```
### Parameter
Die Parameter sind wie folgt definiert:
- `<modus>`: Ausführungsmodus (`sequential`, `parallel`, `simd`)
- `<numTrainingSets>`: Anzahl der Trainingsdaten (z.B. 60000 für den gesamten MNIST-Datensatz)
- `<numInputs>`: Anzahl der Eingangsneuronen (784 für MNIST)
- `<numHiddenNodes>`: Anzahl der Neuronen in der versteckten Schicht (z.B. 128)
- `<numOutputs>`: Anzahl der Ausgangsneuronen (z.B. 10 für die 10 Ziffern)
- `<epochs>`: Anzahl der Epochen für das Training
- `<learningRate>`: Lernrate (z.B. 0.01)




