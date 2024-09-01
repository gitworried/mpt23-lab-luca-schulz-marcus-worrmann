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
./out/mpt_nn <-v> <modus> <numTrainingSets> <numInputs> <numHiddenNodes> <numOutputs> <epochs> <learningRate>
```

### Parameter

Die Parameter sind wie folgt definiert:

- `<-v>` : Aktivierung der visualisierung während des Trainings mit dem MNIST-Datensatz
- `<modus>`: Ausführungsmodus (`sequential`, `parallel`, `simd`)
- `<numTrainingSets>`: Anzahl der Trainingsdaten (z.B. 60000 für den gesamten MNIST-Datensatz)
- `<numInputs>`: Anzahl der Eingangsneuronen (784 für MNIST)
- `<numHiddenNodes>`: Anzahl der Neuronen in der versteckten Schicht (z.B. 128)
- `<numOutputs>`: Anzahl der Ausgangsneuronen (z.B. 10 für die 10 Ziffern)
- `<epochs>`: Anzahl der Epochen für das Training
- `<learningRate>`: Lernrate (z.B. 0.01)

### Beispiel

```bash
./out/mpt_nn simd 60000 784 10 10 10 0.1
```

Startet das Training mit dem MNIST-Datensatz mit:

- 60000 Trainingsdaten
- 784 Eingangsneuronen
- 10 versteckte schichten
- 10 Outputs
- 10 epochen
- Einer Lernrate von 0.1

## Unit Tests

Um sicherzustellen, dass alle implementierten funktionen wie gewollt funktionieren wurden unit test definiert. Dies befinden sich in der Datei mpt_nn_test.c und testen die Kern functionen (sigmoid, forwardpass, backpropagation) in allen drei Modi(Sequential, Parallel, SIMD).

Die test lassen sich einfach mittels make ausführen.

```bash
make test
```

## Memory und Thread errors

Damit sichergestellt werden kann, dass es keine memory leaks oder thread errors (race conditions) gibt, wurde das überprüfen dieser mit valgrind (memory leaks) und dessen tool helgrind (thread errors) als Makefile Rule implementiert.

Mit

```bash
make valgrind
make helgrind
```

lassen sich beide tools ausführen.

## Benchmarks

Auch das Erstellen von Benchmarks wurde als Makefile Rule implementiert. Hierbei wird das Tool hyperfine benutzt.

Ebenfalls lassen sich die Benchmarks mit

```bash
make benchmark
```

erstellen.

## Dokumentation

Zurzeit ist es leider nur via Doxygen möglich detaillierte Auskunft über die funktionen des mpt_nn zu erhalten.<br>
Dabei lässt sich ein Doxygen Dokumentations Ordner über die Kommandozeile kompilieren. <br>

```bash
doxygen
```

In dem Ordner doxygen befindet sich ein Unterordner latex, in dem eine generiertes latex Projekt inkl. Makefile mit Informationen zu jedem source file zur Verfügung steht.

```bash
make
```

Um eine PDF zu generieren.

## TODO

- [x] Doxygen Doku<br>
- [ ] Benchmark Visualisierung (Mit python script oder R?) <br>
- [ ] Villeicht: Default Parameter zum starten des mpt_nn (z.B angepasst an den MNIST-Datensatz, also 60000 trainingsdaten, 784 eingangsneuronen... etc)<br>
- [ ] Code optimieren(Benchmarks) Zurzeit kommen komische Benchmark Ergebnisse raus. Sequentielles ausführen ist in den meisten Fällen deutlich schneller als parallel und SIMD. Das kann natürlich sein, aber ist eher unwahrscheinlich. (Programm auf mehreren Maschinen testen).<br>
- [ ] Code optimieren(helgrind, valgrind): helgrind zeigt noch etliche potentielle race conditions an.<br>
- [ ] Doku schreiben.<br>
