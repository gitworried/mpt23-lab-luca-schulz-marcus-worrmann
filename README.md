# MPT Neuronales Netzwerk für den MNIST-Datensatz

Dieses Projekt implementiert ein einfaches neuronales Netzwerk zur Klassifikation des MNIST-Datensatzes. Das Netzwerk unterstützt verschiedene Ausführungsmodi (sequentiell, parallel und SIMD) und kann über die Kommandozeile konfiguriert werden.

## Installation

1. **Voraussetzungen:**

   - Linux Distribution
   - GCC-Compiler mit OpenMP-Unterstützung
   - GNU Make
   - MNIST-Datensatz (`train-images.idx3-ubyte` und `train-labels.idx1-ubyte`) im Projektverzeichnis
   - R mit zusätzlichen Bibliotheken

2. **Projekt kompilieren:**

Das Netzwerk kann einfach mittels make gebaut werden:

```bash
make
```

3. **Starten des Programmes:**

Wenn das Netzwerk erfolgreich gebaut wurde, kann es einfach über verschiedene Kommandozeilenoptionen konfiguriert und gestartet werden.

```bash
./out/mpt_nn -v -m<modus> -t<numTrainingSets> -i<numInputs> -h<numHiddenNodes> -o<numOutputs> -e<epochs> -l<learningRate> -d<dropoutRate>
```

### Parameter

Die Parameter sind wie folgt definiert:

- `-D` : Startet das Netzwerk mit vordefinierten default parametern
- `-d` : Droput Rate (Setzt zufällige neuronen auf 0 während forward pass und backpropagation, z.b 0.1 für 10% droput Rate)
- `-v` : Aktivierung der visualisierung während des Trainings mit dem MNIST-Datensatz
- `-m <modus>`: Ausführungsmodus (`sequential`, `parallel`, `simd`)
- `-t <numTrainingSets>`: Anzahl der Trainingsdaten (z.B. 60000 für den gesamten MNIST-Datensatz)
- `-i <numInputs>`: Anzahl der Eingangsneuronen (784 für MNIST)
- `-h <numHiddenNodes>`: Anzahl der Neuronen in der versteckten Schicht (z.B. 128)
- `-o <numOutputs>`: Anzahl der Ausgangsneuronen (z.B. 10 für die 10 Ziffern)
- `-e <epochs>`: Anzahl der Epochen für das Training
- `-l <learningRate>`: Lernrate (z.B. 0.01)
- `-n <numThreads>` : Setzt die Anzahl an verwendetend Threads fest, die beim ausführen eine Parallelregion benutzt werden
- `-? <--help>` : Zeigt die verfügbaren Kommandozeilenoptionen

**WICHTIG:** Das mpt_nn setzt gewisse Parameter zum starten vorraus. Entweder nur `-D`, da dieser vordefinierte default Parameter setzt,
oder `-t , -m, -i, -h. -o, -e, -l` um das Netzwerk nach belieben zu knfigurieren. Die restilichen Parameter sind optional und sorgen für
weiter knofigurations Möglichkeiten.

**INFO:** Beim wälen der Dropout Rate ist es wichtig die Größe des Netzwerkes in Betracht zu ziehen.
Obwohl eine Dropout Rate in der Regel zu natürlicheren Ergebnissen führen kann ist es wahrscheinlich, dass eine zu hohe
Dropout Rate bei einem kleinen Netzwerk eher für schlechtere Ergebnisse sorgen wird. <br>
**Empfehlung:** Nach testen mit ca. 128 - 256 hidden nodes ist eine Dropout Rate zwischen 0.0 und 0.2 zu empfehlen.
Ab 256 hidden nodes ist es auch möglich höher zu gehen, allerdings ist ein wert von mehr als 0.4 -0.5 nicht empfehlenswert.

### Beispiel

```bash
./out/mpt_nn -m2 -t60000 -i784 -h128 -o10 -e10 -l0.1 -d0.1
```

Startet das Training mit dem MNIST-Datensatz mit:

- 60000 Trainingsdaten
- 784 Eingangsneuronen
- 10 versteckte schichten
- 10 Outputs
- 10 epochen
- Einer Lernrate von 0.1
- Einer Dropout Rate von 10%(0.1)

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

## R Plot

Für die Visualisierung der Benchmark-Ergebnisse wurde das Programm R genutzt. Dieses kann über folgende Kommandozeile installiert werden:

```bash
sudo apt-get install r-base
```

Anschließend müssen nachfolgende Bibliotheken in R installiert werden. Öffnen Sie dafür eine R-Instanz durch den Befehl:

```bash
R
```

In der R-Konsole können die notwendigen Bibliotheken mit dem folgenden Code installiert werden:

```bash
install.packages(c("ggplot2", "dplyr", "tidyr", "readr"))
```

Zur Visualisierung der Ergebnisse stehen abschließend zwei Skripte bereit. Das erste Skript dient der Ausgabe der mit Hyperfine ermittelten Benchmarkergebnisse, während das zweite eine grafische Darstellung der ermittelten Genauigkeiten bereitstellt.
Mit folgendem Befehl können die Benchmark-Ergebnisse ausgegeben werden:

```bash
make plot-benchmark
```

Analog dazu kann mit dem nachfolgenden Befehl eine grafische Auswertung der Genauigkeit generiert werden:

```bash
make plot-accuracy
```

Die Ergebnisse beider Auswertungen stehen im entsprechenden Benchmark-Verzeichnis zur Verfügung.

## Dokumentation
### Doxygen
Neben der vorhandenen Wissenschaftlichen Arbeit zum Projekt sind alle funktionen mittels doxygen Kommentaren beschrieben.
Doxygen ist ein automatisiertes Dokumentationswerkzeug, das Quellcode analysiert und daraus strukturierte, lesbare Dokumentationen in verschiedenen Formaten generiert.
Ein Doxyfile, das für die Konfiguration der doxygen Dokumentation gebraucht wird, ist in dem Projektordner zu finden.
Doxygen kann über die Kommandozeile installieren.
```bash
sudo apt-get install doxygen
```
Anschließen lässt sich für eine detaillierte übersicht der Funktionen des mpt_nn, eine doxygen Dokumentation mit make erstellen.<br>
```bash
make doxygen
```
Der Befehlt baut ein doxygen Projekt in dem ordner Documentation und compiliert eine PDF in dem dazugehörigen latex ordner.