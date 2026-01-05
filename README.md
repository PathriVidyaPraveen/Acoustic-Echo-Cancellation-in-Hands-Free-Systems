# Acoustic Echo Cancellation in Hands-Free Systems
**Using Digital Signal Processing**

This project implements a real-time Acoustic Echo Cancellation (AEC) system using classical Digital Signal Processing techniques only.

The system is designed to cancel acoustic echo between a loudspeaker and microphone, handle double-talk scenarios robustly, and suppress residual echo artifacts using non-linear processing.

---

## 1. Project Features

* **Partitioned Block Frequency-Domain Adaptive Filter (PBFDAF)**
* Frequency-domain NLMS adaptation
* Coherence-based Double-Talk Detection (DTD)
* Residual Echo Suppression with gain smoothing (NLP)
* Offline simulation and evaluation
* Real-time microphone–speaker demo (PC)
* Modular DSP pipeline suitable for embedded porting

---

## 2. Folder Structure

```text
aec/
│
├── config/
│   └── aec_config.py       # Global DSP parameters
│
├── core/
│   ├── overlap_save.py     # FFT block buffering
│   ├── pbfdaf.py           # Partitioned FDAF filter
│   └── adaptive_update.py  # Frequency-domain NLMS
│
├── dtd/
│   └── coherence_dtd.py    # Double-talk detection
│
├── nlp/
│   └── smoothing.py        # NLP gain smoothing
│
├── simulation/
│   └── generate_signals.py # Synthetic test signals
│
├── realtime/
│   ├── audio_io.py         # Mic / speaker streaming
│   └── realtime_aec.py     # Real-time AEC pipeline
│
├── analysis/
│   ├── plot_erle_vs_time.py
│   ├── plot_filter_norm.py
│   ├── plot_dtd_behavior.py
│   ├── plot_nlp_gains.py
│   └── plot_waveform_comparison.py
│
├── results/
│   └── *.png               # Generated plots
│
├── main.py                 # Offline AEC simulation
└── README.md               # This file
```
--- 

## 3. Requirements
Python 3.9 or higher

Required Python packages:

numpy

matplotlib

sounddevice (for real-time demo only)

Installation:

```bash
pip install numpy matplotlib sounddevice
```
NOTE: > * On Linux, PortAudio must be installed separately for sounddevice.

On Windows, sounddevice works out-of-the-box.
---
## 4. Configuration
All DSP parameters are defined in config/aec_config.py.

Important parameters include:

SAMPLE_RATE

BLOCK_SIZE

FFT_SIZE

NUM_PARTITIONS

STEP_SIZE

DTD threshold

NLP aggressiveness and gain floor

You can tune these parameters to study convergence and stability.
---
## 5. Running Offline Simulation
This runs the complete AEC pipeline on synthetic signals and reports ERLE. Use this for algorithm validation.

From the project root directory:

```bash

python main.py
```
Output:

Console summary (ERLE, double-talk blocks)

No audio playback

## 6. Generating Analysis Plots
All plots are saved to the results/ directory. Run each script from the project root using the module flag (-m).

ERLE vs Time:

```bash
python -m analysis.plot_erle_vs_time
```
Filter Norm vs Time:

```bash
python -m analysis.plot_filter_norm
```
Double-Talk Detector Behavior:

```bash

python -m analysis.plot_dtd_behavior
```
NLP Gain Analysis:

```bash

python -m analysis.plot_nlp_gains
```
Waveform Comparison:

```bash
python -m analysis.plot_waveform_comparison
```
## 7. Real-Time AEC Demo (PC)
WARNING: Use headphones to avoid acoustic feedback.

Run from the project root:

```bash

python -m realtime.realtime_aec
```
Behavior:

Microphone input is processed in real time.

Loudspeaker echo is cancelled.

Adaptation is suspended during detected double-talk.

Troubleshooting: If loud artifacts occur:

Lower STEP_SIZE.

Ensure adapt = not is_double_talk.

Reduce NLP aggressiveness.
---
## 8. Expected Results
ERLE: ~15–30 dB (simulation dependent)

Stability: Stable filter norm (no divergence)

DTD: Correct double-talk detection

Demo: Audible echo suppression in real-time
---
## 9. Notes on Double-Talk Handling
Adaptive filter updates are disabled during double-talk.

Echo suppression (NLP) continues during double-talk.

This mirrors the behavior of commercial AEC systems.
---
## 10. Limitations
Linear echo path assumption.

Single-microphone system.

No loudspeaker non-linearity modeling.

Fixed thresholds (can be extended).
---
## 11. Future Extensions
Embedded (ESP32) fixed-point implementation.

Multi-microphone AEC.

Dynamic threshold adaptation.

Non-linear echo modeling.
---
