# ESP32-Based Multi-Agent Reinforcement Learning (MARL) Testbed for Dynamic Spectrum Allocation

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PlatformIO](https://img.shields.io/badge/PlatformIO-Core%206.0%2B-orange.svg)](https://platformio.org/)
[![ESP32](https://img.shields.io/badge/Hardware-ESP32--WROOM--32-green.svg)](https://www.espressif.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Overview

In dense wireless deployments, unlicensed 2.4 GHz spectrum experiences severe co-channel interference and dynamic contention. This project implements a complete **Multi-Agent Reinforcement Learning (MARL)** system that dynamically allocates non-overlapping 802.11 channels (Channels 1, 6, and 11) among multiple autonomous wireless nodes.

The system features:
- **Dual-Mode Operation**: A high-fidelity synthetic 2.4 GHz RF simulation mode (runnable immediately without hardware) and a real ESP32 hardware testbed mode.
- **Independent Q-Learning (IQL)**: Each node functions as an autonomous reinforcement learning agent learning to maximize throughput, minimize latency and packet drops, and enforce network-wide fairness.
- **Fair Baseline Benchmarks**: Automated comparative evaluation against **Fixed Channel** and **Random Channel** allocation.
- **Publication-Quality Analytics**: Automatically produces 11 publication-ready figures and structured CSV telemetry logs.

---

## 2. Project Directory Structure

```
MARL_Dynamic_Spectrum/
│
├── README.md                           # Main user and setup guide
│
├── python/                             # Python MARL & Controller System
│   ├── config.py                       # Global settings, reward weights, hyperparameters
│   ├── metrics.py                      # Jain's fairness index, reward math, discretization
│   ├── data_logger.py                  # CSV telemetry logging and evaluation persistence
│   ├── environment.py                  # Dual environment (RF Simulation + ESP32 Hardware)
│   ├── marl_agent.py                   # Multi-agent independent Q-learning algorithms
│   ├── train.py                        # Training pipeline with live terminal dashboard
│   ├── evaluate.py                     # Scientific baseline evaluation (MARL vs Fixed vs Random)
│   ├── controller.py                   # Real-time interactive spectrum controller
│   ├── requirements.txt                # Python package dependencies
│   └── data/                           # Saved Q-table model checkpoints
│       └── marl_qtable_checkpoint.json
│
├── esp32_node/                         # ESP32 Microcontroller Firmware (PlatformIO)
│   ├── platformio.ini                  # Multi-node environments (Node 1 to Node 5)
│   └── src/
│       └── main.cpp                    # Production C++ firmware with REST API & Serial CLI
│
├── results/                            # Outputs & Diagnostics
│   ├── plots/                          # Generated high-resolution visualization graphs
│   └── csv/                            # Exported experimental telemetry datasets
│
└── docs/                               # Scientific & Technical Documentation
    └── implementation.md               # Detailed Dec-POMDP formulation & RF theory
```

---

## 3. File Descriptions

| File | Purpose |
| :--- | :--- |
| `python/config.py` | Central configuration containing hyperparameters ($\alpha, \gamma, \epsilon$), channel lists (`[1, 6, 11]`), reward weights, and `SIMULATION_MODE` toggle. |
| `python/metrics.py` | Implementation of multi-objective reward calculation, Jain's Fairness Index, and observation discretization. |
| `python/data_logger.py` | Structured CSV logger logging per-step telemetry (`timestamp, node_id, current_channel, selected_channel, rssi, throughput, delay, loss, reward`). |
| `python/environment.py` | Unified environment providing `SimulatedSpectrumEnvironment` (CSMA/CA contention + jammer) and `HardwareSpectrumEnvironment` (ESP32 REST/Serial). |
| `python/marl_agent.py` | Independent Q-Learning (IQL) implementation with Bellman TD updates, $\epsilon$-greedy exploration, and JSON model persistence. |
| `python/train.py` | Executes training loops across episodes, displays terminal dashboards, and plots training curves. |
| `python/evaluate.py` | Fair scientific benchmark comparing MARL, Random, and Fixed channel policies under identical RF conditions. Generates all 11 plots. |
| `python/controller.py` | Live execution runtime for hardware testbeds or real-time simulation with interactive terminal display. |
| `esp32_node/src/main.cpp` | Complete ESP32 firmware exposing HTTP REST endpoints (`/metrics`, `/channel`, `/scan`, `/traffic`) and USB Serial command parser. |
| `esp32_node/platformio.ini` | PlatformIO configuration with pre-configured build flags for flashing Node 1 through Node 5. |
| `docs/implementation.md` | In-depth academic report detailing Dec-POMDP mathematics, 802.11 spectral masks, and contention theory. |

---

## 4. Installation & Quick Start

### 4.1 Prerequisites
- Python 3.8+ (Compatible with Python 3.8 up to 3.14)
- Visual Studio Code with the **Python** extension
- Optional for hardware: **PlatformIO IDE** extension in VS Code

### 4.2 Python Virtual Environment Setup

Open a terminal inside `MARL_Dynamic_Spectrum`:

```bash
# 1. Create a Python virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r python/requirements.txt
```

---

## 5. Running the Simulation (No Hardware Needed)

The project ships configured in **Simulation Mode** by default (`SIMULATION_MODE = True` in `python/config.py`). You can immediately train, evaluate, and test the system.

### Step 1: Train the MARL Agent
Train 5 independent agents over 500 episodes (takes ~20 seconds):
```bash
python python/train.py
```
*Output*:
- Real-time terminal dashboard showing learning progress.
- Checkpoint saved to `python/data/marl_qtable_checkpoint.json`.
- Training curves saved to `results/plots/training_curves.png`.
- Per-step data logged to `results/csv/training_log.csv`.

### Step 2: Run Scientific Benchmark Evaluation
Compare trained MARL against Fixed Channel and Random Channel baselines:
```bash
python python/evaluate.py
```
*Output*:
- Detailed performance comparison table in your terminal.
- 11 publication-grade figures saved to `results/plots/`.
- Full comparison dataset saved to `results/csv/evaluation_comparison.csv`.

### Step 3: Run Interactive Controller
Run the live real-time controller loop:
```bash
python python/controller.py --steps 50 --interval 0.5
```

---

## 6. Understanding the Generated Graphs

The evaluation script generates 11 distinct research graphs in `results/plots/`:

1. **`1_rssi_vs_time.png`**: Tracks RSSI values for all 5 nodes over time, showing temporal signal variations and shadowing effects.
2. **`2_throughput_vs_time.png`**: Compares the average per-node throughput achieved by MARL against Random and Fixed baselines.
3. **`3_delay_vs_time.png`**: Demonstrates how MARL keeps packet transmission delay low by avoiding crowded channels.
4. **`4_packet_loss_vs_time.png`**: Shows packet drop rates across time. Fixed channels suffer high loss (>40%) due to contention, while MARL suppresses drops.
5. **`5_reward_vs_episode.png`**: Bar chart showing mean reward per decision step across the three policies.
6. **`6_channel_selection_frequency.png`**: Histogram showing how nodes distribute themselves across Channels 1, 6, and 11. Demonstrates load balancing.
7. **`7_marl_vs_random_throughput.png`**: Highlights the throughput gain region between MARL and uncoordinated random selection.
8. **`8_marl_vs_fixed_throughput.png`**: Shows significant throughput improvement (typically >200% gain) over single-channel fixed allocation.
9. **`9_marl_vs_random_packet_loss.png`**: Direct comparison of packet loss rates between MARL and Random selection.
10. **`10_marl_vs_fixed_packet_loss.png`**: Compares packet loss between MARL and static Channel 1 allocation.
11. **`11_performance_summary_grid.png`**: 4-panel dashboard summarizing Mean Throughput, Latency, Packet Loss, and Jain's Fairness Index across all methods.

---

## 7. ESP32 Hardware Setup & Configuration

### 7.1 Hardware Bill of Materials
- 5 × ESP32 DevKit V1 (ESP32-WROOM-32)
- 5 × Micro-USB data cables
- 1 × Powered 5V USB Hub or breadboard power rail
- 1 × 2.4 GHz Wi-Fi Router or Mobile Hotspot

### 7.2 Hardware Wiring & Power
Connect each ESP32 via USB to your PC or a powered USB hub:

```
[PC / Laptop USB Hub]
      │
      ├── COM3 ──> [ESP32 Node 1] ──> Status LED on GPIO 2
      ├── COM4 ──> [ESP32 Node 2] ──> Status LED on GPIO 2
      ├── COM5 ──> [ESP32 Node 3] ──> Status LED on GPIO 2
      ├── COM6 ──> [ESP32 Node 4] ──> Status LED on GPIO 2
      └── COM7 ──> [ESP32 Node 5] ──> Status LED on GPIO 2
```

### 7.3 Flashing ESP32 Firmware with PlatformIO
The project includes pre-configured environments for all 5 nodes in `esp32_node/platformio.ini`.

To flash each node:
1. Connect Node 1 to USB and identify its COM port (e.g. `COM3`).
2. Run:
   ```bash
   cd esp32_node
   pio run -e node1 --target upload --upload-port COM3
   ```
3. Connect Node 2 (e.g. `COM4`) and run:
   ```bash
   pio run -e node2 --target upload --upload-port COM4
   ```
4. Repeat for Node 3 (`-e node3`), Node 4 (`-e node4`), and Node 5 (`-e node5`).

To monitor serial debug output on any node:
```bash
pio device monitor -b 115200 --port COM3
```

---

## 8. Real Hardware Mode Execution

### 8.1 Network Configuration
1. Set up a 2.4 GHz Wi-Fi access point or hotspot with:
   - **SSID**: `MARL_SPECTRUM_TESTBED`
   - **Password**: `SpectrumAccess2026`
   *(Or adjust credentials in `esp32_node/src/main.cpp` and `python/config.py`)*.
2. Note the IP addresses assigned to the nodes by your router (or assign static DHCP leases).
3. Update `ESP32_NODES` in `python/config.py`:
   ```python
   ESP32_NODES = {
       1: {"ip": "192.168.4.2", "port": 80, "serial_port": "COM3"},
       2: {"ip": "192.168.4.3", "port": 80, "serial_port": "COM4"},
       3: {"ip": "192.168.4.4", "port": 80, "serial_port": "COM5"},
       4: {"ip": "192.168.4.5", "port": 80, "serial_port": "COM6"},
       5: {"ip": "192.168.4.6", "port": 80, "serial_port": "COM7"},
   }
   ```

### 8.2 Switch to Hardware Mode
In `python/config.py`, change:
```python
SIMULATION_MODE = False
```

### 8.3 Run Live Hardware Testing
```bash
# Run controller on real hardware
python python/controller.py --steps 100 --interval 1.0
```

---

## 9. Important Wi-Fi Networking Realities & Safe Architecture

> [!IMPORTANT]
> **Commercial Wi-Fi Limitation**: When independent ESP32 clients connect in Station (STA) mode to a standard commercial Wi-Fi Access Point, their physical radio transceiver is slaved to the Access Point's beacon channel. A client cannot independently change the router's operating channel.

### How this project accurately and safely addresses this:
1. **RSSI & Channel Scanning**: The ESP32 utilizes `WiFi.scanNetworks()` to periodically sweep Channels 1, 6, and 11, measuring real physical ambient interference and RSSI across the 2.4 GHz band.
2. **Physical Radio Frequency Switching**: In ESP-NOW or raw 802.11 promiscuous mode, `esp_wifi_set_channel()` directly re-tunes the physical local radio hardware.
3. **Transparent Reporting**: The ESP32 firmware reports `hardware_channel_locked: true/false`. The controller never falsely pretends a commercial router channel was changed when it was not.

---

## 10. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Python was not found` | Python not in Windows PATH | Ensure virtual environment is activated (`.venv\Scripts\activate`) or call `python` via full path. |
| `pio: Command not found` | PlatformIO CLI not on PATH | Install via `pip install platformio` inside your `.venv`. |
| ESP32 Upload Fails (`Timed out waiting for packet header`) | Bootloader not triggered | Hold down the **BOOT** button on the ESP32 board while the upload starts. |
| Node HTTP Timeout in Controller | ESP32 IP address mismatch or Wi-Fi dropped | Verify IP in router DHCP client list and update `python/config.py`. The controller gracefully continues if one node drops. |
| Permission Denied on COM Port | Serial monitor still open in another window | Close any active PuTTY, Arduino IDE, or PlatformIO monitor on that COM port before flashing. |
