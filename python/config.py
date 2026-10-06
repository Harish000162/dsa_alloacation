"""
Configuration Module for MARL Dynamic Spectrum Allocation
==========================================================
Provides global hyperparameters, reward weights, simulation models,
and hardware communication settings for the ESP32 testbed.
"""

from pathlib import Path
from typing import Dict, List, Any

# ==============================================================================
# 1. CORE TESTBED TOPOLOGY & EXPERIMENT MODES
# ==============================================================================

# Number of autonomous wireless nodes (Agents)
NUM_NODES: int = 5

# 2.4 GHz orthogonal Wi-Fi channels (non-overlapping in 802.11 b/g/n)
AVAILABLE_CHANNELS: List[int] = [1, 6, 11]

# Mode Toggle:
# True  -> Run synthetic high-fidelity 2.4 GHz RF simulation (No hardware needed)
# False -> Connect to physical ESP32 nodes over Wi-Fi / Serial
SIMULATION_MODE: bool = True

# Random seed for reproducible evaluation and training
RANDOM_SEED: int = 42


# ==============================================================================
# 2. MARL REINFORCEMENT LEARNING HYPERPARAMETERS
# ==============================================================================

# Training duration
EPISODES: int = 500
STEPS_PER_EPISODE: int = 50

# Independent Q-Learning parameters
LEARNING_RATE: float = 0.1          # Alpha: Learning step size
DISCOUNT_FACTOR: float = 0.9        # Gamma: Future reward discount factor

# Exploration vs Exploitation (Epsilon-greedy schedule)
EPSILON_START: float = 1.0          # Initial exploration rate (100% random actions)
EPSILON_MIN: float = 0.05           # Minimum exploration floor
EPSILON_DECAY: float = 0.995        # Exponential decay factor per episode

# State Space Discretization (Tabular Q-Learning)
# Observation levels: RSSI (Weak, Fair, Good), Interference (Low, Med, High), Traffic (Low, High)
NUM_RSSI_BINS: int = 3              # [-100 to -75 dBm: Weak, -75 to -60: Fair, > -60: Good]
NUM_INTERFERENCE_BINS: int = 3      # [Low, Medium, High contention]
NUM_TRAFFIC_BINS: int = 2           # [Low, High burst]


# ==============================================================================
# 3. REWARD FUNCTION WEIGHTS & NORMALIZATION
# ==============================================================================
# Total Reward = (W_TP * norm_tp) - (W_DELAY * norm_delay) 
#                - (W_LOSS * norm_loss) - (W_INTERF * norm_interf) 
#                + (W_FAIR * fairness_bonus)

WEIGHT_THROUGHPUT: float = 1.0
WEIGHT_DELAY: float = 0.5
WEIGHT_PACKET_LOSS: float = 0.8
WEIGHT_INTERFERENCE: float = 0.6
WEIGHT_FAIRNESS: float = 0.4

# Normalization Upper Bounds (used to scale physical values to [0, 1])
MAX_EXPECTED_THROUGHPUT_MBPS: float = 30.0   # Theoretical practical max for single-stream ESP32
MAX_EXPECTED_DELAY_MS: float = 100.0         # Practical round-trip timeout threshold
MAX_EXPECTED_INTERFERENCE_DBM: float = 90.0  # Measured interference power relative to noise floor


# ==============================================================================
# 4. SIMULATION ENVIRONMENT PARAMETERS
# ==============================================================================
# Used when SIMULATION_MODE == True to synthesize realistic 2.4 GHz RF dynamics.

SIM_CHANNEL_BANDWIDTH_MHZ: float = 20.0
SIM_NOISE_FLOOR_DBM: float = -95.0
SIM_NODE_TX_POWER_DBM: float = 20.0          # Standard ESP32 transmit power (+20 dBm)

# CSMA/CA Contention Model:
# When multiple nodes select the same channel, co-channel contention causes backoff,
# increasing delay, causing packet collisions, and reducing per-node throughput.
SIM_CONTENTION_PENALTY_FACTOR: float = 0.45  # Throughput reduction exponent under sharing

# Dynamic External Interference (Simulates microwave oven or legacy AP on Channel 6)
SIM_EXTERNAL_INTERFERENCE_ENABLED: bool = True
SIM_EXTERNAL_INTERFERENCE_CHANNEL: int = 6
SIM_EXTERNAL_INTERFERENCE_START_STEP: int = 15
SIM_EXTERNAL_INTERFERENCE_END_STEP: int = 35
SIM_EXTERNAL_INTERFERENCE_POWER_DB: float = 18.0


# ==============================================================================
# 5. HARDWARE COMMUNICATION SETTINGS (ESP32 NODES)
# ==============================================================================
# Used when SIMULATION_MODE == False.
# Each ESP32 node runs an HTTP WebServer and exposes /metrics and /channel.

ESP32_COMM_MODE: str = "HTTP"  # Options: "HTTP" or "SERIAL"
HTTP_TIMEOUT_SECONDS: float = 2.0
SERIAL_BAUDRATE: int = 115200

# Mapping of Node IDs to their network endpoints or USB serial COM ports
ESP32_NODES: Dict[int, Dict[str, Any]] = {
    1: {"ip": "192.168.4.2", "port": 80, "serial_port": "COM3"},
    2: {"ip": "192.168.4.3", "port": 80, "serial_port": "COM4"},
    3: {"ip": "192.168.4.4", "port": 80, "serial_port": "COM5"},
    4: {"ip": "192.168.4.5", "port": 80, "serial_port": "COM6"},
    5: {"ip": "192.168.4.6", "port": 80, "serial_port": "COM7"},
}

# Wi-Fi Router Testbed Credentials (for ESP32 connection in STA mode)
WIFI_SSID: str = "MARL_SPECTRUM_TESTBED"
WIFI_PASSWORD: str = "SpectrumAccess2026"


# ==============================================================================
# 6. DIRECTORIES & DATA PATHS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON_DIR = BASE_DIR / "python"
DATA_DIR = PYTHON_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CSV_DIR = RESULTS_DIR / "csv"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINT_PATH = DATA_DIR / "marl_qtable_checkpoint.json"

# Ensure all target folders exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 7. DASHBOARD & LOGGING SETTINGS
# ==============================================================================

DASHBOARD_REFRESH_INTERVAL: int = 5   # Steps between terminal dashboard prints
SAVE_CHECKPOINT_EVERY_EPISODES: int = 50
LOG_EVERY_N_STEPS: int = 1            # Log every single step to CSV
