"""
Metrics and Reward Module for Dynamic Spectrum Allocation
=========================================================
Implements Jain's Fairness Index, state discretization, normalization utilities,
and the multi-objective reward function.
"""

from typing import List, Tuple, Dict, Any
import numpy as np

import config


def calculate_jains_fairness_index(throughputs: List[float]) -> float:
    """
    Computes Jain's Fairness Index for resource allocation across N agents.
    
    Formula:
        J(T_1, T_2, ..., T_N) = (sum(T_i))^2 / (N * sum(T_i^2))
    
    Properties:
        - Bounded in [1/N, 1.0]
        - J = 1.0 indicates perfectly equal throughput distribution
        - J = 1/N indicates severe starvation (one node consumes all bandwidth)
    
    Args:
        throughputs: List of individual agent throughput values in Mbps.
        
    Returns:
        Fairness index bounded between 0.0 and 1.0.
    """
    if not throughputs:
        return 1.0
    
    t_array = np.array(throughputs, dtype=np.float64)
    # Ensure non-negative
    t_array = np.clip(t_array, 0.0, None)
    
    sum_t = np.sum(t_array)
    sum_sq_t = np.sum(t_array ** 2)
    n = len(t_array)
    
    if sum_sq_t <= 1e-9 or sum_t <= 1e-9:
        # If all throughputs are 0, fairness is treated as neutral (1.0)
        return 1.0
    
    fairness = (sum_t ** 2) / (n * sum_sq_t)
    return float(np.clip(fairness, 0.0, 1.0))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Normalizes a physical metric linearly to the interval [0.0, 1.0].
    """
    if max_val <= min_val:
        return 0.0
    clipped = np.clip(value, min_val, max_val)
    return float((clipped - min_val) / (max_val - min_val))


def calculate_reward(
    throughput_mbps: float,
    delay_ms: float,
    packet_loss_percent: float,
    interference_dbm: float,
    fairness_bonus: float
) -> float:
    """
    Calculates multi-objective scalar reward for dynamic spectrum allocation.
    
    Objective:
        Reward = + (W_TP * normalized_throughput)
                 - (W_DELAY * normalized_delay)
                 - (W_LOSS * normalized_packet_loss)
                 - (W_INTERF * normalized_interference)
                 + (W_FAIR * fairness_bonus)
                 
    All weights are defined in config.py.
    """
    # 1. Normalized throughput: 0 Mbps -> 0.0, MAX_TP -> 1.0
    norm_tp = normalize(throughput_mbps, 0.0, config.MAX_EXPECTED_THROUGHPUT_MBPS)
    
    # 2. Normalized delay: 0 ms -> 0.0, MAX_DELAY -> 1.0
    norm_delay = normalize(delay_ms, 0.0, config.MAX_EXPECTED_DELAY_MS)
    
    # 3. Normalized packet loss: 0% -> 0.0, 100% -> 1.0
    norm_loss = normalize(packet_loss_percent, 0.0, 100.0)
    
    # 4. Normalized interference power relative to noise floor:
    # Noise floor (-95 dBm) is lowest (0.0 interference), higher dBm is stronger interference
    interf_rel = max(0.0, interference_dbm - config.SIM_NOISE_FLOOR_DBM)
    norm_interf = normalize(interf_rel, 0.0, config.MAX_EXPECTED_INTERFERENCE_DBM)
    
    # 5. Fairness is already bounded in [0.0, 1.0]
    norm_fair = float(np.clip(fairness_bonus, 0.0, 1.0))
    
    reward = (
        (config.WEIGHT_THROUGHPUT * norm_tp)
        - (config.WEIGHT_DELAY * norm_delay)
        - (config.WEIGHT_PACKET_LOSS * norm_loss)
        - (config.WEIGHT_INTERFERENCE * norm_interf)
        + (config.WEIGHT_FAIRNESS * norm_fair)
    )
    return float(reward)


def discretize_state(
    current_channel: int,
    rssi_dbm: float,
    interference_dbm: float,
    traffic_load_mbps: float
) -> Tuple[int, int, int, int]:
    """
    Discretizes continuous environmental measurements into a compact state tuple
    suitable for Tabular Q-Learning.
    
    State Tuple:
        (channel_idx, rssi_bin, interference_bin, traffic_bin)
        
    Discretization thresholds:
        - Channel Index: 0 for Ch 1, 1 for Ch 6, 2 for Ch 11
        - RSSI:
            0 (Weak):   RSSI < -75 dBm
            1 (Fair):   -75 dBm <= RSSI < -60 dBm
            2 (Good):   RSSI >= -60 dBm
        - Interference:
            0 (Low):    Interference < -80 dBm
            1 (Medium): -80 dBm <= Interference < -65 dBm
            2 (High):   Interference >= -65 dBm
        - Traffic Load:
            0 (Low):    Load < 10.0 Mbps
            1 (High):   Load >= 10.0 Mbps
    """
    # Channel index
    if current_channel in config.AVAILABLE_CHANNELS:
        ch_idx = config.AVAILABLE_CHANNELS.index(current_channel)
    else:
        ch_idx = 0
        
    # RSSI Bin
    if rssi_dbm < -75.0:
        rssi_bin = 0
    elif rssi_dbm < -60.0:
        rssi_bin = 1
    else:
        rssi_bin = 2
        
    # Interference Bin
    if interference_dbm < -80.0:
        interf_bin = 0
    elif interference_dbm < -65.0:
        interf_bin = 1
    else:
        interf_bin = 2
        
    # Traffic Load Bin
    traffic_bin = 1 if traffic_load_mbps >= 10.0 else 0
    
    return (ch_idx, rssi_bin, interf_bin, traffic_bin)


def state_to_key(state_tuple: Tuple[int, int, int, int]) -> str:
    """Converts a state tuple to a string key for JSON serialization."""
    return f"{state_tuple[0]}_{state_tuple[1]}_{state_tuple[2]}_{state_tuple[3]}"


def key_to_state(key_str: str) -> Tuple[int, int, int, int]:
    """Parses a string key back into a state tuple."""
    parts = [int(p) for p in key_str.split("_")]
    return (parts[0], parts[1], parts[2], parts[3])
