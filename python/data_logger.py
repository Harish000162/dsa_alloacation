"""
Data Logger Module for Dynamic Spectrum Allocation
==================================================
Handles persistent CSV logging of per-step agent decisions, wireless telemetry,
reward feedback, and summary statistics.
"""

import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

import config


class ExperimentLogger:
    """
    Thread-safe, robust CSV data logger for reinforcement learning trials
    and live hardware telemetry.
    """
    
    CSV_HEADERS: List[str] = [
        "timestamp",
        "node_id",
        "current_channel",
        "selected_channel",
        "rssi",
        "throughput_mbps",
        "delay_ms",
        "packet_loss_percent",
        "traffic_load",
        "reward",
        "episode",
        "step"
    ]
    
    def __init__(self, filename: Optional[str] = None):
        if filename is None:
            time_str = time.strftime("%Y%m%d_%H%M%S")
            self.filepath = config.CSV_DIR / f"experiment_log_{time_str}.csv"
        else:
            self.filepath = config.CSV_DIR / filename
            
        self._init_csv()
        
    def _init_csv(self) -> None:
        """Initializes the CSV file with headers if it does not exist."""
        try:
            if not self.filepath.exists():
                with open(self.filepath, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.CSV_HEADERS)
        except Exception as e:
            print(f"[DataLogger Warning] Could not initialize CSV at {self.filepath}: {e}")
            
    def log_step(
        self,
        node_id: int,
        current_channel: int,
        selected_channel: int,
        rssi: float,
        throughput_mbps: float,
        delay_ms: float,
        packet_loss_percent: float,
        traffic_load: float,
        reward: float,
        episode: int,
        step: int,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Appends a single telemetry and decision record to the CSV log.
        """
        if timestamp is None:
            timestamp = time.time()
            
        row = [
            f"{timestamp:.3f}",
            int(node_id),
            int(current_channel),
            int(selected_channel),
            f"{float(rssi):.2f}",
            f"{float(throughput_mbps):.2f}",
            f"{float(delay_ms):.2f}",
            f"{float(packet_loss_percent):.2f}",
            f"{float(traffic_load):.2f}",
            f"{float(reward):.4f}",
            int(episode),
            int(step)
        ]
        
        try:
            with open(self.filepath, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()
        except Exception as e:
            print(f"[DataLogger Error] Failed to write record to {self.filepath}: {e}")

    def load_dataframe(self) -> pd.DataFrame:
        """Reads the currently written CSV file into a pandas DataFrame."""
        if not self.filepath.exists():
            return pd.DataFrame(columns=self.CSV_HEADERS)
        try:
            return pd.read_csv(self.filepath)
        except Exception as e:
            print(f"[DataLogger Error] Failed to load DataFrame: {e}")
            return pd.DataFrame(columns=self.CSV_HEADERS)


def save_evaluation_results(
    records: List[Dict[str, Any]],
    output_filename: str = "evaluation_comparison.csv"
) -> Path:
    """
    Saves cross-method evaluation metrics (MARL vs Random vs Fixed) into a clean CSV.
    """
    target_path = config.CSV_DIR / output_filename
    if not records:
        print("[DataLogger Warning] No evaluation records to save.")
        return target_path
        
    df = pd.DataFrame(records)
    df.to_csv(target_path, index=False)
    print(f"[DataLogger] Saved evaluation benchmark to: {target_path}")
    return target_path
