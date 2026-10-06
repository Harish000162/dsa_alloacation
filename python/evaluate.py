"""
Benchmark Evaluation and Visualization Module
=============================================
Fairly evaluates and compares:
  - MODE 1: Fixed Channel Allocation
  - MODE 2: Random Channel Selection
  - MODE 3: MARL Channel Selection (Trained Multi-Agent Policy)

Generates publication-quality comparative plots into results/plots/
and exports numerical summaries to results/csv/evaluation_comparison.csv.
"""

import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import metrics
from environment import get_environment, NodeObservation
from marl_agent import MultiAgentSpectrumController
from data_logger import save_evaluation_results


def run_benchmark_mode(
    mode_name: str,
    controller: MultiAgentSpectrumController,
    num_steps: int = 100
) -> List[Dict[str, Any]]:
    """
    Runs a single benchmark mode under identical environmental conditions.
    """
    env = get_environment(simulation_mode=config.SIMULATION_MODE)
    obs_dict = env.reset()
    records: List[Dict[str, Any]] = []

    for step in range(1, num_steps + 1):
        actions: Dict[int, int] = {}

        if mode_name == "Fixed":
            # All nodes fixed to Channel 1 (or statically assigned)
            actions = {nid: config.AVAILABLE_CHANNELS[0] for nid in obs_dict.keys()}
        elif mode_name == "Random":
            # Uniform random channel choice
            actions = {nid: random.choice(config.AVAILABLE_CHANNELS) for nid in obs_dict.keys()}
        elif mode_name == "MARL":
            # Pure exploitation using learned Q-tables
            actions = controller.select_actions(obs_dict, explore=False)
        else:
            raise ValueError(f"Unknown benchmark mode: {mode_name}")

        next_obs_dict, rewards, done, info = env.step(actions)

        throughputs = [obs.throughput_mbps for obs in next_obs_dict.values()]
        fairness = metrics.calculate_jains_fairness_index(throughputs)

        timestamp = time.time()
        for nid, obs in next_obs_dict.items():
            records.append({
                "mode": mode_name,
                "step": step,
                "node_id": nid,
                "current_channel": obs.current_channel,
                "selected_channel": actions.get(nid, obs.current_channel),
                "rssi": obs.rssi,
                "throughput_mbps": obs.throughput_mbps,
                "delay_ms": obs.delay_ms,
                "packet_loss_percent": obs.packet_loss_percent,
                "traffic_load": obs.traffic_load_mbps,
                "reward": rewards.get(nid, 0.0),
                "fairness_index": fairness,
                "timestamp": timestamp
            })

        obs_dict = next_obs_dict

    return records


def generate_all_plots(df: pd.DataFrame) -> None:
    """
    Generates all 10 required research visualization figures and saves to results/plots/.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {"MARL": "#1a73e8", "Random": "#fbbc04", "Fixed": "#ea4335"}

    # Filter by mode
    df_marl = df[df["mode"] == "MARL"]
    df_rand = df[df["mode"] == "Random"]
    df_fixed = df[df["mode"] == "Fixed"]

    # --------------------------------------------------------------------------
    # 1. RSSI vs Time (MARL Mode)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for nid in sorted(df_marl["node_id"].unique()):
        sub = df_marl[df_marl["node_id"] == nid]
        ax.plot(sub["step"], sub["rssi"], marker="o", markersize=3, label=f"Node {nid}")
    ax.set_title("1. RSSI vs Time (MARL Adaptive Selection)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("RSSI (dBm)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "1_rssi_vs_time.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 2. Throughput vs Time (Node-level and Aggregate)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    step_marl_tp = df_marl.groupby("step")["throughput_mbps"].mean()
    step_rand_tp = df_rand.groupby("step")["throughput_mbps"].mean()
    step_fixed_tp = df_fixed.groupby("step")["throughput_mbps"].mean()

    ax.plot(step_marl_tp.index, step_marl_tp.values, color=colors["MARL"], linewidth=2.5, label="MARL (Adaptive)")
    ax.plot(step_rand_tp.index, step_rand_tp.values, color=colors["Random"], linewidth=1.8, linestyle="--", label="Random Baseline")
    ax.plot(step_fixed_tp.index, step_fixed_tp.values, color=colors["Fixed"], linewidth=1.8, linestyle=":", label="Fixed Baseline")
    ax.set_title("2. Average Per-Node Throughput vs Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Throughput (Mbps)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "2_throughput_vs_time.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 3. Delay vs Time
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    step_marl_del = df_marl.groupby("step")["delay_ms"].mean()
    step_rand_del = df_rand.groupby("step")["delay_ms"].mean()
    step_fixed_del = df_fixed.groupby("step")["delay_ms"].mean()

    ax.plot(step_marl_del.index, step_marl_del.values, color=colors["MARL"], linewidth=2.5, label="MARL")
    ax.plot(step_rand_del.index, step_rand_del.values, color=colors["Random"], linewidth=1.8, linestyle="--", label="Random")
    ax.plot(step_fixed_del.index, step_fixed_del.values, color=colors["Fixed"], linewidth=1.8, linestyle=":", label="Fixed")
    ax.set_title("3. Packet Transmission Delay vs Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Delay (ms)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "3_delay_vs_time.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 4. Packet Loss % vs Time
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    step_marl_loss = df_marl.groupby("step")["packet_loss_percent"].mean()
    step_rand_loss = df_rand.groupby("step")["packet_loss_percent"].mean()
    step_fixed_loss = df_fixed.groupby("step")["packet_loss_percent"].mean()

    ax.plot(step_marl_loss.index, step_marl_loss.values, color=colors["MARL"], linewidth=2.5, label="MARL")
    ax.plot(step_rand_loss.index, step_rand_loss.values, color=colors["Random"], linewidth=1.8, linestyle="--", label="Random")
    ax.plot(step_fixed_loss.index, step_fixed_loss.values, color=colors["Fixed"], linewidth=1.8, linestyle=":", label="Fixed")
    ax.set_title("4. Packet Loss Rate vs Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Packet Loss (%)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "4_packet_loss_vs_time.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 5. Reward vs Mode
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    mean_rewards = [df_marl["reward"].mean(), df_rand["reward"].mean(), df_fixed["reward"].mean()]
    modes = ["MARL", "Random", "Fixed"]
    bar_colors = [colors["MARL"], colors["Random"], colors["Fixed"]]
    bars = ax.bar(modes, mean_rewards, color=bar_colors, width=0.5, edgecolor="black", alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("5. Average Reward per Step across Allocation Policies", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "5_reward_vs_episode.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 6. Channel Selection Frequency (Distribution across Ch 1, 6, 11)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(config.AVAILABLE_CHANNELS))
    width = 0.25

    marl_counts = [sum(df_marl["selected_channel"] == ch) for ch in config.AVAILABLE_CHANNELS]
    rand_counts = [sum(df_rand["selected_channel"] == ch) for ch in config.AVAILABLE_CHANNELS]
    fixed_counts = [sum(df_fixed["selected_channel"] == ch) for ch in config.AVAILABLE_CHANNELS]

    ax.bar(x - width, marl_counts, width, label="MARL", color=colors["MARL"], edgecolor="black")
    ax.bar(x, rand_counts, width, label="Random", color=colors["Random"], edgecolor="black")
    ax.bar(x + width, fixed_counts, width, label="Fixed", color=colors["Fixed"], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Channel {ch}" for ch in config.AVAILABLE_CHANNELS], fontsize=11)
    ax.set_title("6. Channel Selection Frequency Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Selection Count")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "6_channel_selection_frequency.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 7. MARL vs Random Throughput
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(step_marl_tp.index, step_marl_tp.values, color=colors["MARL"], linewidth=2.2, label="MARL")
    ax.plot(step_rand_tp.index, step_rand_tp.values, color=colors["Random"], linewidth=2.0, linestyle="--", label="Random")
    ax.fill_between(step_marl_tp.index, step_marl_tp.values, step_rand_tp.values, where=(step_marl_tp.values >= step_rand_tp.values), color=colors["MARL"], alpha=0.2, label="MARL Gain")
    ax.set_title("7. Throughput Comparison: MARL vs Random", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Throughput (Mbps)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "7_marl_vs_random_throughput.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 8. MARL vs Fixed Channel Throughput
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(step_marl_tp.index, step_marl_tp.values, color=colors["MARL"], linewidth=2.2, label="MARL")
    ax.plot(step_fixed_tp.index, step_fixed_tp.values, color=colors["Fixed"], linewidth=2.0, linestyle=":", label="Fixed Channel 1")
    ax.fill_between(step_marl_tp.index, step_marl_tp.values, step_fixed_tp.values, where=(step_marl_tp.values >= step_fixed_tp.values), color=colors["MARL"], alpha=0.2, label="MARL Gain")
    ax.set_title("8. Throughput Comparison: MARL vs Fixed Channel", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Throughput (Mbps)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "8_marl_vs_fixed_throughput.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 9. MARL vs Random Packet Loss
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(step_marl_loss.index, step_marl_loss.values, color=colors["MARL"], linewidth=2.2, label="MARL")
    ax.plot(step_rand_loss.index, step_rand_loss.values, color=colors["Random"], linewidth=2.0, linestyle="--", label="Random")
    ax.set_title("9. Packet Loss Comparison: MARL vs Random", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Packet Loss (%)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "9_marl_vs_random_packet_loss.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 10. MARL vs Fixed Packet Loss
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(step_marl_loss.index, step_marl_loss.values, color=colors["MARL"], linewidth=2.2, label="MARL")
    ax.plot(step_fixed_loss.index, step_fixed_loss.values, color=colors["Fixed"], linewidth=2.0, linestyle=":", label="Fixed Channel 1")
    ax.set_title("10. Packet Loss Comparison: MARL vs Fixed Channel", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Packet Loss (%)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "10_marl_vs_fixed_packet_loss.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 11. Comprehensive Multi-Metric Comparison Bar Chart
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Throughput
    axes[0, 0].bar(modes, [df_marl["throughput_mbps"].mean(), df_rand["throughput_mbps"].mean(), df_fixed["throughput_mbps"].mean()], color=bar_colors, edgecolor="black")
    axes[0, 0].set_title("Mean Throughput (Mbps) [Higher is Better]", fontweight="bold")
    axes[0, 0].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Delay
    axes[0, 1].bar(modes, [df_marl["delay_ms"].mean(), df_rand["delay_ms"].mean(), df_fixed["delay_ms"].mean()], color=bar_colors, edgecolor="black")
    axes[0, 1].set_title("Mean Latency (ms) [Lower is Better]", fontweight="bold")
    axes[0, 1].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Packet Loss
    axes[1, 0].bar(modes, [df_marl["packet_loss_percent"].mean(), df_rand["packet_loss_percent"].mean(), df_fixed["packet_loss_percent"].mean()], color=bar_colors, edgecolor="black")
    axes[1, 0].set_title("Mean Packet Loss (%) [Lower is Better]", fontweight="bold")
    axes[1, 0].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Fairness
    axes[1, 1].bar(modes, [df_marl["fairness_index"].mean(), df_rand["fairness_index"].mean(), df_fixed["fairness_index"].mean()], color=bar_colors, edgecolor="black")
    axes[1, 1].set_title("Jain's Fairness Index [Higher is Better]", fontweight="bold")
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("System-Wide Performance Comparison: MARL vs Baselines", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "11_performance_summary_grid.png", dpi=300)
    plt.close()

    print(f"[Evaluation] All 11 publication-quality plots successfully saved to: {config.PLOTS_DIR}")


def evaluate(num_steps: int = 100) -> None:
    """
    Main evaluation pipeline comparing MARL against Fixed and Random baselines.
    """
    print("\n" + "=" * 65)
    print("      MARL DYNAMIC SPECTRUM ALLOCATION: BENCHMARK EVALUATION")
    print("=" * 65)
    print(f"Evaluation Steps per Policy: {num_steps}")
    print(f"Evaluating Policies: [1] Fixed Channel  [2] Random  [3] MARL")
    print("=" * 65 + "\n")

    # Load trained controller
    controller = MultiAgentSpectrumController()
    loaded = controller.load_checkpoint()
    if not loaded:
        print("[Evaluation Warning] Trained checkpoint not found! Proceeding with fresh Q-tables.")

    all_records: List[Dict[str, Any]] = []

    # 1. Run Fixed Baseline
    print("--> Running Mode 1: Fixed Channel Allocation...")
    fixed_records = run_benchmark_mode("Fixed", controller, num_steps=num_steps)
    all_records.extend(fixed_records)

    # 2. Run Random Baseline
    print("--> Running Mode 2: Random Channel Selection...")
    rand_records = run_benchmark_mode("Random", controller, num_steps=num_steps)
    all_records.extend(rand_records)

    # 3. Run MARL Policy
    print("--> Running Mode 3: MARL Adaptive Spectrum Allocation...")
    marl_records = run_benchmark_mode("MARL", controller, num_steps=num_steps)
    all_records.extend(marl_records)

    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    save_evaluation_results(all_records, "evaluation_comparison.csv")

    # Generate all plots
    print("--> Generating all 10+ research plots...")
    generate_all_plots(df)

    # Print Summary Comparison Table
    print("\n" + "=" * 70)
    print("                     BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25}{'Fixed Channel':<15}{'Random':<15}{'MARL (Trained)':<15}")
    print("-" * 70)

    summary = {}
    for mode in ["Fixed", "Random", "MARL"]:
        sub = df[df["mode"] == mode]
        summary[mode] = {
            "tp": sub["throughput_mbps"].mean(),
            "delay": sub["delay_ms"].mean(),
            "loss": sub["packet_loss_percent"].mean(),
            "fairness": sub["fairness_index"].mean(),
            "reward": sub["reward"].mean()
        }

    print(f"{'Throughput (Mbps)':<25}{summary['Fixed']['tp']:<15.2f}{summary['Random']['tp']:<15.2f}{summary['MARL']['tp']:<15.2f}")
    print(f"{'Packet Delay (ms)':<25}{summary['Fixed']['delay']:<15.2f}{summary['Random']['delay']:<15.2f}{summary['MARL']['delay']:<15.2f}")
    print(f"{'Packet Loss (%)':<25}{summary['Fixed']['loss']:<15.2f}{summary['Random']['loss']:<15.2f}{summary['MARL']['loss']:<15.2f}")
    print(f"{'Jain Fairness Index':<25}{summary['Fixed']['fairness']:<15.3f}{summary['Random']['fairness']:<15.3f}{summary['MARL']['fairness']:<15.3f}")
    print(f"{'Average Reward':<25}{summary['Fixed']['reward']:<15.3f}{summary['Random']['reward']:<15.3f}{summary['MARL']['reward']:<15.3f}")
    print("-" * 70)

    # Compute percentage improvements
    fixed_tp = summary["Fixed"]["tp"]
    marl_tp = summary["MARL"]["tp"]
    rand_tp = summary["Random"]["tp"]
    tp_gain_fixed = ((marl_tp - fixed_tp) / max(0.01, fixed_tp)) * 100.0
    tp_gain_rand = ((marl_tp - rand_tp) / max(0.01, rand_tp)) * 100.0

    print(f"MARL Throughput Gain over Fixed:  +{tp_gain_fixed:.1f}%")
    print(f"MARL Throughput Gain over Random: +{tp_gain_rand:.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate()
