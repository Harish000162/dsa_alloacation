"""
Central Controller for Dynamic Spectrum Allocation Testbed
==========================================================
Executes real-time dynamic spectrum allocation inference using trained MARL policy
or live learning across physical ESP32 nodes / high-fidelity simulation.

Displays formatted real-time terminal dashboard, logs telemetry to CSV,
and handles hardware communication timeouts gracefully.
"""

import sys
import time
import argparse
from typing import Dict, Any, Optional

import config
import metrics
from environment import get_environment, NodeObservation
from marl_agent import MultiAgentSpectrumController
from data_logger import ExperimentLogger


def render_dashboard(
    step: int,
    observations: Dict[int, NodeObservation],
    actions: Dict[int, int],
    rewards: Dict[int, float],
    fairness: float,
    mode_str: str
) -> None:
    """
    Renders terminal dashboard matching exact project specifications.
    """
    print("\n" + "=" * 60)
    print("        MARL DYNAMIC SPECTRUM ALLOCATION CONTROLLER")
    print("=" * 60)
    print(f"Step: {step} | Environment: {mode_str} | Active Nodes: {len(observations)}")
    print(f"Network Jain's Fairness Index: {fairness:.3f}")
    print("-" * 60)
    print(f"{'Node':<6}{'CurrCh':<8}{'Selected':<10}{'RSSI(dBm)':<11}{'TP(Mbps)':<10}{'Delay(ms)':<11}{'Loss(%)':<9}{'Reward':<8}")
    print("-" * 60)

    for node_id in sorted(observations.keys()):
        obs = observations[node_id]
        sel = actions.get(node_id, obs.current_channel)
        r = rewards.get(node_id, 0.0)
        status_flag = "" if obs.online else " [OFFLINE]"
        print(f"{node_id:<6}{obs.current_channel:<8}{sel:<10}{obs.rssi:<11.1f}{obs.throughput_mbps:<10.1f}{obs.delay_ms:<11.1f}{obs.packet_loss_percent:<9.1f}{r:<8.3f}{status_flag}")

    print("-" * 60)
    for node_id in sorted(actions.keys()):
        print(f"Node {node_id} MARL Decision: Channel {actions[node_id]}")
    print("=" * 60)


def run_controller(
    steps: int = 100,
    interval_sec: float = 1.0,
    explore: bool = False
) -> None:
    """
    Main runtime controller loop.
    """
    mode_str = "SIMULATION" if config.SIMULATION_MODE else "REAL ESP32 HARDWARE"
    print("\n" + "=" * 65)
    print("      STARTING DYNAMIC SPECTRUM ALLOCATION CONTROLLER")
    print("=" * 65)
    print(f"Target Environment: {mode_str}")
    print(f"Available Spectrum: Channels {config.AVAILABLE_CHANNELS}")
    print(f"Inference Interval: {interval_sec}s | Max Steps: {steps if steps > 0 else 'Continuous'}")
    print("Press Ctrl+C at any time to pause or exit gracefully.")
    print("=" * 65 + "\n")

    # 1. Initialize Environment
    env = get_environment(simulation_mode=config.SIMULATION_MODE)
    
    # 2. Initialize MARL Policy
    controller = MultiAgentSpectrumController()
    loaded = controller.load_checkpoint()
    if not loaded:
        print("[Controller Warning] Pre-trained weights not found. Using initial Q-policy.")

    logger = ExperimentLogger(filename="controller_runtime_log.csv")
    obs_dict = env.reset()
    step = 0

    try:
        while True:
            step += 1
            if steps > 0 and step > steps:
                break

            # 1. MARL Action Selection
            actions = controller.select_actions(obs_dict, explore=explore)

            # 2. Dispatch to Hardware / Simulation
            next_obs_dict, rewards, done, info = env.step(actions)

            # 3. Optional online fine-tuning
            controller.update_all(obs_dict, actions, rewards, next_obs_dict, done)

            # 4. Compute metrics
            throughputs = [obs.throughput_mbps for obs in next_obs_dict.values()]
            fairness = metrics.calculate_jains_fairness_index(throughputs)

            # 5. Log step
            ts = time.time()
            for nid, obs in next_obs_dict.items():
                logger.log_step(
                    node_id=nid,
                    current_channel=obs.current_channel,
                    selected_channel=actions.get(nid, obs.current_channel),
                    rssi=obs.rssi,
                    throughput_mbps=obs.throughput_mbps,
                    delay_ms=obs.delay_ms,
                    packet_loss_percent=obs.packet_loss_percent,
                    traffic_load=obs.traffic_load_mbps,
                    reward=rewards.get(nid, 0.0),
                    episode=1,
                    step=step,
                    timestamp=ts
                )

            # 6. Render Terminal Dashboard
            render_dashboard(
                step=step,
                observations=next_obs_dict,
                actions=actions,
                rewards=rewards,
                fairness=fairness,
                mode_str=mode_str
            )

            obs_dict = next_obs_dict
            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n[Controller] Execution interrupted by user. Shutting down gracefully...")

    print(f"\n[Controller] Runtime finished. Log saved to: {logger.filepath}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARL Spectrum Controller")
    parser.add_argument("--steps", type=int, default=50, help="Number of control steps (0 for infinite)")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--explore", action="store_true", help="Enable epsilon-greedy exploration")
    args = parser.parse_args()

    run_controller(steps=args.steps, interval_sec=args.interval, explore=args.explore)
