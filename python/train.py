"""
Training Pipeline for Multi-Agent Reinforcement Learning
========================================================
Executes MARL training across multiple episodes, logs training telemetry,
displays interactive terminal dashboards, and saves trained Q-table models.
"""

import sys
import time
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

import config
from environment import get_environment
from marl_agent import MultiAgentSpectrumController
from data_logger import ExperimentLogger


def print_training_dashboard(
    episode: int,
    total_episodes: int,
    step: int,
    observations: dict,
    actions: dict,
    rewards: dict,
    epsilon: float,
    fairness: float
) -> None:
    """Renders formatted terminal dashboard matching project specifications."""
    print("\n" + "=" * 60)
    print("       MARL DYNAMIC SPECTRUM ALLOCATION - TRAINING")
    print("=" * 60)
    print(f"Episode: {episode + 1}/{total_episodes} | Step: {step} | Mode: {'SIMULATION' if config.SIMULATION_MODE else 'HARDWARE'}")
    print(f"Exploration Rate (Epsilon): {epsilon:.4f} | Jain's Fairness: {fairness:.3f}")
    print("-" * 60)
    print(f"{'Node':<6}{'CurrCh':<8}{'ActionCh':<10}{'RSSI(dBm)':<11}{'TP(Mbps)':<10}{'Delay(ms)':<11}{'Loss(%)':<9}{'Reward':<8}")
    print("-" * 60)
    
    total_tp = 0.0
    for node_id in sorted(observations.keys()):
        obs = observations[node_id]
        act = actions.get(node_id, obs.current_channel)
        r = rewards.get(node_id, 0.0)
        total_tp += obs.throughput_mbps
        print(f"{node_id:<6}{obs.current_channel:<8}{act:<10}{obs.rssi:<11.1f}{obs.throughput_mbps:<10.1f}{obs.delay_ms:<11.1f}{obs.packet_loss_percent:<9.1f}{r:<8.3f}")
        
    print("-" * 60)
    channel_summary = ", ".join([f"Node {nid}->Ch {actions.get(nid, '?')}" for nid in sorted(actions.keys())])
    print(f"MARL Decisions: {channel_summary}")
    print(f"Aggregated Network Throughput: {total_tp:.2f} Mbps")
    print("=" * 60)


def plot_training_curves(
    episode_rewards: List[float],
    episode_throughputs: List[float],
    episode_epsilons: List[float],
    output_path=config.PLOTS_DIR / "training_curves.png"
) -> None:
    """Generates publication-quality training convergence plots."""
    try:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        episodes_range = range(1, len(episode_rewards) + 1)
        
        # 1. Episode Cumulative Reward
        axes[0].plot(episodes_range, episode_rewards, color="#1f77b4", alpha=0.3, label="Raw Episode Reward")
        # Moving average
        window = max(5, len(episode_rewards) // 20)
        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
            axes[0].plot(range(window, len(episode_rewards) + 1), moving_avg, color="#0d47a1", linewidth=2.0, label=f"Smoothed ({window} eps)")
        axes[0].set_ylabel("Total Episode Reward", fontsize=11)
        axes[0].set_title("MARL Spectrum Allocation: Training Convergence", fontsize=13, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="lower right")

        # 2. Average Network Throughput
        axes[1].plot(episodes_range, episode_throughputs, color="#2ca02c", alpha=0.3, label="Raw Mean Throughput")
        if len(episode_throughputs) >= window:
            moving_tp = np.convolve(episode_throughputs, np.ones(window)/window, mode="valid")
            axes[1].plot(range(window, len(episode_throughputs) + 1), moving_tp, color="#1b5e20", linewidth=2.0, label="Smoothed Mean")
        axes[1].set_ylabel("Mean Throughput (Mbps)", fontsize=11)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend(loc="lower right")

        # 3. Epsilon Decay Schedule
        axes[2].plot(episodes_range, episode_epsilons, color="#d62728", linewidth=2.0, label="Epsilon (Exploration Rate)")
        axes[2].set_xlabel("Training Episode", fontsize=11)
        axes[2].set_ylabel("Epsilon", fontsize=11)
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[Training] Saved training convergence curves to: {output_path}")
    except Exception as e:
        print(f"[Training Warning] Could not generate training plot: {e}")


def train(
    episodes: int = config.EPISODES,
    steps_per_episode: int = config.STEPS_PER_EPISODE
) -> None:
    """
    Main training routine for Multi-Agent Reinforcement Learning.
    """
    print("\n" + "=" * 60)
    print("      INITIALIZING MARL SPECTRUM ALLOCATION TRAINING")
    print("=" * 60)
    print(f"Agents (Nodes): {config.NUM_NODES}")
    print(f"Available 2.4 GHz Channels: {config.AVAILABLE_CHANNELS}")
    print(f"Episodes: {episodes} | Steps per Episode: {steps_per_episode}")
    print(f"Learning Rate (alpha): {config.LEARNING_RATE} | Gamma: {config.DISCOUNT_FACTOR}")
    print(f"Mode: {'SIMULATION' if config.SIMULATION_MODE else 'REAL ESP32 HARDWARE'}")
    print("=" * 60 + "\n")

    # Initialize components
    env = get_environment(simulation_mode=config.SIMULATION_MODE)
    controller = MultiAgentSpectrumController(
        num_nodes=config.NUM_NODES,
        channels=config.AVAILABLE_CHANNELS
    )
    logger = ExperimentLogger(filename="training_log.csv")

    history_rewards: List[float] = []
    history_throughputs: List[float] = []
    history_epsilons: List[float] = []

    start_time = time.time()

    for ep in range(episodes):
        obs_dict = env.reset()
        episode_reward = 0.0
        episode_tp_list: List[float] = []

        for step in range(1, steps_per_episode + 1):
            # 1. Action selection via epsilon-greedy
            actions = controller.select_actions(obs_dict, explore=True)

            # 2. Step environment
            next_obs_dict, rewards, done, info = env.step(actions)

            # 3. Update MARL agents via TD-error
            controller.update_all(obs_dict, actions, rewards, next_obs_dict, done)

            # 4. Log telemetry for all nodes
            step_time = time.time()
            for node_id, obs in next_obs_dict.items():
                logger.log_step(
                    node_id=node_id,
                    current_channel=obs.current_channel,
                    selected_channel=actions.get(node_id, obs.current_channel),
                    rssi=obs.rssi,
                    throughput_mbps=obs.throughput_mbps,
                    delay_ms=obs.delay_ms,
                    packet_loss_percent=obs.packet_loss_percent,
                    traffic_load=obs.traffic_load_mbps,
                    reward=rewards.get(node_id, 0.0),
                    episode=ep + 1,
                    step=step,
                    timestamp=step_time
                )
                episode_tp_list.append(obs.throughput_mbps)

            episode_reward += sum(rewards.values())
            obs_dict = next_obs_dict

            # Periodic terminal dashboard display
            if (ep + 1) % 50 == 0 and step == steps_per_episode:
                print_training_dashboard(
                    episode=ep,
                    total_episodes=episodes,
                    step=step,
                    observations=obs_dict,
                    actions=actions,
                    rewards=rewards,
                    epsilon=controller.get_current_epsilon(),
                    fairness=info.get("fairness_index", 1.0)
                )

        # Decay exploration rate
        controller.decay_epsilons()

        # Track history
        history_rewards.append(episode_reward)
        mean_ep_tp = float(np.mean(episode_tp_list)) if episode_tp_list else 0.0
        history_throughputs.append(mean_ep_tp)
        history_epsilons.append(controller.get_current_epsilon())

        # Progress indicator
        if (ep + 1) % 25 == 0 or ep == 0:
            elapsed = time.time() - start_time
            print(f"[Episode {ep+1:3d}/{episodes}] Avg Reward: {episode_reward:7.2f} | Mean TP: {mean_ep_tp:5.2f} Mbps | Eps: {controller.get_current_epsilon():.3f} | Elapsed: {elapsed:.1f}s")

        # Periodic checkpoint
        if (ep + 1) % config.SAVE_CHECKPOINT_EVERY_EPISODES == 0:
            controller.save_checkpoint()

    # Final save & plot
    controller.save_checkpoint()
    plot_training_curves(history_rewards, history_throughputs, history_epsilons)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("                 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Final Model Checkpoint: {config.CHECKPOINT_PATH}")
    print(f"Training Log CSV: {logger.filepath}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train()
