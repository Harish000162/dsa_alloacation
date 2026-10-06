"""
Multi-Agent Reinforcement Learning (MARL) Agent Module
=====================================================
Implements Multi-Agent Independent Q-Learning (IQL) for Dynamic Spectrum Allocation.

ALGORITHM OVERVIEW (For B.Tech / M.Tech Research):
--------------------------------------------------
In Independent Q-Learning (IQL):
1. Each ESP32 node i in {1, 2, ..., N} is treated as an independent learning agent.
2. The environment is modeled as a Decentralized Partially Observable MDP (Dec-POMDP).
3. Each agent maintains its own action-value function Q_i(s_i, a_i), representing the
   expected cumulative discounted reward for taking action a_i in state s_i.
4. Action Space:
     Action 0 -> Channel 1  (2412 MHz)
     Action 1 -> Channel 6  (2437 MHz)
     Action 2 -> Channel 11 (2462 MHz)
5. Bellman Temporal Difference (TD) Update:
     Q_i(s_i, a_i) <- Q_i(s_i, a_i) + alpha * [ r_i + gamma * max_a' Q_i(s'_i, a') - Q_i(s_i, a_i) ]
     where:
       alpha = Learning Rate
       gamma = Discount Factor
       r_i   = Multi-objective scalar reward
6. Exploration Policy:
     Epsilon-greedy action selection with exponential decay schedule.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

import config
import metrics
from environment import NodeObservation


class IndependentQLearningAgent:
    """
    Independent Q-Learning Agent for a single ESP32 node.
    Maintains a discrete Q-table mapping discretized state keys to action values.
    """

    def __init__(
        self,
        node_id: int,
        actions: List[int] = config.AVAILABLE_CHANNELS,
        learning_rate: float = config.LEARNING_RATE,
        discount_factor: float = config.DISCOUNT_FACTOR,
        epsilon: float = config.EPSILON_START,
        epsilon_min: float = config.EPSILON_MIN,
        epsilon_decay: float = config.EPSILON_DECAY
    ):
        self.node_id = node_id
        self.actions = actions
        self.num_actions = len(actions)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-Table: Dict[state_key_str, List[float of length num_actions]]
        self.q_table: Dict[str, List[float]] = {}
        
        # Diagnostic tracking
        self.total_updates = 0
        self.td_errors: List[float] = []

    def _get_q_values(self, state_key: str) -> List[float]:
        """Returns Q-values for a state, initializing to 0.0 if unvisited."""
        if state_key not in self.q_table:
            # Optimistic initialization or zero initialization
            self.q_table[state_key] = [0.0] * self.num_actions
        return self.q_table[state_key]

    def select_action(self, state_key: str, explore: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        Returns:
            Chosen physical Wi-Fi channel (e.g. 1, 6, or 11).
        """
        q_vals = self._get_q_values(state_key)
        
        if explore and random.random() < self.epsilon:
            # Exploration: Choose random channel
            action_idx = random.randrange(self.num_actions)
        else:
            # Exploitation: Select action with highest Q-value (break ties randomly)
            max_val = max(q_vals)
            best_indices = [idx for idx, val in enumerate(q_vals) if abs(val - max_val) < 1e-6]
            action_idx = random.choice(best_indices)
            
        return self.actions[action_idx]

    def update(
        self,
        state_key: str,
        action: int,
        reward: float,
        next_state_key: str,
        done: bool = False
    ) -> float:
        """
        Performs Q-value update via Bellman Temporal Difference equation.
        Returns:
            The computed TD-error.
        """
        if action not in self.actions:
            return 0.0
            
        action_idx = self.actions.index(action)
        q_vals = self._get_q_values(state_key)
        current_q = q_vals[action_idx]
        
        # Target Q computation
        if done:
            target_q = reward
        else:
            next_q_vals = self._get_q_values(next_state_key)
            target_q = reward + self.gamma * max(next_q_vals)
            
        td_error = target_q - current_q
        
        # Q-learning update step
        q_vals[action_idx] = current_q + self.lr * td_error
        self.q_table[state_key] = q_vals
        
        self.total_updates += 1
        if len(self.td_errors) < 1000:
            self.td_errors.append(abs(td_error))
            
        return float(td_error)

    def decay_epsilon(self) -> None:
        """Decays exploration parameter epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class MultiAgentSpectrumController:
    """
    Coordinator managing multiple independent Q-learning agents
    (one per ESP32 node) in the shared spectrum environment.
    """

    def __init__(
        self,
        num_nodes: int = config.NUM_NODES,
        channels: List[int] = config.AVAILABLE_CHANNELS
    ):
        self.num_nodes = num_nodes
        self.channels = channels
        
        # Initialize an independent agent for each node
        self.agents: Dict[int, IndependentQLearningAgent] = {
            node_id: IndependentQLearningAgent(node_id=node_id, actions=channels)
            for node_id in range(1, num_nodes + 1)
        }

    def get_state_key(self, obs: NodeObservation) -> str:
        """Encodes NodeObservation into a discrete state string key."""
        discrete_tuple = metrics.discretize_state(
            current_channel=obs.current_channel,
            rssi_dbm=obs.rssi,
            interference_dbm=obs.interference,
            traffic_load_mbps=obs.traffic_load_mbps
        )
        return metrics.state_to_key(discrete_tuple)

    def select_actions(
        self,
        observations: Dict[int, NodeObservation],
        explore: bool = True
    ) -> Dict[int, int]:
        """
        Coordinates action selection for all agents simultaneously.
        Returns:
            Dict mapping node_id -> selected channel.
        """
        actions: Dict[int, int] = {}
        for node_id, obs in observations.items():
            agent = self.agents.get(node_id)
            if agent:
                state_key = self.get_state_key(obs)
                actions[node_id] = agent.select_action(state_key, explore=explore)
            else:
                actions[node_id] = random.choice(self.channels)
        return actions

    def update_all(
        self,
        observations: Dict[int, NodeObservation],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_observations: Dict[int, NodeObservation],
        done: bool = False
    ) -> Dict[int, float]:
        """
        Updates Q-tables of all agents given their transitions.
        """
        td_errors: Dict[int, float] = {}
        for node_id, agent in self.agents.items():
            if node_id in observations and node_id in next_observations:
                s_key = self.get_state_key(observations[node_id])
                s_next_key = self.get_state_key(next_observations[node_id])
                act = actions.get(node_id, self.channels[0])
                r = rewards.get(node_id, 0.0)
                
                td = agent.update(s_key, act, r, s_next_key, done)
                td_errors[node_id] = td
        return td_errors

    def decay_epsilons(self) -> None:
        """Decays epsilon for all agents."""
        for agent in self.agents.values():
            agent.decay_epsilon()

    def get_current_epsilon(self) -> float:
        """Returns representative epsilon across agents."""
        if not self.agents:
            return 0.0
        return list(self.agents.values())[0].epsilon

    def save_checkpoint(self, filepath: Optional[Path] = None) -> None:
        """Saves all agents' Q-tables and training states to JSON."""
        if filepath is None:
            filepath = config.CHECKPOINT_PATH
            
        data = {
            "num_nodes": self.num_nodes,
            "channels": self.channels,
            "agents": {}
        }
        
        for node_id, agent in self.agents.items():
            data["agents"][str(node_id)] = {
                "epsilon": agent.epsilon,
                "total_updates": agent.total_updates,
                "q_table": agent.q_table
            }
            
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[MARL Controller] Checkpoint saved successfully to: {filepath}")
        except Exception as e:
            print(f"[MARL Controller Error] Failed saving checkpoint: {e}")

    def load_checkpoint(self, filepath: Optional[Path] = None) -> bool:
        """Loads Q-tables and parameters from saved JSON checkpoint."""
        if filepath is None:
            filepath = config.CHECKPOINT_PATH
            
        if not filepath.exists():
            print(f"[MARL Controller Warning] No checkpoint found at {filepath}")
            return False
            
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            agents_data = data.get("agents", {})
            for node_str, a_info in agents_data.items():
                node_id = int(node_str)
                if node_id in self.agents:
                    self.agents[node_id].q_table = a_info.get("q_table", {})
                    self.agents[node_id].epsilon = a_info.get("epsilon", config.EPSILON_MIN)
                    self.agents[node_id].total_updates = a_info.get("total_updates", 0)
                    
            print(f"[MARL Controller] Checkpoint loaded successfully from: {filepath}")
            return True
        except Exception as e:
            print(f"[MARL Controller Error] Failed loading checkpoint: {e}")
            return False
