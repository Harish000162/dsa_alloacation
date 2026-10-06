"""
Wireless Environment Module for Dynamic Spectrum Allocation
===========================================================
Provides unified simulated and hardware environments.
Simulates realistic 2.4 GHz RF propagation, CSMA/CA contention,
external jamming/interference, and interfaces directly with ESP32 nodes.
"""

import time
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import requests
import numpy as np

import config
import metrics


@dataclass
class NodeObservation:
    """Telemetry and state observation for a single wireless agent."""
    node_id: int
    current_channel: int
    rssi: float                 # dBm
    interference: float         # dBm
    throughput_mbps: float      # Measured / estimated application throughput
    delay_ms: float             # Round-trip packet latency
    packet_loss_percent: float  # Percentage of dropped packets [0, 100]
    traffic_load_mbps: float    # Offered traffic demand
    channel_scan: Dict[int, float] = field(default_factory=dict)
    hardware_applied: bool = True
    online: bool = True


class BaseSpectrumEnvironment:
    """Base interface for spectrum allocation environments."""
    
    def reset(self) -> Dict[int, NodeObservation]:
        raise NotImplementedError

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[Dict[int, NodeObservation], Dict[int, float], bool, Dict[str, Any]]:
        raise NotImplementedError


class SimulatedSpectrumEnvironment(BaseSpectrumEnvironment):
    """
    High-fidelity 2.4 GHz RF simulation environment.
    Models:
      - Orthogonal 802.11 channels (1, 6, 11)
      - Log-distance path loss with log-normal shadowing
      - CSMA/CA co-channel contention overhead
      - Dynamic external interference (e.g., microwave oven or non-cooperative AP)
    """

    def __init__(self, num_nodes: int = config.NUM_NODES):
        self.num_nodes = num_nodes
        self.channels = config.AVAILABLE_CHANNELS
        self.current_step = 0
        
        # Spatial positioning of 5 nodes in a 20m x 20m indoor grid (distance to AP/receiver in meters)
        # Allows realistic path loss differences per node
        self.node_distances = {
            1: 3.5,
            2: 7.2,
            3: 5.0,
            4: 9.8,
            5: 4.2
        }
        
        # Current active channel of each node
        self.node_channels: Dict[int, int] = {i: random.choice(self.channels) for i in range(1, self.num_nodes + 1)}
        
        # Offered traffic load per node (in Mbps)
        self.traffic_demands: Dict[int, float] = {
            1: 15.0,
            2: 12.0,
            3: 18.0,
            4: 10.0,
            5: 14.0
        }

    def reset(self) -> Dict[int, NodeObservation]:
        """Resets the environment for a new episode."""
        self.current_step = 0
        self.node_channels = {i: random.choice(self.channels) for i in range(1, self.num_nodes + 1)}
        return self._compute_observations()

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[Dict[int, NodeObservation], Dict[int, float], bool, Dict[str, Any]]:
        """
        Executes channel decisions for all nodes, calculates RF interactions,
        and computes per-agent rewards.
        """
        self.current_step += 1
        
        # Apply actions (validate channels)
        for node_id, chosen_channel in actions.items():
            if chosen_channel in self.channels:
                self.node_channels[node_id] = chosen_channel
            else:
                # Fallback to closest valid channel
                self.node_channels[node_id] = self.channels[0]
                
        # Generate telemetry based on channel collision & interference
        observations = self._compute_observations()
        
        # Compute individual throughputs for fairness calculation
        throughputs = [obs.throughput_mbps for obs in observations.values()]
        fairness_index = metrics.calculate_jains_fairness_index(throughputs)
        
        # Compute multi-objective reward per node
        rewards: Dict[int, float] = {}
        for node_id, obs in observations.items():
            r = metrics.calculate_reward(
                throughput_mbps=obs.throughput_mbps,
                delay_ms=obs.delay_ms,
                packet_loss_percent=obs.packet_loss_percent,
                interference_dbm=obs.interference,
                fairness_bonus=fairness_index
            )
            rewards[node_id] = r
            
        done = self.current_step >= config.STEPS_PER_EPISODE
        info = {
            "step": self.current_step,
            "fairness_index": fairness_index,
            "external_interference_active": self._is_external_interference_active()
        }
        
        return observations, rewards, done, info

    def _is_external_interference_active(self) -> bool:
        """Checks if dynamic external interference is currently firing."""
        if not config.SIM_EXTERNAL_INTERFERENCE_ENABLED:
            return False
        return (
            config.SIM_EXTERNAL_INTERFERENCE_START_STEP
            <= self.current_step
            <= config.SIM_EXTERNAL_INTERFERENCE_END_STEP
        )

    def _compute_observations(self) -> Dict[int, NodeObservation]:
        """
        Computes realistic RF values: RSSI, co-channel contention,
        throughput, latency, and packet loss for all nodes.
        """
        # Count how many nodes share each channel
        channel_occupancy: Dict[int, List[int]] = {ch: [] for ch in self.channels}
        for node_id, ch in self.node_channels.items():
            channel_occupancy[ch].append(node_id)
            
        ext_interf_active = self._is_external_interference_active()
        ext_channel = config.SIM_EXTERNAL_INTERFERENCE_CHANNEL
        
        observations: Dict[int, NodeObservation] = {}
        
        for node_id in range(1, self.num_nodes + 1):
            ch = self.node_channels[node_id]
            dist = self.node_distances.get(node_id, 5.0)
            demand = self.traffic_demands.get(node_id, 12.0)
            
            # 1. Path Loss & Shadowing (Log-distance model for 2.4 GHz indoor)
            # PL(d) = PL(d0) + 10 * n * log10(d/d0) + X_sigma
            # At 1m, 2.4 GHz free-space loss ~ 40 dB. Indoor path-loss exponent n ~ 3.0
            shadowing = random.gauss(0, 2.0)  # Log-normal shadowing (std = 2 dB)
            path_loss = 40.0 + 30.0 * math.log10(max(1.0, dist)) + shadowing
            rssi = config.SIM_NODE_TX_POWER_DBM - path_loss
            rssi = float(np.clip(rssi, -95.0, -35.0))
            
            # 2. Co-channel Contention
            sharing_nodes = channel_occupancy[ch]
            num_sharers = len(sharing_nodes)
            
            # 3. Base Interference Calculation
            # Background noise floor
            interf_mw = 10.0 ** (config.SIM_NOISE_FLOOR_DBM / 10.0)
            
            # Add co-channel interference power from other sharing nodes
            if num_sharers > 1:
                # Each other transmitter adds interference power
                co_channel_power_mw = (num_sharers - 1) * (10.0 ** (-72.0 / 10.0))
                interf_mw += co_channel_power_mw
                
            # Add external interference burst if active on this channel
            if ext_interf_active and ch == ext_channel:
                ext_power_mw = 10.0 ** ((-60.0 + config.SIM_EXTERNAL_INTERFERENCE_POWER_DB) / 10.0)
                interf_mw += ext_power_mw
                
            interference_dbm = 10.0 * math.log10(interf_mw)
            
            # 4. Realistic 802.11 Throughput under Contention
            # Single node achieves high throughput up to its demand
            # Sharing nodes divide bandwidth + suffer CSMA/CA collision overhead
            if num_sharers == 1:
                contention_efficiency = 0.95
                collision_loss_rate = random.uniform(0.1, 1.2)  # Low base packet loss
                delay_base = random.uniform(4.0, 7.0)          # Low base latency
            else:
                # Contention backoff penalty scales with sharing nodes
                contention_efficiency = 1.0 / (num_sharers ** (1.0 + config.SIM_CONTENTION_PENALTY_FACTOR))
                collision_loss_rate = min(45.0, 4.0 * (num_sharers ** 1.8) + random.uniform(0, 3.0))
                delay_base = 5.0 + 8.5 * (num_sharers - 1) + random.uniform(1.0, 4.0)
                
            # Impact of external jammer on Channel 6
            if ext_interf_active and ch == ext_channel:
                contention_efficiency *= 0.3
                collision_loss_rate += 35.0
                delay_base += 40.0
                
            achieved_throughput = min(demand, 28.0) * contention_efficiency
            achieved_throughput = max(0.5, achieved_throughput + random.gauss(0, 0.4))
            
            packet_loss = float(np.clip(collision_loss_rate, 0.0, 100.0))
            delay = float(np.clip(delay_base, 2.0, config.MAX_EXPECTED_DELAY_MS))
            
            # Simulated channel scan of all 3 channels
            scan_dict = {}
            for c in self.channels:
                c_sharers = len(channel_occupancy[c])
                scan_noise = -90.0 + (c_sharers * 6.0)
                if ext_interf_active and c == ext_channel:
                    scan_noise += config.SIM_EXTERNAL_INTERFERENCE_POWER_DB
                scan_dict[c] = float(np.clip(scan_noise + random.gauss(0, 1.5), -95.0, -40.0))
                
            observations[node_id] = NodeObservation(
                node_id=node_id,
                current_channel=ch,
                rssi=round(rssi, 2),
                interference=round(interference_dbm, 2),
                throughput_mbps=round(achieved_throughput, 2),
                delay_ms=round(delay, 2),
                packet_loss_percent=round(packet_loss, 2),
                traffic_load_mbps=demand,
                channel_scan=scan_dict,
                hardware_applied=True,
                online=True
            )
            
        return observations


class HardwareSpectrumEnvironment(BaseSpectrumEnvironment):
    """
    Physical hardware interface environment connecting directly to ESP32 nodes
    over Wi-Fi HTTP REST endpoints or Serial USB ports.
    """

    def __init__(self, node_configs: Dict[int, Dict[str, Any]] = config.ESP32_NODES):
        self.node_configs = node_configs
        self.num_nodes = len(node_configs)
        self.channels = config.AVAILABLE_CHANNELS
        self.current_step = 0
        self.last_known_observations: Dict[int, NodeObservation] = {}
        
        # Initialize fallback observations
        for node_id in self.node_configs.keys():
            self.last_known_observations[node_id] = NodeObservation(
                node_id=node_id,
                current_channel=self.channels[0],
                rssi=-70.0,
                interference=-85.0,
                throughput_mbps=10.0,
                delay_ms=12.0,
                packet_loss_percent=2.0,
                traffic_load_mbps=10.0,
                channel_scan={1: -85.0, 6: -85.0, 11: -85.0},
                hardware_applied=False,
                online=False
            )

    def reset(self) -> Dict[int, NodeObservation]:
        """Queries all ESP32 nodes for baseline measurements."""
        self.current_step = 0
        return self._poll_all_hardware_nodes()

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[Dict[int, NodeObservation], Dict[int, float], bool, Dict[str, Any]]:
        """
        Sends channel assignment commands to all ESP32s, retrieves real-world
        telemetry, and computes multi-agent rewards.
        """
        self.current_step += 1
        
        # 1. Dispatch channel assignments to ESP32s
        for node_id, target_channel in actions.items():
            self._send_channel_command(node_id, target_channel)
            
        # Small settling time for RF switching & packet transmission
        time.sleep(0.05)
        
        # 2. Collect measurements
        observations = self._poll_all_hardware_nodes()
        
        # 3. Compute fairness and rewards
        throughputs = [obs.throughput_mbps for obs in observations.values()]
        fairness_index = metrics.calculate_jains_fairness_index(throughputs)
        
        rewards: Dict[int, float] = {}
        for node_id, obs in observations.items():
            r = metrics.calculate_reward(
                throughput_mbps=obs.throughput_mbps,
                delay_ms=obs.delay_ms,
                packet_loss_percent=obs.packet_loss_percent,
                interference_dbm=obs.interference,
                fairness_bonus=fairness_index
            )
            rewards[node_id] = r
            
        done = self.current_step >= config.STEPS_PER_EPISODE
        info = {
            "step": self.current_step,
            "fairness_index": fairness_index,
            "online_nodes": sum(1 for obs in observations.values() if obs.online)
        }
        
        return observations, rewards, done, info

    def _send_channel_command(self, node_id: int, target_channel: int) -> bool:
        """Sends channel action to an individual ESP32 node via HTTP POST."""
        node_cfg = self.node_configs.get(node_id)
        if not node_cfg:
            return False
            
        ip = node_cfg.get("ip")
        port = node_cfg.get("port", 80)
        url = f"http://{ip}:{port}/channel"
        
        try:
            payload = {"channel": int(target_channel)}
            resp = requests.post(url, json=payload, timeout=config.HTTP_TIMEOUT_SECONDS)
            return resp.status_code == 200
        except Exception:
            # Handle hardware timeout gracefully without breaking execution
            return False

    def _poll_all_hardware_nodes(self) -> Dict[int, NodeObservation]:
        """Polls every ESP32 node for real-time telemetry."""
        observations: Dict[int, NodeObservation] = {}
        
        for node_id, node_cfg in self.node_configs.items():
            ip = node_cfg.get("ip")
            port = node_cfg.get("port", 80)
            url = f"http://{ip}:{port}/metrics"
            
            try:
                start_t = time.time()
                resp = requests.get(url, timeout=config.HTTP_TIMEOUT_SECONDS)
                rtt_delay_ms = (time.time() - start_t) * 1000.0
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    obs = NodeObservation(
                        node_id=node_id,
                        current_channel=int(data.get("current_channel", self.channels[0])),
                        rssi=float(data.get("rssi", -70.0)),
                        interference=float(data.get("interference", -85.0)),
                        throughput_mbps=float(data.get("throughput_mbps", 12.0)),
                        delay_ms=float(data.get("delay_ms", rtt_delay_ms)),
                        packet_loss_percent=float(data.get("packet_loss_percent", 0.0)),
                        traffic_load_mbps=float(data.get("traffic_load_mbps", 10.0)),
                        channel_scan=data.get("channel_scan", {}),
                        hardware_applied=bool(data.get("hardware_applied", True)),
                        online=True
                    )
                    self.last_known_observations[node_id] = obs
                    observations[node_id] = obs
                else:
                    observations[node_id] = self._get_fallback_observation(node_id)
            except Exception:
                observations[node_id] = self._get_fallback_observation(node_id)
                
        return observations

    def _get_fallback_observation(self, node_id: int) -> NodeObservation:
        """Returns safe fallback observation when an ESP32 is offline or unreachable."""
        fallback = self.last_known_observations[node_id]
        fallback.online = False
        return fallback


def get_environment(simulation_mode: Optional[bool] = None) -> BaseSpectrumEnvironment:
    """Factory helper to obtain the appropriate environment based on mode."""
    if simulation_mode is None:
        simulation_mode = config.SIMULATION_MODE
        
    if simulation_mode:
        return SimulatedSpectrumEnvironment()
    else:
        return HardwareSpectrumEnvironment()
