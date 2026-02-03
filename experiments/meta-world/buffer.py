"""
Trajectory-aware replay buffer for contrastive RL.

This buffer stores full trajectories and supports sampling random future states
from within the same trajectory, following the geometric distribution approach
used in contrastive RL papers.
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class Transition:
    """Single transition in a trajectory."""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    episode_id: int  # Track which episode this transition belongs to


class TrajectoryBuffer:
    """
    Replay buffer that stores full trajectories and supports sampling
    random future states from within the same trajectory.
    
    Key features:
    1. Stores transitions grouped by trajectory/episode
    2. Can sample full trajectories
    3. Can sample random future states from a trajectory using geometric distribution
    4. Maintains episode boundaries for proper future state sampling
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: str = "cpu",
        episode_length: int = 500,  # Typical episode length for MetaWorld
        gamma: float = 0.99,  # Discount factor for geometric distribution
        goal_start_idx: int = 4,  # Start index for goal extraction (indices 4,5,6)
        goal_end_idx: int = 7,  # End index (exclusive)
    ):
        self._buffer_size = buffer_size
        self.device = device
        self.episode_length = episode_length
        self.gamma = gamma
        self.goal_start_idx = goal_start_idx
        self.goal_end_idx = goal_end_idx
        
        # Storage: list of trajectories, where each trajectory is a list of transitions
        self.trajectories: List[List[Transition]] = []
        self.episode_ids: Dict[int, int] = {}  # transition_idx -> episode_id
        self.current_episode_id = 0
        self.current_trajectory: List[Transition] = []
        
        # Track buffer position for circular buffer behavior
        self.insert_position = 0
        self.sample_position = 0
        
        # Store observation and action shapes/dtypes
        if hasattr(observation_space, 'shape'):
            self.obs_shape = observation_space.shape
            self.obs_dtype = observation_space.dtype
        else:
            # Dict space - extract from a sample
            self.obs_shape = None
            self.obs_dtype = None
        
        if hasattr(action_space, 'shape'):
            self.action_shape = action_space.shape
            self.action_dtype = action_space.dtype
        else:
            self.action_shape = None
            self.action_dtype = None
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        infos,
        truncations=None,
    ):
        """
        Add transitions to the buffer.
        
        Args:
            obs: Dict observations from GoalObsWrapper (batch_size, obs_dim)
            next_obs: Dict next observations from GoalObsWrapper
            actions: Actions taken (batch_size, action_dim)
            rewards: Rewards received (batch_size,)
            terminations: Whether episode terminated (batch_size,)
            infos: Info dict from vectorized env (gymnasium AsyncVectorEnv)
        """
        batch_size = rewards.shape[0]
        infos_list = []
        for i in range(batch_size):
            info_dict = {}
            if "final_info" in infos and infos["final_info"] is not None:
                if i < len(infos["final_info"]) and infos["final_info"][i] is not None:
                    info_dict.update(infos["final_info"][i])
            for key, value in infos.items():
                if key in ["final_info", "final_observation"]:
                    continue
                info_dict[key] = value[i]
            infos_list.append(info_dict)

        for i in range(batch_size):
            obs_i = {key: np.squeeze(val[i]) for key, val in obs.items()}
            next_obs_i = {key: np.squeeze(val[i]) for key, val in next_obs.items()}

            transition = Transition(
                observation=obs_i,
                action=actions[i],
                reward=float(rewards[i]),
                next_observation=next_obs_i,
                terminated=bool(terminations[i]),
                truncated=bool(truncations[i]) if truncations is not None else False,
                info=infos_list[i],
                episode_id=self.current_episode_id,
            )
            
            self.current_trajectory.append(transition)
            
            # Check if episode ended
            episode_ended = transition.terminated or transition.truncated
            
            if episode_ended:
                # Save current trajectory
                if len(self.current_trajectory) > 0:
                    self._add_trajectory(self.current_trajectory)
                    self.current_trajectory = []
                self.current_episode_id += 1
    
    def _add_trajectory(self, trajectory: List[Transition]):
        """Add a complete trajectory to the buffer."""
        if len(trajectory) == 0:
            return
        
        # If buffer is full, remove oldest trajectory
        if len(self.trajectories) >= self._buffer_size:
            # Remove oldest trajectory
            removed_trajectory = self.trajectories.pop(0)
            # Update sample_position if needed
            if self.sample_position > 0:
                self.sample_position = max(0, self.sample_position - len(removed_trajectory))
        
        self.trajectories.append(trajectory)
        self.insert_position = len(self.trajectories)
    
    @property
    def buffer_size(self) -> int:
        """Return maximum buffer size (number of trajectories)."""
        return self._buffer_size
    
    def size(self) -> int:
        """Return total number of transitions in buffer."""
        return sum(len(traj) for traj in self.trajectories)
    
    def sample_trajectories(self, num_trajectories: int) -> List[List[Transition]]:
        """
        Sample full trajectories from the buffer.
        
        Args:
            num_trajectories: Number of trajectories to sample
            
        Returns:
            List of trajectories, each is a list of transitions
        """
        if len(self.trajectories) == 0:
            return []
        
        # Sample random trajectories
        available_trajectories = len(self.trajectories)
        num_trajectories = min(num_trajectories, available_trajectories)
        
        indices = np.random.choice(available_trajectories, size=num_trajectories, replace=True)
        return [self.trajectories[idx] for idx in indices]
    
    def sample_with_future_goals(
        self,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample transitions by flattening trajectories with future goals.
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            Dictionary with keys (compatible with DictReplayBuffer format):
            - observations: Dict with 'observation', 'desired_goal', 'critic_goal'
            - actions: (batch_size, action_dim)
            - next_observations: Dict with 'observation', 'desired_goal', 'critic_goal'
            - rewards: (batch_size,)
            - terminations: (batch_size,)
            - trajectory_ids: (batch_size,) - IDs for tracking source trajectory
        """
        if self.size() == 0:
            raise ValueError("Cannot sample from empty buffer")
        if len(self.trajectories) < 2:
            raise ValueError("Need at least 2 trajectories for contrastive sampling")
        samples = []
        traj_indices = np.random.permutation(len(self.trajectories))
        for traj_idx in traj_indices:
            sample = self._sample_from_trajectory(self.trajectories[traj_idx])
            if sample is not None:
                samples.append(sample)
            if len(samples) >= batch_size:
                break

        while len(samples) < batch_size:
            traj_idx = np.random.randint(0, len(self.trajectories))
            sample = self._sample_from_trajectory(self.trajectories[traj_idx])
            if sample is not None:
                samples.append(sample)

        return self._build_batch(samples)

    def _flatten_trajectory(self, trajectory: List[Transition]) -> List[Tuple[dict, dict, np.ndarray, float, bool, int]]:
        if len(trajectory) < 2:
            return []
        flat = []
        for t in range(len(trajectory) - 1):
            transition = trajectory[t]
            future_idx = self._sample_future_state_idx(t, len(trajectory))
            future_transition = trajectory[future_idx]
            goal = future_transition.observation["critic_goal"]
            obs_dict = {
                "observation": transition.observation["observation"],
                "desired_goal": goal,
                "critic_goal": goal,
            }
            next_obs_dict = {
                "observation": transition.next_observation["observation"],
                "desired_goal": goal,
                "critic_goal": goal,
            }
            flat.append(
                (
                    obs_dict,
                    next_obs_dict,
                    transition.action,
                    transition.reward,
                    transition.terminated,
                    transition.episode_id,
                )
            )
        return flat

    def _sample_from_trajectory(self, trajectory: List[Transition]):
        if len(trajectory) < 2:
            return None
        t = np.random.randint(0, len(trajectory) - 1)
        transition = trajectory[t]
        future_idx = self._sample_future_state_idx(t, len(trajectory))
        future_transition = trajectory[future_idx]
        goal = future_transition.observation["critic_goal"]
        obs_dict = {
            "observation": transition.observation["observation"],
            "desired_goal": goal,
            "critic_goal": goal,
        }
        next_obs_dict = {
            "observation": transition.next_observation["observation"],
            "desired_goal": goal,
            "critic_goal": goal,
        }
        return (
            obs_dict,
            next_obs_dict,
            transition.action,
            transition.reward,
            transition.terminated,
            transition.episode_id,
        )

    def _build_batch(self, samples):
        observations = {
            "observation": np.stack([s[0]["observation"] for s in samples], axis=0),
            "desired_goal": np.stack([s[0]["desired_goal"] for s in samples], axis=0),
            "critic_goal": np.stack([s[0]["critic_goal"] for s in samples], axis=0),
        }
        next_observations = {
            "observation": np.stack([s[1]["observation"] for s in samples], axis=0),
            "desired_goal": np.stack([s[1]["desired_goal"] for s in samples], axis=0),
            "critic_goal": np.stack([s[1]["critic_goal"] for s in samples], axis=0),
        }
        actions = np.stack([s[2] for s in samples], axis=0)
        rewards = np.array([s[3] for s in samples], dtype=np.float32)
        terminations = np.array([s[4] for s in samples], dtype=np.bool_)
        trajectory_ids = np.array([s[5] for s in samples], dtype=np.int64)

        device = self.device

        class Batch:
            def __init__(self):
                self.observations = type("Obs", (), {
                    "observation": torch.as_tensor(observations["observation"], device=device, dtype=torch.float32),
                    "desired_goal": torch.as_tensor(observations["desired_goal"], device=device, dtype=torch.float32),
                    "critic_goal": torch.as_tensor(observations["critic_goal"], device=device, dtype=torch.float32),
                })()
                self.actions = torch.as_tensor(actions, device=device, dtype=torch.float32)
                self.next_observations = type("NextObs", (), {
                    "observation": torch.as_tensor(next_observations["observation"], device=device, dtype=torch.float32),
                    "desired_goal": torch.as_tensor(next_observations["desired_goal"], device=device, dtype=torch.float32),
                    "critic_goal": torch.as_tensor(next_observations["critic_goal"], device=device, dtype=torch.float32),
                })()
                self.rewards = torch.as_tensor(rewards, device=device, dtype=torch.float32)
                self.terminations = torch.as_tensor(terminations, device=device, dtype=torch.bool)
                self.trajectory_ids = torch.as_tensor(trajectory_ids, device=device, dtype=torch.long)

        return Batch()
    
    def _sample_future_state_idx(self, current_idx: int, trajectory_length: int) -> int:
        """
        Sample a random future state index from the same trajectory.
        
        Uses geometric distribution: P(k) = (1 - gamma) * gamma^(k-1)
        where k is the number of steps into the future.
        
        Args:
            current_idx: Current time step index in trajectory
            trajectory_length: Total length of trajectory
            
        Returns:
            Index of future state in trajectory
        """
        # Available future indices
        future_indices = list(range(current_idx + 1, trajectory_length))
        
        if len(future_indices) == 0:
            # No future states, return last state
            return trajectory_length - 1
        
        # Create probability distribution using geometric distribution
        # P(step k ahead) = (1 - gamma) * gamma^(k-1)
        num_future = len(future_indices)
        steps_ahead = np.arange(1, num_future + 1)
        
        # Geometric probabilities
        probs = (1 - self.gamma) * (self.gamma ** (steps_ahead - 1))
        probs = probs / probs.sum()  # Normalize
        
        # Sample
        sampled_step_idx = np.random.choice(num_future, p=probs)
        return future_indices[sampled_step_idx]
    
    def sample(self, batch_size: int) -> Any:
        """
        Standard sampling interface (for compatibility with DictReplayBuffer).
        Returns samples with future goals from same trajectory.
        """
        return self.sample_with_future_goals(batch_size)
