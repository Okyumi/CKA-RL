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
            obs: Current observations (batch_size, obs_dim) or dict
            next_obs: Next observations (batch_size, obs_dim) or dict
            actions: Actions taken (batch_size, action_dim)
            rewards: Rewards received (batch_size,)
            terminations: Whether episode terminated (batch_size,)
            infos: Info dicts from environment (dict for vectorized env, or list)
        """
        # Handle vectorized environment format
        # Vectorized envs return infos as dict with keys like 'final_info', 'final_observation', etc.
        if isinstance(infos, dict):
            batch_size = len(obs) if isinstance(obs, (list, np.ndarray)) else 1
            infos_list = []
            for i in range(batch_size):
                info_dict = {}
                # Extract per-env info if available
                if 'final_info' in infos and infos['final_info'] is not None:
                    if i < len(infos['final_info']) and infos['final_info'][i] is not None:
                        info_dict.update(infos['final_info'][i])
                # Add other keys that might be per-env
                for key, value in infos.items():
                    if key not in ['final_info', 'final_observation']:
                        if isinstance(value, (list, np.ndarray)) and len(value) > i:
                            info_dict[key] = value[i]
                        elif not isinstance(value, (list, np.ndarray)):
                            info_dict[key] = value
                infos_list.append(info_dict)
            infos = infos_list
        
        batch_size = len(obs) if isinstance(obs, (list, np.ndarray)) else 1
        
        # Handle single vs batch observations
        if not isinstance(obs, (list, np.ndarray)):
            obs = [obs]
            next_obs = [next_obs]
            actions = [actions]
            rewards = [rewards] if not isinstance(rewards, (list, np.ndarray)) else rewards
            terminations = [terminations] if not isinstance(terminations, (list, np.ndarray)) else terminations
        
        for i in range(batch_size):
            transition = Transition(
                observation=obs[i],
                action=actions[i] if isinstance(actions, (list, np.ndarray)) else actions,
                reward=float(rewards[i] if isinstance(rewards, (list, np.ndarray)) else rewards),
                next_observation=next_obs[i],
                terminated=bool(terminations[i] if isinstance(terminations, (list, np.ndarray)) else terminations),
                truncated=False,  # Will be set from infos if needed
                info=infos[i] if i < len(infos) else {},
                episode_id=self.current_episode_id,
            )
            
            # Check if this is a truncation
            if truncations is not None:
                if isinstance(truncations, (list, np.ndarray)):
                    transition.truncated = bool(truncations[i] if i < len(truncations) else False)
                else:
                    transition.truncated = bool(truncations)
            elif i < len(infos):
                # Check for truncation in various formats
                if isinstance(infos[i], dict):
                    if 'TimeLimit.truncated' in infos[i]:
                        transition.truncated = bool(infos[i]['TimeLimit.truncated'])
                    # Also check truncations array if present
                    if 'truncations' in infos[i]:
                        transition.truncated = bool(infos[i]['truncations'])
            
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
        Sample transitions and their corresponding future goals from the same trajectory.
        
        This implements the contrastive RL sampling strategy:
        1. Sample trajectories
        2. For each transition (s_t, a_t) in each trajectory, sample a random future state
           from the same trajectory using geometric distribution
        3. Extract goals from future states
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary with keys (compatible with DictReplayBuffer format):
            - observations: Dict with 'observation', 'desired_goal', 'critic_goal'
            - actions: (batch_size, action_dim)
            - next_observations: Dict with 'observation', 'desired_goal', 'critic_goal'
            - rewards: (batch_size,)
            - terminations: (batch_size,)
        """
        if self.size() == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Sample trajectories until we have enough transitions
        # We sample more trajectories than needed to account for variable episode lengths
        trajectories = []
        total_transitions = 0
        max_trajectories = min(len(self.trajectories), 100)  # Limit to avoid infinite loop
        
        while total_transitions < batch_size and len(trajectories) < max_trajectories:
            sampled = self.sample_trajectories(1)
            if len(sampled) > 0 and len(sampled[0]) > 1:  # Need at least 2 transitions for future sampling
                trajectories.extend(sampled)
                total_transitions += len(sampled[0]) - 1  # -1 because last transition has no future
        
        # Collect transitions and their future goals
        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminations = []
        
        for trajectory in trajectories:
            if len(trajectory) < 2:  # Need at least 2 transitions
                continue
            
            # For each transition in trajectory, sample a future goal
            for t in range(len(trajectory) - 1):  # Last transition has no future
                if len(observations) >= batch_size:
                    break
                    
                transition = trajectory[t]
                
                # Sample random future state from same trajectory using geometric distribution
                future_idx = self._sample_future_state_idx(t, len(trajectory))
                future_transition = trajectory[future_idx]
                
                # Extract goal from future state's observation
                if isinstance(future_transition.observation, dict):
                    # Dict observation: extract critic_goal
                    goal = future_transition.observation.get('critic_goal', 
                                                             future_transition.observation.get('desired_goal'))
                else:
                    # Flat observation: extract goal from indices [goal_start_idx:goal_end_idx]
                    goal = future_transition.observation[self.goal_start_idx:self.goal_end_idx]
                
                # Build observation dict with goal from future state
                if isinstance(transition.observation, dict):
                    obs_dict = {
                        'observation': transition.observation['observation'],
                        'desired_goal': transition.observation['desired_goal'],
                        'critic_goal': goal,  # Use goal from future state
                    }
                    next_obs_dict = {
                        'observation': transition.next_observation['observation'],
                        'desired_goal': transition.next_observation['desired_goal'],
                        'critic_goal': goal,  # Use same goal for next_obs
                    }
                else:
                    # Flat observation - reconstruct with goal
                    state = transition.observation[:self.goal_start_idx]
                    obs_dict = {
                        'observation': state,
                        'desired_goal': goal,
                        'critic_goal': goal,
                    }
                    next_state = transition.next_observation[:self.goal_start_idx]
                    next_obs_dict = {
                        'observation': next_state,
                        'desired_goal': goal,
                        'critic_goal': goal,
                    }
                
                observations.append(obs_dict)
                actions.append(transition.action)
                next_observations.append(next_obs_dict)
                rewards.append(transition.reward)
                terminations.append(transition.terminated)
            
            if len(observations) >= batch_size:
                break
        
        # Truncate to batch_size
        observations = observations[:batch_size]
        actions = actions[:batch_size]
        next_observations = next_observations[:batch_size]
        rewards = rewards[:batch_size]
        terminations = terminations[:batch_size]
        
        # Convert to tensors (compatible with DictReplayBuffer format)
        def dict_to_tensor(dict_list, key):
            """Extract a key from list of dicts and convert to tensor."""
            values = [d[key] for d in dict_list]
            if isinstance(values[0], np.ndarray):
                return torch.from_numpy(np.array(values)).float()
            elif isinstance(values[0], torch.Tensor):
                return torch.stack(values).float()
            else:
                return torch.tensor(values).float()
        
        def to_tensor(arr_list):
            """Convert list of arrays/tensors to single tensor."""
            if len(arr_list) == 0:
                return torch.empty(0)
            if isinstance(arr_list[0], np.ndarray):
                return torch.from_numpy(np.array(arr_list)).float()
            elif isinstance(arr_list[0], torch.Tensor):
                return torch.stack(arr_list).float()
            else:
                return torch.tensor(arr_list).float()
        
        # Return in DictReplayBuffer format (using a simple class for compatibility)
        class Batch:
            def __init__(self):
                self.observations = type('Obs', (), {
                    'observation': dict_to_tensor(observations, 'observation').to(self.device),
                    'desired_goal': dict_to_tensor(observations, 'desired_goal').to(self.device),
                    'critic_goal': dict_to_tensor(observations, 'critic_goal').to(self.device),
                })()
                self.actions = to_tensor(actions).to(self.device)
                self.next_observations = type('NextObs', (), {
                    'observation': dict_to_tensor(next_observations, 'observation').to(self.device),
                    'desired_goal': dict_to_tensor(next_observations, 'desired_goal').to(self.device),
                    'critic_goal': dict_to_tensor(next_observations, 'critic_goal').to(self.device),
                })()
                self.rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                self.terminations = torch.tensor(terminations, dtype=torch.bool).to(self.device)
        
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
