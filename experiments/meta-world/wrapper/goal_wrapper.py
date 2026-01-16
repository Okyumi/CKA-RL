import gymnasium as gym
import numpy as np


class GoalObsWrapper(gym.Wrapper):
    """Minimal goal wrapper for MetaWorld goal-observable tasks.

    If the underlying env already returns a dict, this is a passthrough.
    Otherwise, it splits the flat observation into (state, goal) by assuming
    the last `goal_dim` entries represent the goal.
    """

    def __init__(self, env: gym.Env, goal_dim: int = 3):
        super().__init__(env)
        self.goal_dim = goal_dim
        self.critic_goal_indices = [0, 1, 2]
        self._set_observation_space()

    def _set_observation_space(self):
        obs_space = self.env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            state_space = obs_space["observation"]
            critic_low = state_space.low[self.critic_goal_indices]
            critic_high = state_space.high[self.critic_goal_indices]
            obs_space.spaces["critic_goal"] = gym.spaces.Box(
                low=critic_low, high=critic_high, dtype=state_space.dtype
            )
            self.observation_space = obs_space
            return
        obs_dim = int(np.prod(obs_space.shape))
        if self.goal_dim <= 0 or self.goal_dim >= obs_dim:
            raise ValueError(
                f"Invalid goal_dim={self.goal_dim} for obs_dim={obs_dim}"
            )
        state_dim = obs_dim - self.goal_dim
        low = obs_space.low.flatten()
        high = obs_space.high.flatten()
        state_low = low[:state_dim]
        state_high = high[:state_dim]
        goal_low = low[state_dim:]
        goal_high = high[state_dim:]
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    low=state_low, high=state_high, dtype=obs_space.dtype
                ),
                "desired_goal": gym.spaces.Box(
                    low=goal_low, high=goal_high, dtype=obs_space.dtype
                ),
                "achieved_goal": gym.spaces.Box(
                    low=goal_low, high=goal_high, dtype=obs_space.dtype
                ),
                "critic_goal": gym.spaces.Box(
                    low=low[self.critic_goal_indices],
                    high=high[self.critic_goal_indices],
                    dtype=obs_space.dtype,
                ),
            }
        )

    def _to_dict_obs(self, obs):
        if isinstance(obs, dict):
            obs["critic_goal"] = obs["observation"][self.critic_goal_indices]
            return obs
        obs = np.asarray(obs).flatten()
        state = obs[:-self.goal_dim]
        goal = obs[-self.goal_dim:]
        critic_goal = obs[self.critic_goal_indices]
        achieved = (
            state[-self.goal_dim :]
            if state.shape[0] >= self.goal_dim
            else np.zeros_like(goal)
        )
        return {
            "observation": state.astype(np.float32, copy=False),
            "desired_goal": goal.astype(np.float32, copy=False),
            "achieved_goal": achieved.astype(np.float32, copy=False),
            "critic_goal": critic_goal.astype(np.float32, copy=False),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._to_dict_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_dict_obs(obs), reward, terminated, truncated, info
