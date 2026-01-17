import gymnasium as gym
import numpy as np


class GoalObsWrapper(gym.Wrapper):
    """Minimal goal wrapper for MetaWorld goal-observable tasks.

    Splits the flat observation into (state, goal) by assuming
    the last `goal_dim` entries represent the goal.
    """

    def __init__(self, env: gym.Env, goal_dim: int = 3):
        super().__init__(env)
        self.goal_dim = goal_dim
        self.critic_goal_indices = [0, 1, 2]
        self._set_observation_space()

    def _set_observation_space(self):
        obs_space = self.env.observation_space
        obs_dim = int(np.prod(obs_space.shape))

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
                "critic_goal": gym.spaces.Box(
                    low=low[self.critic_goal_indices],
                    high=high[self.critic_goal_indices],
                    dtype=obs_space.dtype,
                ),
            }
        )

    def _to_dict_obs(self, obs):
        obs = np.asarray(obs).flatten()
        state = obs[:-self.goal_dim]
        goal = obs[-self.goal_dim:]
        critic_goal = obs[self.critic_goal_indices]
        return {
            "observation": state.astype(np.float32, copy=False),
            "desired_goal": goal.astype(np.float32, copy=False),
            "critic_goal": critic_goal.astype(np.float32, copy=False),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        result = self._to_dict_obs(obs)
        return result, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        result = self._to_dict_obs(obs)
        return result, reward, terminated, truncated, info
