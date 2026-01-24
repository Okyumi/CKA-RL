import gymnasium as gym
import numpy as np


class GoalObsWrapper(gym.Wrapper):
    """Minimal goal wrapper for MetaWorld goal-observable tasks.
    Splits the flat observation into (state, goal) by assuming
    the last `goal_dim` entries represent the goal.
    """

    def __init__(
        self,
        env: gym.Env,
        goal_dim: int = 3,
        augment_goal: bool = False,
        eef_indices=(0, 1, 2),
    ):
        super().__init__(env)
        self.goal_dim = goal_dim
        self.augment_goal = augment_goal
        self.eef_indices = list(eef_indices)
        self.critic_goal_indices = [4, 5, 6]
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
        if self.augment_goal:
            eef_low = low[self.eef_indices]
            eef_high = high[self.eef_indices]
            delta_low = eef_low - goal_high
            delta_high = eef_high - goal_low
            desired_low = np.concatenate([goal_low, delta_low], axis=0)
            desired_high = np.concatenate([goal_high, delta_high], axis=0)
            critic_low = desired_low
            critic_high = desired_high
        else:
            desired_low = goal_low
            desired_high = goal_high
            critic_low = low[self.critic_goal_indices]
            critic_high = high[self.critic_goal_indices]

        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    low=state_low, high=state_high, dtype=obs_space.dtype
                ),
                "desired_goal": gym.spaces.Box(
                    low=desired_low, high=desired_high, dtype=obs_space.dtype
                ),
                "critic_goal": gym.spaces.Box(
                    low=critic_low,
                    high=critic_high,
                    dtype=obs_space.dtype,
                ),
            }
        )

    def _to_dict_obs(self, obs):
        obs = np.asarray(obs).flatten()
        state = obs[:-self.goal_dim]
        goal = obs[-self.goal_dim:]
        critic_goal = obs[self.critic_goal_indices]
        if self.augment_goal:
            eef_pos = obs[self.eef_indices]
            delta = eef_pos - critic_goal
            desired_goal = np.concatenate([goal, np.zeros_like(delta)], axis=0)
            critic_goal = np.concatenate([critic_goal, delta], axis=0)
        else:
            desired_goal = goal
        return {
            "observation": state.astype(np.float32, copy=False),
            "desired_goal": desired_goal.astype(np.float32, copy=False),
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
