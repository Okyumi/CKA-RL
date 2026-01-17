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
        
        # ===== DEBUGGING: Print environment information =====
        print("\n" + "="*80)
        print("DEBUG: GoalObsWrapper - Environment Inspection")
        print("="*80)
        
        # Print environment class and type
        print(f"\n[Environment Type]")
        print(f"  env class: {type(env).__name__}")
        print(f"  env module: {type(env).__module__}")
        print(f"  env type: {type(env)}")
        print(f"  env MRO: {[cls.__name__ for cls in type(env).__mro__]}")
        
        # Print environment attributes
        print(f"\n[Environment Attributes]")
        env_attrs = [attr for attr in dir(env) if not attr.startswith('_')]
        print(f"  public attributes count: {len(env_attrs)}")
        important_attrs = ['max_path_length', 'max_episode_steps', 'reward_range', 
                          'metadata', 'spec', 'unwrapped', 'render_mode']
        for attr in important_attrs:
            if hasattr(env, attr):
                value = getattr(env, attr)
                print(f"  {attr}: {value} (type: {type(value)})")
        
        # Print any custom MetaWorld-specific attributes
        metaworld_attrs = [attr for attr in env_attrs if 'goal' in attr.lower() or 'task' in attr.lower() or 'seed' in attr.lower()]
        if metaworld_attrs:
            print(f"\n  MetaWorld-specific attributes:")
            for attr in metaworld_attrs[:10]:  # Limit to first 10
                try:
                    value = getattr(env, attr)
                    if not callable(value):
                        print(f"    {attr}: {value} (type: {type(value)})")
                except:
                    pass
        
        # Print ALL object/position related attributes
        print(f"\n[Object/Position Related Attributes]")
        obj_attrs = [attr for attr in env_attrs if any(keyword in attr.lower() for keyword in 
                    ['obj', 'object', 'pos', 'position', 'hand', 'end', 'effector', 'target', 'gripper', 'arm'])]
        if obj_attrs:
            print(f"  Found {len(obj_attrs)} object/position related attributes:")
            for attr in sorted(obj_attrs):
                try:
                    value = getattr(env, attr)
                    if not callable(value):
                        if isinstance(value, np.ndarray):
                            print(f"    {attr}: shape={value.shape}, dtype={value.dtype}, sample={value[:5] if value.size > 5 else value}")
                        elif isinstance(value, (list, tuple)) and len(value) > 0:
                            print(f"    {attr}: length={len(value)}, first={value[0] if len(value) > 0 else 'N/A'}")
                        else:
                            print(f"    {attr}: {value} (type: {type(value)})")
                except Exception as e:
                    print(f"    {attr}: <error accessing: {e}>")
        
        # Print methods that might help identify goal target
        print(f"\n[Methods Related to Goal/Objects]")
        goal_methods = [attr for attr in env_attrs if any(keyword in attr.lower() for keyword in 
                       ['goal', 'obj', 'target', 'success', 'achieved', 'get_', 'compute'])]
        if goal_methods:
            print(f"  Found {len(goal_methods)} potentially relevant methods:")
            for method in sorted(goal_methods)[:15]:  # Limit to first 15
                try:
                    value = getattr(env, method)
                    if callable(value):
                        # Try to get method signature if possible
                        import inspect
                        try:
                            sig = inspect.signature(value)
                            print(f"    {method}{sig}")
                        except:
                            print(f"    {method}() - callable")
                except:
                    pass
        
        # Print action space
        print(f"\n[Action Space]")
        print(f"  type: {type(env.action_space)}")
        print(f"  shape: {env.action_space.shape}")
        print(f"  dtype: {env.action_space.dtype}")
        if hasattr(env.action_space, 'low'):
            print(f"  low: {env.action_space.low}")
        if hasattr(env.action_space, 'high'):
            print(f"  high: {env.action_space.high}")
        
        # Print observation space
        print(f"\n[Observation Space]")
        print(f"  type: {type(env.observation_space)}")
        print(f"  is Dict: {isinstance(env.observation_space, gym.spaces.Dict)}")
        
        if isinstance(env.observation_space, gym.spaces.Dict):
            print(f"  Dict keys: {list(env.observation_space.spaces.keys())}")
            for key, space in env.observation_space.spaces.items():
                print(f"    '{key}':")
                print(f"      type: {type(space)}")
                print(f"      shape: {space.shape}")
                print(f"      dtype: {space.dtype}")
                if hasattr(space, 'low'):
                    print(f"      low shape: {space.low.shape if hasattr(space.low, 'shape') else 'N/A'}")
                    print(f"      low sample: {space.low[:5] if hasattr(space.low, '__getitem__') else space.low}")
                if hasattr(space, 'high'):
                    print(f"      high shape: {space.high.shape if hasattr(space.high, 'shape') else 'N/A'}")
                    print(f"      high sample: {space.high[:5] if hasattr(space.high, '__getitem__') else space.high}")
        else:
            print(f"  shape: {env.observation_space.shape}")
            print(f"  dtype: {env.observation_space.dtype}")
            if hasattr(env.observation_space, 'low'):
                print(f"  low shape: {env.observation_space.low.shape if hasattr(env.observation_space.low, 'shape') else 'N/A'}")
                print(f"  low sample: {env.observation_space.low[:5] if hasattr(env.observation_space.low, '__getitem__') else env.observation_space.low}")
            if hasattr(env.observation_space, 'high'):
                print(f"  high shape: {env.observation_space.high.shape if hasattr(env.observation_space.high, 'shape') else 'N/A'}")
                print(f"  high sample: {env.observation_space.high[:5] if hasattr(env.observation_space.high, '__getitem__') else env.observation_space.high}")
        
        # Test reset to see actual observation format
        print(f"\n[RAW MetaWorld reset() - Complete Output]")
        try:
            test_obs, test_info = env.reset()
            
            print(f"\n  [Observation from reset()]")
            print(f"    type: {type(test_obs)}")
            print(f"    is dict: {isinstance(test_obs, dict)}")
            print(f"    is numpy array: {isinstance(test_obs, np.ndarray)}")
            
            if isinstance(test_obs, dict):
                print(f"    dict keys: {list(test_obs.keys())}")
                print(f"    number of keys: {len(test_obs.keys())}")
                for key, value in test_obs.items():
                    print(f"\n    Key '{key}':")
                    print(f"      type: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"      shape: {value.shape}")
                        print(f"      dtype: {value.dtype}")
                        print(f"      size: {value.size}")
                        print(f"      min: {np.min(value)}")
                        print(f"      max: {np.max(value)}")
                        print(f"      mean: {np.mean(value)}")
                        if value.size <= 20:
                            print(f"      full values: {value}")
                        else:
                            print(f"      first 10 values: {value[:10]}")
                            print(f"      last 10 values: {value[-10:]}")
                    elif isinstance(value, (list, tuple)):
                        print(f"      length: {len(value)}")
                        print(f"      first few: {value[:5] if len(value) > 5 else value}")
                    else:
                        print(f"      value: {value}")
            else:
                if isinstance(test_obs, np.ndarray):
                    print(f"    shape: {test_obs.shape}")
                    print(f"    dtype: {test_obs.dtype}")
                    print(f"    size: {test_obs.size}")
                    print(f"    min: {np.min(test_obs)}")
                    print(f"    max: {np.max(test_obs)}")
                    print(f"    mean: {np.mean(test_obs)}")
                    if test_obs.size <= 20:
                        print(f"    full values: {test_obs}")
                    else:
                        print(f"    first 10 values: {test_obs[:10]}")
                        print(f"    last 10 values: {test_obs[-10:]}")
                else:
                    print(f"    value: {test_obs}")
                    print(f"    repr: {repr(test_obs)}")
            
            print(f"\n  [Info dict from reset()]")
            print(f"    type: {type(test_info)}")
            if isinstance(test_info, dict):
                print(f"    dict keys: {list(test_info.keys())}")
                print(f"    number of keys: {len(test_info.keys())}")
                for key, value in test_info.items():
                    print(f"\n    Key '{key}':")
                    print(f"      type: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"      shape: {value.shape}")
                        print(f"      dtype: {value.dtype}")
                        if value.size <= 10:
                            print(f"      values: {value}")
                        else:
                            print(f"      first 5: {value[:5]}")
                    elif isinstance(value, (list, tuple)):
                        print(f"      length: {len(value)}")
                        print(f"      values: {value}")
                    elif isinstance(value, dict):
                        print(f"      nested dict keys: {list(value.keys())}")
                        for k, v in value.items():
                            print(f"        '{k}': {v} (type: {type(v)})")
                    else:
                        print(f"      value: {value}")
            else:
                print(f"    value: {test_info}")
                print(f"    repr: {repr(test_info)}")
        except Exception as e:
            print(f"  ERROR during reset test: {e}")
            import traceback
            traceback.print_exc()
        
        # Test step to see observation format after step
        print(f"\n[RAW MetaWorld step() - Complete Output]")
        try:
            test_action = env.action_space.sample()
            print(f"\n  [Action used for step()]")
            print(f"    type: {type(test_action)}")
            print(f"    shape: {test_action.shape if hasattr(test_action, 'shape') else 'N/A'}")
            print(f"    dtype: {test_action.dtype if hasattr(test_action, 'dtype') else 'N/A'}")
            print(f"    values: {test_action}")
            
            test_obs_step, test_reward, test_term, test_trunc, test_info_step = env.step(test_action)
            
            print(f"\n  [Observation from step()]")
            print(f"    type: {type(test_obs_step)}")
            print(f"    is dict: {isinstance(test_obs_step, dict)}")
            print(f"    is numpy array: {isinstance(test_obs_step, np.ndarray)}")
            
            if isinstance(test_obs_step, dict):
                print(f"    dict keys: {list(test_obs_step.keys())}")
                print(f"    number of keys: {len(test_obs_step.keys())}")
                for key, value in test_obs_step.items():
                    print(f"\n    Key '{key}':")
                    print(f"      type: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"      shape: {value.shape}")
                        print(f"      dtype: {value.dtype}")
                        print(f"      size: {value.size}")
                        print(f"      min: {np.min(value)}")
                        print(f"      max: {np.max(value)}")
                        print(f"      mean: {np.mean(value)}")
                        if value.size <= 20:
                            print(f"      full values: {value}")
                        else:
                            print(f"      first 10 values: {value[:10]}")
                            print(f"      last 10 values: {value[-10:]}")
                    elif isinstance(value, (list, tuple)):
                        print(f"      length: {len(value)}")
                        print(f"      first few: {value[:5] if len(value) > 5 else value}")
                    else:
                        print(f"      value: {value}")
            else:
                if isinstance(test_obs_step, np.ndarray):
                    print(f"    shape: {test_obs_step.shape}")
                    print(f"    dtype: {test_obs_step.dtype}")
                    print(f"    size: {test_obs_step.size}")
                    print(f"    min: {np.min(test_obs_step)}")
                    print(f"    max: {np.max(test_obs_step)}")
                    print(f"    mean: {np.mean(test_obs_step)}")
                    if test_obs_step.size <= 20:
                        print(f"    full values: {test_obs_step}")
                    else:
                        print(f"    first 10 values: {test_obs_step[:10]}")
                        print(f"    last 10 values: {test_obs_step[-10:]}")
                else:
                    print(f"    value: {test_obs_step}")
                    print(f"    repr: {repr(test_obs_step)}")
            
            print(f"\n  [Reward from step()]")
            print(f"    type: {type(test_reward)}")
            print(f"    value: {test_reward}")
            if isinstance(test_reward, np.ndarray):
                print(f"    shape: {test_reward.shape}")
                print(f"    dtype: {test_reward.dtype}")
            
            print(f"\n  [Terminated from step()]")
            print(f"    type: {type(test_term)}")
            print(f"    value: {test_term}")
            if isinstance(test_term, np.ndarray):
                print(f"    shape: {test_term.shape}")
                print(f"    dtype: {test_term.dtype}")
            
            print(f"\n  [Truncated from step()]")
            print(f"    type: {type(test_trunc)}")
            print(f"    value: {test_trunc}")
            if isinstance(test_trunc, np.ndarray):
                print(f"    shape: {test_trunc.shape}")
                print(f"    dtype: {test_trunc.dtype}")
            
            print(f"\n  [Info dict from step()]")
            print(f"    type: {type(test_info_step)}")
            if isinstance(test_info_step, dict):
                print(f"    dict keys: {list(test_info_step.keys())}")
                print(f"    number of keys: {len(test_info_step.keys())}")
                for key, value in test_info_step.items():
                    print(f"\n    Key '{key}':")
                    print(f"      type: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"      shape: {value.shape}")
                        print(f"      dtype: {value.dtype}")
                        if value.size <= 10:
                            print(f"      values: {value}")
                        else:
                            print(f"      first 5: {value[:5]}")
                    elif isinstance(value, (list, tuple)):
                        print(f"      length: {len(value)}")
                        print(f"      values: {value}")
                    elif isinstance(value, dict):
                        print(f"      nested dict keys: {list(value.keys())}")
                        for k, v in value.items():
                            print(f"        '{k}': {v} (type: {type(v)})")
                    else:
                        print(f"      value: {value}")
            else:
                print(f"    value: {test_info_step}")
                print(f"    repr: {repr(test_info_step)}")
        except Exception as e:
            print(f"  ERROR during step test: {e}")
            import traceback
            traceback.print_exc()
        
        print("="*80 + "\n")
        # ===== END DEBUGGING =====
        
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
        
        # ===== DEBUGGING: Print RAW MetaWorld output from reset =====
        if not hasattr(self, '_reset_debugged'):
            print("\n" + "="*80)
            print("[DEBUG: GoalObsWrapper.reset() - RAW MetaWorld Output (First Call)]")
            print("="*80)
            
            print(f"\n[RAW Observation]")
            print(f"  type: {type(obs)}")
            print(f"  is dict: {isinstance(obs, dict)}")
            print(f"  is numpy array: {isinstance(obs, np.ndarray)}")
            
            if isinstance(obs, dict):
                print(f"  dict keys: {list(obs.keys())}")
                print(f"  number of keys: {len(obs.keys())}")
                for key, value in obs.items():
                    print(f"\n  Key '{key}':")
                    print(f"    type: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"    shape: {value.shape}")
                        print(f"    dtype: {value.dtype}")
                        print(f"    size: {value.size}")
                        print(f"    min: {np.min(value)}")
                        print(f"    max: {np.max(value)}")
                        print(f"    mean: {np.mean(value)}")
                        if value.size <= 20:
                            print(f"    full values: {value}")
                        else:
                            print(f"    first 10 values: {value[:10]}")
                            print(f"    last 10 values: {value[-10:]}")
                    elif isinstance(value, (list, tuple)):
                        print(f"    length: {len(value)}")
                        print(f"    values: {value}")
                    else:
                        print(f"    value: {value}")
            else:
                if isinstance(obs, np.ndarray):
                    print(f"  shape: {obs.shape}")
                    print(f"  dtype: {obs.dtype}")
                    print(f"  size: {obs.size}")
                    print(f"  min: {np.min(obs)}")
                    print(f"  max: {np.max(obs)}")
                    print(f"  mean: {np.mean(obs)}")
                    if obs.size <= 20:
                        print(f"  full values: {obs}")
                    else:
                        print(f"  first 10 values: {obs[:10]}")
                        print(f"  last 10 values: {obs[-10:]}")
                else:
                    print(f"  value: {obs}")
                    print(f"  repr: {repr(obs)}")
            
            print(f"\n[RAW Info Dict]")
            print(f"  type: {type(info)}")
            if isinstance(info, dict):
                print(f"  dict keys: {list(info.keys())}")
                print(f"  number of keys: {len(info.keys())}")
                for key, value in info.items():
                    print(f"\n  Key '{key}':")
                    print(f"    type: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"    shape: {value.shape}")
                        print(f"    dtype: {value.dtype}")
                        if value.size <= 10:
                            print(f"    values: {value}")
                        else:
                            print(f"    first 5: {value[:5]}")
                    elif isinstance(value, (list, tuple)):
                        print(f"    length: {len(value)}")
                        print(f"    values: {value}")
                    elif isinstance(value, dict):
                        print(f"    nested dict keys: {list(value.keys())}")
                        for k, v in value.items():
                            print(f"      '{k}': {v} (type: {type(v)})")
                    else:
                        print(f"    value: {value}")
            else:
                print(f"  value: {info}")
                print(f"  repr: {repr(info)}")
            
            print("="*80 + "\n")
            self._reset_debugged = True
        # ===== END DEBUGGING =====
        
        result = self._to_dict_obs(obs)
        return result, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # ===== DEBUGGING: Print RAW MetaWorld output from step =====
        if not hasattr(self, '_step_debugged'):
            print("\n" + "="*80)
            print("[DEBUG: GoalObsWrapper.step() - RAW MetaWorld Output (First Call)]")
            print("="*80)
            
            print(f"\n[RAW Observation]")
            print(f"  type: {type(obs)}")
            print(f"  is dict: {isinstance(obs, dict)}")
            print(f"  is numpy array: {isinstance(obs, np.ndarray)}")
            
            if isinstance(obs, dict):
                print(f"  dict keys: {list(obs.keys())}")
                print(f"  number of keys: {len(obs.keys())}")
                for key, value in obs.items():
                    print(f"\n  Key '{key}':")
                    print(f"    type: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"    shape: {value.shape}")
                        print(f"    dtype: {value.dtype}")
                        print(f"    size: {value.size}")
                        print(f"    min: {np.min(value)}")
                        print(f"    max: {np.max(value)}")
                        print(f"    mean: {np.mean(value)}")
                        if value.size <= 20:
                            print(f"    full values: {value}")
                        else:
                            print(f"    first 10 values: {value[:10]}")
                            print(f"    last 10 values: {value[-10:]}")
                    elif isinstance(value, (list, tuple)):
                        print(f"    length: {len(value)}")
                        print(f"    values: {value}")
                    else:
                        print(f"    value: {value}")
            else:
                if isinstance(obs, np.ndarray):
                    print(f"  shape: {obs.shape}")
                    print(f"  dtype: {obs.dtype}")
                    print(f"  size: {obs.size}")
                    print(f"  min: {np.min(obs)}")
                    print(f"  max: {np.max(obs)}")
                    print(f"  mean: {np.mean(obs)}")
                    if obs.size <= 20:
                        print(f"  full values: {obs}")
                    else:
                        print(f"  first 10 values: {obs[:10]}")
                        print(f"  last 10 values: {obs[-10:]}")
                else:
                    print(f"  value: {obs}")
                    print(f"  repr: {repr(obs)}")
            
            print(f"\n[RAW Reward]")
            print(f"  type: {type(reward)}")
            print(f"  value: {reward}")
            if isinstance(reward, np.ndarray):
                print(f"  shape: {reward.shape}")
                print(f"  dtype: {reward.dtype}")
            
            print(f"\n[RAW Terminated]")
            print(f"  type: {type(terminated)}")
            print(f"  value: {terminated}")
            if isinstance(terminated, np.ndarray):
                print(f"  shape: {terminated.shape}")
                print(f"  dtype: {terminated.dtype}")
            
            print(f"\n[RAW Truncated]")
            print(f"  type: {type(truncated)}")
            print(f"  value: {truncated}")
            if isinstance(truncated, np.ndarray):
                print(f"  shape: {truncated.shape}")
                print(f"  dtype: {truncated.dtype}")
            
            print(f"\n[RAW Info Dict]")
            print(f"  type: {type(info)}")
            if isinstance(info, dict):
                print(f"  dict keys: {list(info.keys())}")
                print(f"  number of keys: {len(info.keys())}")
                for key, value in info.items():
                    print(f"\n  Key '{key}':")
                    print(f"    type: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"    shape: {value.shape}")
                        print(f"    dtype: {value.dtype}")
                        if value.size <= 10:
                            print(f"    values: {value}")
                        else:
                            print(f"    first 5: {value[:5]}")
                    elif isinstance(value, (list, tuple)):
                        print(f"    length: {len(value)}")
                        print(f"    values: {value}")
                    elif isinstance(value, dict):
                        print(f"    nested dict keys: {list(value.keys())}")
                        for k, v in value.items():
                            print(f"      '{k}': {v} (type: {type(v)})")
                    else:
                        print(f"    value: {value}")
            else:
                print(f"  value: {info}")
                print(f"  repr: {repr(info)}")
            
            print("="*80 + "\n")
            self._step_debugged = True
        # ===== END DEBUGGING =====
        
        result = self._to_dict_obs(obs)
        return result, reward, terminated, truncated, info
