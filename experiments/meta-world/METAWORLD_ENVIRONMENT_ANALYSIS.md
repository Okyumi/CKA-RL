# MetaWorld Goal-Observable Environment Analysis

## Overview

This document provides a detailed analysis of MetaWorld goal-observable environments based on runtime inspection. The analysis is based on the `HammerV2GoalObservable` environment, which is representative of MetaWorld's goal-observable task suite.

## Environment Details

### Environment Type

- **Class**: `HammerV2GoalObservable`
- **Module**: `metaworld.envs.mujoco.env_dict`
- **Inheritance Chain**: 
  ```
  HammerV2GoalObservable → SawyerHammerEnvV2 → SawyerXYZEnv → 
  SawyerMocapBase → MujocoEnv → BaseMujocoEnv → Env → Generic → EzPickle → object
  ```
- **API Compatibility**: Gymnasium-compatible (not created via `gym.make()`, but implements Gymnasium interface)

### Key Attributes

- **`max_path_length`**: 500 (maximum episode length)
- **`reward_range`**: (-inf, inf)
- **`metadata`**: `{'render_modes': ['human', 'rgb_array', 'depth_array'], 'render_fps': 80}`
- **`goal`**: `[0., 0.5, 0.]` (numpy array, shape (3,))
- **`goal_space`**: `Box([0.2399, 0.7399, 0.109], [0.2401, 0.7401, 0.111], (3,), float32)`
- **`current_task`**: 0 (task identifier)

## Observation Format

### Observation Space

- **Type**: `gymnasium.spaces.Box` (NOT a Dict!)
- **Shape**: `(39,)` - **Flat 1D array**
- **Dtype**: `float64`
- **Bounds**: 
  - Low: varies (e.g., `[-0.525, 0.348, -0.0525, -1., -inf, ...]`)
  - High: varies (e.g., `[0.525, 1.025, 0.7, 1., inf, ...]`)

### Observation Structure

The observation is a **flat numpy array of 39 dimensions**, NOT a dictionary. The structure appears to be:

```
observation = [
    state_dimensions (first 36 dims),  # Robot state, object positions, etc.
    goal_dimensions (last 3 dims)       # [x, y, z] goal position
]
```

**Evidence from inspection:**
- Last 10 values: `[0.24, 0.64, 0.11, 1., 0., 0., 0., 0.24, 0.74, 0.11]`
- The last 3 values `[0.24, 0.74, 0.11]` match the goal space bounds `[0.2399-0.2401, 0.7399-0.7401, 0.109-0.111]`
- The environment's `goal` attribute is `[0., 0.5, 0.]` (normalized/scaled differently)

### Observation Values (Example)

From `reset()`:
- Shape: `(39,)`
- Dtype: `float64`
- Min: `-0.0297`
- Max: `1.0`
- Mean: `~0.286`
- First 10: `[0.0059, 0.3997, 0.1949, 1.0, 0.0361, 0.4321, 0.0, 0.9996, 0.0002, -0.0297]`
- Last 10: `[0.24, 0.64, 0.11, 1., 0., 0., 0., 0.24, 0.74, 0.11]`

## Action Format

- **Type**: `gymnasium.spaces.Box`
- **Shape**: `(4,)`
- **Dtype**: `float64`
- **Bounds**: `[-1., -1., -1., -1.]` to `[1., 1., 1., 1.]`
- **Interpretation**: 4D continuous action space (likely robot arm control)

## Return Values

### `reset()` Method

**Returns**: `(observation, info)`

- **`observation`**: 
  - Type: `numpy.ndarray`
  - Shape: `(39,)`
  - Dtype: `float64`
  - Format: Flat array (NOT dict)
  
- **`info`**: 
  - Type: `dict`
  - Keys: `[]` (empty dict on reset)
  - Content: No additional information provided on reset

### `step(action)` Method

**Returns**: `(observation, reward, terminated, truncated, info)`

- **`observation`**: 
  - Type: `numpy.ndarray`
  - Shape: `(39,)`
  - Dtype: `float64`
  - Format: Flat array (NOT dict)
  - Same structure as `reset()` observation

- **`reward`**: 
  - Type: `numpy.float64`
  - Value: Continuous reward (e.g., `0.3737`)
  - Range: `(-inf, inf)`

- **`terminated`**: 
  - Type: `bool`
  - Indicates: Whether episode ended due to task completion/failure

- **`truncated`**: 
  - Type: `bool`
  - Indicates: Whether episode ended due to time limit

- **`info`**: 
  - Type: `dict`
  - Keys: `['success', 'near_object', 'grasp_success', 'grasp_reward', 'in_place_reward', 'obj_to_target', 'unscaled_reward']`
  - Content:
    - `success`: `float` (0.0 or 1.0) - Task completion indicator
    - `near_object`: `float` - Distance/proximity metric
    - `grasp_success`: `bool` - Whether object was grasped
    - `grasp_reward`: `float` - Reward component for grasping
    - `in_place_reward`: `float` - Reward component for placement
    - `obj_to_target`: `int` - Object-to-target distance metric
    - `unscaled_reward`: `float` - Raw reward before scaling

## Goal Extraction for Goal-Conditional RL

### ⚠️ Important Limitations

**Critical Issue**: While the goal coordinates are in the last 3 dimensions of the observation, **the flat array alone does not tell us which object/subject the goal refers to**. Different MetaWorld tasks have different success conditions:

1. **Object 1's position** must match goal (e.g., push object 1 to target location)
2. **Object 2's position** must match goal (e.g., place object 2 at target)
3. **Hand/end-effector position** must match goal (e.g., move hand to target location)
4. **Combined conditions** (e.g., object in hand at target location)

### Goal Location in Observation

**Yes, the goal coordinates are in the last 3 dimensions**, but this is only part of the story:

1. **Observation shape**: `(39,)`
2. **Goal dimensions**: Last 3 indices `[36:39]` or `[-3:]`
3. **Goal format**: `[x, y, z]` position coordinates
4. **Goal bounds**: Approximately `[0.24, 0.74, 0.11]` (varies per task/episode)

### Using Environment Attributes to Extract Goal

**Better approach**: Use the environment's `goal` attribute directly rather than extracting from observation:

```python
# Method 1: Direct access to goal attribute (RECOMMENDED)
goal = env.goal  # Shape: (3,), dtype: float32
# This is the canonical goal for the current episode

# Method 2: Extract from observation (also works, but less direct)
observation = env.reset()[0]  # Shape: (39,)
goal_from_obs = observation[-3:]  # Shape: (3,)
# Note: This should match env.goal, but env.goal is more reliable
```

### Goal Space Details

- **Goal Attribute**: `env.goal` - Direct access to current goal
- **Goal Space Type**: `gymnasium.spaces.Box`
- **Goal Space Shape**: `(3,)`
- **Goal Space Dtype**: `float32`
- **Goal Space Bounds**: 
  - Low: `[0.2399, 0.7399, 0.109]`
  - High: `[0.2401, 0.7401, 0.111]`
- **Note**: The goal space bounds are very tight, suggesting goals are sampled from a small region per task

### Identifying Which Object the Goal Refers To

**Problem**: The observation array contains positions for:
- Hand/end-effector (at some indices)
- Object 1 (at some indices)
- Object 2 (at some indices, if present)
- Other state variables

**But the flat array doesn't label which indices correspond to which entity!**

**Solution**: Use environment attributes and methods to identify the target:

1. **Check environment class name**: Different tasks have different target objects
   - `HammerV2GoalObservable`: Goal likely refers to hammer position
   - `PushV2GoalObservable`: Goal likely refers to object position
   - `ReachV2GoalObservable`: Goal likely refers to hand position

2. **Use environment methods**: MetaWorld environments may have methods like:
   - `get_obs()` - might return structured observation
   - `_get_obs()` - internal method with structured data
   - `compute_reward()` - might reveal which object is checked

3. **Inspect object position attributes**: Look for attributes like:
   - `obj_pos` / `object_pos` - object positions
   - `hand_pos` / `end_effector_pos` - hand position
   - `target_pos` - target position

4. **Check info dict**: The `step()` info dict contains task-specific information that might indicate the target

### Recommended Extraction Method

```python
# Best practice: Use environment attributes
def extract_goal(env):
    """Extract goal using environment attribute (most reliable)"""
    return env.goal.copy()  # Returns (3,) array with [x, y, z]

def extract_state_and_goal(env, observation):
    """Extract state and goal from observation"""
    state = observation[:-3]  # First 36 dims
    goal = observation[-3:]  # Last 3 dims (should match env.goal)
    return state, goal

# Example usage
obs, info = env.reset()
goal = env.goal  # Use this!
state = obs[:-3]  # State without goal
```

### Important Notes

1. **Observation is NOT a Dict**: Despite being "goal-observable", MetaWorld returns a **flat array**, not a dictionary with separate `observation` and `goal` keys.

2. **Goal is Embedded**: The goal is concatenated at the end of the state vector, making it easy to extract but requiring manual splitting.

3. **Goal Changes Per Episode**: Each `reset()` call samples a new goal, which appears in the last 3 dimensions of the observation and is also available via `env.goal`.

4. **State Dimensions**: The first 36 dimensions contain the state information (robot pose, object positions, velocities, etc.), but **the indices are not labeled** - you don't know which indices correspond to which object.

5. **Goal Target Varies by Task**: Different tasks have different success conditions:
   - Some tasks: Object 1's position must match goal
   - Some tasks: Object 2's position must match goal  
   - Some tasks: Hand position must match goal
   - Some tasks: Object in hand at goal location
   
   **You need to know the task-specific success condition to properly interpret the goal.**

6. **Use `env.goal` Attribute**: The most reliable way to get the goal is via `env.goal` attribute rather than extracting from observation, as it's the canonical source of truth.

7. **Object Position Indices Unknown**: The observation array doesn't label which indices contain which object's position. You may need to:
   - Inspect environment attributes (e.g., `obj_pos`, `hand_pos`)
   - Check environment class/methods
   - Refer to MetaWorld documentation for task-specific observation structure
   - Use trial and error with known object positions

## Summary

- **Environment**: MetaWorld goal-observable tasks return **flat numpy arrays**, not dictionaries
- **Observation Shape**: `(39,)` - 36 state dims + 3 goal dims
- **Goal Location**: 
  - ✅ Last 3 dimensions `[-3:]` or `[36:39]` in observation
  - ✅ **Better**: Use `env.goal` attribute directly (most reliable)
- **State Location**: First 36 dimensions `[:36]` or `[:-3]`
- **Action Shape**: `(4,)` continuous actions in `[-1, 1]`
- **Info Dict**: Rich task-specific information available in `step()` info dict

### ⚠️ Critical Limitations for Goal-Conditional RL

1. **Goal Coordinates Known, Target Object Unknown**: 
   - You can extract goal coordinates from `observation[-3:]` or `env.goal`
   - **BUT** you don't know which object/subject the goal refers to just from the flat array
   - Different tasks have different success conditions (object 1, object 2, hand, etc.)

2. **State Indices Unlabeled**: 
   - The first 36 dimensions contain state (robot, objects, velocities)
   - **BUT** the indices are not labeled - you don't know which indices correspond to which object's position

3. **Task-Specific Interpretation Required**:
   - Must know task-specific success condition to properly interpret the goal
   - Use environment attributes (`env.goal`, object position attributes) to identify target
   - Check environment class name and methods for task-specific information

### Recommended Approach

```python
# Best practice for goal extraction
goal = env.goal  # Use environment attribute (most reliable)

# For state-goal separation
obs, info = env.reset()
state = obs[:-3]  # State without goal
goal_from_obs = obs[-3:]  # Goal from observation (should match env.goal)

# To identify which object the goal refers to:
# 1. Check environment class name (task-specific)
# 2. Inspect environment attributes (obj_pos, hand_pos, etc.)
# 3. Check info dict for task-specific metrics
# 4. Refer to MetaWorld documentation for task structure
```
