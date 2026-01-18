# Trajectory Buffer Explanation

## JAX Implementation Analysis

### Key Components

1. **TrajectoryUniformSamplingQueue**
   - Stores data in shape `(max_replay_size, num_envs, data_size)`
   - Maintains `insert_position` and `sample_position` for circular buffer
   - Tracks trajectories by environment and episode

2. **insert_internal**
   - Inserts full trajectories (episode_length transitions)
   - Handles buffer rolling when full
   - Updates insert_position

3. **sample_internal**
   - Samples random environments
   - For each env, samples a random starting position
   - Returns consecutive transitions of length `episode_length` (full trajectories)

4. **flatten_crl_fn** (Key Function)
   - Takes a trajectory (sequence of transitions)
   - For each state-action pair at time `t`:
     - Creates an upper triangular matrix where `is_future_mask[i,j] = 1` if `j > i`
     - Applies geometric discount: `probs = is_future_mask * gamma^(j-i)`
     - Ensures same episode: `probs = probs * (seed[i] == seed[j])`
     - Samples a future state index using categorical distribution
   - Extracts goals from sampled future states
   - Returns transitions with goals from future states

## PyTorch Implementation

### TrajectoryBuffer Class

**Key Features:**
1. Stores full trajectories as lists of `Transition` objects
2. Groups transitions by episode ID
3. Samples random future states using geometric distribution
4. Extracts goals from future states

**Methods:**

1. **`add()`**: 
   - Adds transitions one by one
   - Groups them into trajectories
   - Saves trajectory when episode ends

2. **`sample_trajectories()`**:
   - Samples full trajectories from buffer
   - Returns list of trajectories

3. **`sample_with_future_goals()`**:
   - Samples trajectories
   - For each transition (s_t, a_t) in trajectory:
     - Samples random future state from same trajectory using `_sample_future_state_idx()`
     - Extracts goal from future state
     - Creates observation dict with goal from future state
   - Returns batch in DictReplayBuffer-compatible format

4. **`_sample_future_state_idx()`**:
   - Implements geometric distribution: `P(k) = (1 - gamma) * gamma^(k-1)`
   - Samples a future state index based on this distribution

## Differences from JAX Implementation

1. **Storage**: 
   - JAX: Stores flattened data in 3D array `(max_replay_size, num_envs, data_size)`
   - PyTorch: Stores trajectories as list of lists (more Pythonic, easier to work with)

2. **Sampling**:
   - JAX: Samples full trajectories, then processes them with `flatten_crl_fn`
   - PyTorch: Samples trajectories and processes them inline

3. **Episode Tracking**:
   - JAX: Uses seed matching to ensure same episode
   - PyTorch: Uses episode_id stored in each transition

4. **Goal Extraction**:
   - JAX: Extracts from flat observation using indices
   - PyTorch: Handles both dict and flat observations

## Usage

The buffer is designed to replace `DictReplayBuffer` in the training loop:

```python
from buffer import TrajectoryBuffer

rb = TrajectoryBuffer(
    buffer_size=args.buffer_size,
    observation_space=envs.single_observation_space,
    action_space=envs.single_action_space,
    device=device,
    episode_length=500,
    gamma=0.99,
    goal_start_idx=4,
    goal_end_idx=7,
)

# Add transitions (same interface as DictReplayBuffer)
rb.add(obs, next_obs, actions, rewards, terminations, infos)

# Sample (automatically uses future goals from same trajectory)
data = rb.sample(batch_size)
```

## Next Steps

1. Replace `DictReplayBuffer` with `TrajectoryBuffer` in `run_contrastiveRL.py`
2. Update training loop to use the new buffer
3. Verify that goals are correctly sampled from future states in same trajectory
4. Test that negative sampling works correctly (goals from different trajectories)
