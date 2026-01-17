# MetaWorld Task Goal and Success Criterion Analysis

This document provides a detailed analysis of each task in the 10-task MetaWorld sequence, identifying:
1. What the goal position coordinates refer to
2. What the success criterion is for each task

Based on log file `13761473.out` inspection.

---

## Task 0: HammerV2GoalObservable

### Goal Position Reference
- **Goal attribute**: `[0., 0.5, 0.]` (normalized/scaled)
- **Goal space**: `Box([0.2399, 0.7399, 0.109], [0.2401, 0.7401, 0.111], (3,), float32)`
- **Observation last 3 values**: `[0.24, 0.74, 0.11]` (matches goal space bounds)
- **Object positions**:
  - `nail_init_pos`: `[0.24, 0.635, 0.11]`
  - `hammer_init_pos`: `[0.06578325, 0.43034913, 0.]`
  - `obj_init_pos`: `[0.06578325, 0.43034913, 0.]` (same as hammer)

**Conclusion**: The goal refers to the **nail's target position**. The nail needs to be hammered to the goal location.

### Success Criterion
- **Info dict metrics**:
  - `success`: `0.0` (float, 0.0 or 1.0)
  - `obj_to_target`: `0` (int) - distance metric
  - `grasp_success`: `False` (bool) - whether hammer is grasped
  - `in_place_reward`: `0.058` (float) - reward for placement
  - `grasp_reward`: `0.028` (float) - reward for grasping

**Success condition**: The **nail's position** must be within `TARGET_RADIUS` (0.05) of the goal position. The task requires:
1. Grasping the hammer (optional, but rewarded)
2. Using the hammer to drive the nail to the goal location

---

## Task 1: PushWallV2GoalObservable

### Goal Position Reference
- **Goal attribute**: `[0.05, 0.8, 0.015]`
- **Goal space**: `Box([-0.05, 0.85, 0.01], [0.05, 0.9, 0.02], (3,), float32)`
- **Observation last 3 values**: `[0.01005672, 0.89621789, 0.01987216]` (within goal space)
- **Object positions**:
  - `obj_init_pos`: `[0.03289162, 0.61517457, 0.01987216]`

**Conclusion**: The goal refers to the **object's (block's) target position** against the wall. The object needs to be pushed to the goal location.

### Success Criterion
- **Info dict metrics**:
  - `success`: `0.0`
  - `obj_to_target`: `0.280` (float) - distance from object to target
  - `grasp_success`: `0.0` (float) - no grasping required
  - `in_place_reward`: `0.141` (float) - reward for placement
  - `grasp_reward`: `0.026` (float) - minimal/no grasping reward

**Success condition**: The **object's position** must be within `TARGET_RADIUS` (0.05) of the goal position. The task requires pushing the object to the target location (no grasping needed).

---

## Task 2: FaucetCloseV2GoalObservable

### Goal Position Reference
- **Goal attribute**: Not explicitly shown in attributes, but `goal_space` exists
- **Goal space**: `Box([-0.5, 0.4, -0.15], [0.5, 1., 0.5], (3,), float32)`
- **Observation last 3 values**: `[-0.13888545, 0.81604869, 0.125]` (within goal space)
- **Object positions**:
  - `obj_init_pos`: `[0.06578325, 0.81517458, 0.]` (faucet handle)

**Conclusion**: The goal refers to the **faucet handle's target position** (closed position). The handle needs to be rotated/moved to the goal location.

### Success Criterion
- **Info dict metrics**:
  - `success`: `0.0`
  - `obj_to_target`: `0.259` (float) - distance from handle to target
  - `grasp_success`: `1.0` (float) - handle is grasped
  - `in_place_reward`: `0.062` (float) - reward for placement
  - `grasp_reward`: `0.502` (float) - significant grasping reward

**Success condition**: The **faucet handle's position** must be within `TARGET_RADIUS` (0.05) of the goal position. The task requires:
1. Grasping the faucet handle
2. Moving/rotating it to the closed position (goal location)

---

## Task 3: PushBackV2GoalObservable

### Goal Position Reference
- **Goal attribute**: `[0., 0.6, 0.02]`
- **Goal space**: `Box([-0.1, 0.6, 0.0199], [0.1, 0.7, 0.0201], (3,), float32)`
- **Observation last 3 values**: `[0.06039558, 0.61790289, 0.01993605]` (within goal space)
- **Object positions**:
  - `obj_init_pos`: `[0.06578325, 0.81517458, 0.01993605]`

**Conclusion**: The goal refers to the **object's target position** (pushed back location). The object needs to be pushed backward to the goal location.

### Success Criterion
- **Info dict metrics**:
  - `success`: `0.0`
  - `obj_to_target`: `0.217` (float) - distance from object to target
  - `grasp_success`: `0.0` (float) - no grasping required
  - `in_place_reward`: `0.158` (float) - reward for placement
  - `grasp_reward`: `0.013` (float) - minimal grasping reward

**Success condition**: The **object's position** must be within `TARGET_RADIUS` (0.05) of the goal position. The task requires pushing the object backward to the target location (no grasping needed).

---

## Task 4: StickPullV2GoalObservable

### Goal Position Reference
- **Goal attribute**: `[0., 0.6, 0.02]`
- **Goal space**: `Box([0.35, 0.45, 0.0199], [0.45, 0.55, 0.0201], (3,), float32)`
- **Observation last 3 values**: `[0.41005671, 0.54243583, 0.02]` (within goal space)
- **Object positions**:
  - `obj_init_pos`: `[0.2, 0.6, 0.]` (main object)
  - `stick_init_pos`: `[-0.01710838, 0.58034912, 0.02]` (stick)

**Conclusion**: The goal refers to the **object's target position** after being pulled. The object needs to be pulled to the goal location using the stick.

### Success Criterion
- **Info dict metrics**:
  - `success`: `0.0`
  - `obj_to_target`: `0.245` (float) - distance from object to target
  - `grasp_success`: `0.0` (float) - no direct grasping required
  - `in_place_reward`: `0.086` (float) - reward for placement
  - `grasp_reward`: `0.001` (float) - minimal grasping reward

**Success condition**: The **object's position** must be within `TARGET_RADIUS` (0.05) of the goal position. The task requires using the stick to pull the object to the target location.

---

## Task 5: HandlePressSideV2GoalObservable

### Goal Position Reference
- **Goal attribute**: `[-0.2, 0.7, 0.14]`
- **Goal space**: `Box([-0.5, 0.4, 0.05], [0.5, 1., 0.5], (3,), float32)`
- **Observation last 3 values**: `[-0.06594279, 0.68192532, 0.07431072]` (within goal space)
- **Object positions**:
  - `obj_init_pos`: `[-0.26710838, 0.68034911, 0.00074964]` (handle)

**Conclusion**: The goal refers to the **handle's pressed position**. The handle needs to be pressed to the goal location.

### Success Criterion
- **Info dict metrics**:
  - `success`: `0.0`
  - `obj_to_target`: `0.096` (float) - distance from handle to target
  - `grasp_success`: `1.0` (float) - handle is grasped
  - `in_place_reward`: `0.100` (float) - reward for placement
  - `grasp_reward`: `0.102` (float) - grasping reward

**Success condition**: The **handle's position** must be within `TARGET_RADIUS` (0.02) of the goal position. The task requires:
1. Grasping the handle
2. Pressing it to the target location (goal position)

---

## Task 6: PushV2GoalObservable

### Goal Position Reference
- **Goal attribute**: `[0.1, 0.8, 0.02]`
- **Goal space**: `Box([-0.1, 0.8, 0.01], [0.1, 0.9, 0.02], (3,), float32)`
- **Observation last 3 values**: `[0.02011345, 0.8924358, 0.01987216]` (within goal space)
- **Object positions**:
  - `obj_init_pos`: `[0.06578325, 0.63034914, 0.01987216]`

**Conclusion**: The goal refers to the **object's target position**. The object needs to be pushed to the goal location.

### Success Criterion
- **Info dict metrics**:
  - `success`: `0.0`
  - `obj_to_target`: `0.261` (float) - distance from object to target
  - `grasp_success`: `0.0` (float) - no grasping required
  - `in_place_reward`: `0.145` (float) - reward for placement
  - `grasp_reward`: `0.026` (float) - minimal grasping reward

**Success condition**: The **object's position** must be within `TARGET_RADIUS` (0.05) of the goal position. The task requires pushing the object to the target location (no grasping needed).

---

## Task 7: ShelfPlaceV2GoalObservable

### Goal Position Reference
- **Goal attribute**: `[0., 0.85, 0.301]`
- **Goal space**: `Box([-0.1, 0.8, 0.299], [0.1, 0.9, 0.301], (3,), float32)`
- **Observation last 3 values**: `[0.02011345, 0.8924358, 0.30057573]` (within goal space)
- **Object positions**:
  - `obj_init_pos`: `[0.06578325, 0.53034913, 0.01993605]`

**Conclusion**: The goal refers to the **object's target position on the shelf**. The object needs to be placed on the shelf at the goal location.

### Success Criterion
- **Info dict metrics**:
  - `success`: `0.0`
  - `obj_to_target`: `0.457` (float) - distance from object to target
  - `grasp_success`: `0.0` (float) - no grasping shown in first step
  - `in_place_reward`: `0.123` (float) - reward for placement
  - `grasp_reward`: `0.0` (float) - no grasping reward

**Success condition**: The **object's position** must be within `TARGET_RADIUS` (0.05) of the goal position on the shelf. The task requires:
1. Grasping the object
2. Placing it on the shelf at the target location (goal position)

---

## Task 8: WindowCloseV2GoalObservable

### Goal Position Reference
- **Goal attribute**: Not explicitly shown, but `goal_space` exists
- **Goal space**: `Box([-0.5, 0.4, 0.05], [0.5, 1., 0.5], (3,), float32)`
- **Observation last 3 values**: `[0., 0.798146, 0.2]` (within goal space)
- **Object positions**:
  - `obj_init_pos`: `[0., 0.79552368, 0.2]` (window object)
  - `window_handle_pos_init`: `[0.11, 0.69, 0.202]` (window handle)

**Conclusion**: The goal refers to the **window's closed position**. The window needs to be closed to the goal location.

### Success Criterion
- **Info dict metrics**:
  - `success`: `0.0`
  - `obj_to_target`: `0.210` (float) - distance from window to target
  - `grasp_success`: `1.0` (float) - window handle is grasped
  - `in_place_reward`: `0.1` (float) - reward for placement
  - `grasp_reward`: `0.098` (float) - grasping reward

**Success condition**: The **window's position** must be within `TARGET_RADIUS` (0.05) of the goal position (closed position). The task requires:
1. Grasping the window handle
2. Closing the window to the target location (goal position)

---

## Task 9: PegUnplugSideV2GoalObservable

### Goal Position Reference
- **Goal attribute**: `[-0.225, 0.6, 0.]`
- **Goal space**: `Box([-0.056, 0.6, 0.13], [0.044, 0.8, 0.132], (3,), float32)`
- **Observation last 3 values**: `[0.01205727, 0.6641947, 0.13031072]` (within goal space)
- **Object positions**:
  - `obj_init_pos`: `[-0.08310838, 0.66069827, 0.13174964]` (peg)

**Conclusion**: The goal refers to the **peg's unplugged position**. The peg needs to be unplugged/pulled to the goal location.

### Success Criterion
- **Info dict metrics**:
  - `success`: `0.0`
  - `obj_to_target`: `0.110` (float) - distance from peg to target
  - `grasp_success`: `0.0` (float) - no grasping shown in first step
  - `in_place_reward`: `0.273` (float) - reward for placement
  - `grasp_reward`: `0.026` (float) - minimal grasping reward

**Success condition**: The **peg's position** must be within `TARGET_RADIUS` (0.05) of the goal position. The task requires:
1. Grasping the peg
2. Unplugging/pulling it to the target location (goal position)

---

## Summary

### Goal Position References by Task

| Task | Environment | Goal Refers To | Success Condition |
|------|-------------|----------------|-------------------|
| 0 | HammerV2 | **Nail position** | Nail within 0.05 of goal |
| 1 | PushWallV2 | **Object position** | Object within 0.05 of goal |
| 2 | FaucetCloseV2 | **Faucet handle position** | Handle within 0.05 of goal |
| 3 | PushBackV2 | **Object position** | Object within 0.05 of goal |
| 4 | StickPullV2 | **Object position** | Object within 0.05 of goal |
| 5 | HandlePressSideV2 | **Handle position** | Handle within 0.02 of goal |
| 6 | PushV2 | **Object position** | Object within 0.05 of goal |
| 7 | ShelfPlaceV2 | **Object position** | Object within 0.05 of goal |
| 8 | WindowCloseV2 | **Window position** | Window within 0.05 of goal |
| 9 | PegUnplugSideV2 | **Peg position** | Peg within 0.05 of goal |

### Key Patterns

1. **Most tasks (7/10)**: Goal refers to **object position** - the main manipulable object must be moved to the goal location
2. **Some tasks (2/10)**: Goal refers to **handle/mechanism position** - a handle or mechanism must be moved to the goal location
3. **One task (1/10)**: Goal refers to **nail position** - a nail must be driven to the goal location

4. **Success criteria**:
   - All tasks use `TARGET_RADIUS` (0.05) except HandlePressSideV2 (0.02)
   - Success is determined by `info['success']` being 1.0
   - `obj_to_target` metric indicates distance to goal (lower is better)
   - Most tasks require grasping (`grasp_success`), but some don't (push tasks)

5. **Common info dict structure**:
   - `success`: Binary indicator (0.0 or 1.0)
   - `obj_to_target`: Distance metric (float)
   - `grasp_success`: Whether object is grasped (bool/float)
   - `in_place_reward`: Reward for placement proximity
   - `grasp_reward`: Reward for grasping
   - `unscaled_reward`: Total reward before scaling

### Important Notes

1. **Goal extraction**: The goal coordinates are always in the last 3 dimensions of the observation (`observation[-3:]`), but the **target object varies by task**.

2. **Success determination**: Success is not just about reaching the goal position - it also requires:
   - Proper grasping (for tasks that require it)
   - Correct manipulation (e.g., hammering, pressing, closing)
   - Object being within the target radius

3. **Task-specific interpretation**: The same goal coordinate format is used across all tasks, but the **semantic meaning** (which object/entity) varies. You must know the task type to correctly interpret the goal.
