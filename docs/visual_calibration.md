# Visual calibration

SyncWorld conditions on a short **calibration sweep**: a clip of the arm exercising each
end-effector DoF, filmed from the same camera as the episode. The model reads it to work out what
"+x" or "+yaw" *look like* in this particular view, instead of having one action frame baked into
the weights. That is what lets a single checkpoint be evaluated across RoboCasa, ManiSkill, LIBERO
and a real arm, whose camera placements and action conventions all differ.

One clip sits next to each episode:

```
episode_0/camera_poses_0/
├── expert/               # the episode itself
└── calibration/
    ├── <view>/video.mp4  # the sweep, one video per camera view
    ├── pose.pkl          # per-frame ground-truth EE pose + gripper state
    └── move_range.pkl    # which direction each DoF was swept
```

## 1. Open the gripper off-camera

The gripper is opened during an **unrecorded** warmup, so the sweep never shows it moving. A
calibration clip is about the arm's coordinate frame, not about grasping, and a visible open/close
would be a spurious cue.

```python
def base_template(grip=gripper_open_cmd):
    a = np.asarray(template_action, dtype=np.float32).copy()
    a[es:es + 6] = 0.0                 # zero EE delta
    a[gripper_slice] = grip            # explicit gripper command (negative = open)
    return a

# ---- UNRECORDED warmup: open the gripper + let physics settle ----
for _ in range(int(warmup_steps)):
    env.step(base_template())
```

## 2. Randomize the sweep

Every DoF gets a random sign and magnitude, and the three translation axes are swept in random
order. This is deliberate: an identical sweep every time could be memorized, so randomizing forces
the model to actually read the motion it observes.

```python
x_sign, y_sign, z_sign = (1 if rng_py.random() < 0.5 else -1 for _ in range(3))
roll_sign  = 1 if rng_py.random() < 0.5 else -1
pitch_sign = 1 if rng_py.random() < 0.5 else -1
yaw_sign   = 1 if rng_py.random() < 0.5 else -1

x_mv = x_sign * rng_py.uniform(0.15, 0.25) * magnitude_scale      # metres
y_mv = y_sign * rng_py.uniform(0.15, 0.25) * magnitude_scale
z_mv = z_sign * rng_py.uniform(0.15, 0.25) * magnitude_scale
roll_ang  = roll_sign  * rng_py.uniform(5, 10)  * magnitude_scale  # degrees
pitch_ang = pitch_sign * rng_py.uniform(10, 15) * magnitude_scale
yaw_ang   = yaw_sign   * rng_py.uniform(10, 15) * magnitude_scale

pos_moves = [("X", np.array([x_mv, 0, 0])), ("Y", np.array([0, y_mv, 0])),
             ("Z", np.array([0, 0, z_mv]))]
rng_py.shuffle(pos_moves)                    # translation order is randomized
```

Rotations are always swept Pitch → Yaw → Roll; only the translation order is shuffled.

Because the signs are random the clip alone is ambiguous — a sweep along `-x` looks like `+x` played
backwards — so the realized directions are saved to `move_range.pkl` as
`movement_order`, e.g. `["Z-", "X+", "Y-", "Pitch+", "Yaw-", "Roll+"]`.

## 3. Sweep each DoF out and back

Each axis moves out, pauses, returns, pauses. The pauses leave a clean boundary between axes, which
is what lets the consumer find one contiguous run of monotone motion per DoF.

```python
for name, vec in pos_moves:
    a = base_template(); a[es:es + 3] = vec
    for _ in range(pos_steps):
        step_record(a)                       # move out
    for _ in range(pause):
        step_record(base_template())         # hold
    a = base_template(); a[es:es + 3] = -vec
    for _ in range(pos_steps):
        step_record(a)                       # move back
    for _ in range(pause):
        step_record(base_template())
```

`step_record` advances the sim and, every `stride` steps, captures one frame per camera view
alongside the ground-truth pose:

```python
def step_record(a):
    env.step(np.asarray(a))
    prev_open[0] = _compute_gripper_open_state_from_action(a, gripper_slice, prev_open[0])
    if step_idx[0] % stride == 0:
        for v, cam in enumerate(agent_cams):
            agent_frames[v].append(_render(env, cam, H, W))
        p7, m = capture_ee_pose(env, esid)   # site_xpos / site_xmat -> 7-vec + 4x4
        poses7.append(p7); mats.append(m); gopen.append(prev_open[0])
    step_idx[0] += 1
```

That produces `pose.pkl`:

| key              | shape       | meaning                                                          |
| ---------------- | ----------- | ---------------------------------------------------------------- |
| `gripper_pose`   | `(T, 7)`    | EE position + quaternion `[x,y,z, qx,qy,qz,qw]`                  |
| `gripper_matrix` | `(T, 4, 4)` | the same pose as a homogeneous transform                         |
| `gripper_open`   | `(T,)`      | 1.0 open / 0.0 closed, from the commanded action with hysteresis |
| `save_stride`    | scalar      | sim steps per recorded frame                                     |

## How the model consumes it

[`calib_segments.py`](../cosmos_framework/data/vfm/action/calib_segments.py) turns the clip into K
conditioning items. It does **not** rely on the recorded ordering: it re-detects each axis from the
pose trajectory by finding the best contiguous run of monotone motion along it, and uses
`movement_order` only for the sign.

```python
seg_lists = build_calib_segment_indices(
    mats,                    # (T,4,4) calibration poses
    args.calib_seg_len,      # frames per segment
    efficient,               # True -> 6 segments (one per DoF); False -> 12 (both signs)
    move_order,              # from move_range.pkl; None -> assume positive
)
```

Each index list slices the video and poses into one vision item plus its paired action block. The K
items are prepended to the sample, ahead of the history and the current+future clip, as
fully-conditioning context. They are emitted in a canonical order (x, y, z, yaw, pitch, roll)
regardless of the order they were recorded in, so the model always sees the DoFs in the same slots.

`move_range.pkl` is optional — without it the segmenter assumes a positive sweep per axis, which
costs accuracy on datasets that sweep some axes negative. The published ManiSkill and LIBERO sets
have none; the real set does.

See [Evaluation](../README.md#2-prepare-the-evaluation-set) for the on-disk layout, and
`--calib-null` for evaluating without calibration at all.
