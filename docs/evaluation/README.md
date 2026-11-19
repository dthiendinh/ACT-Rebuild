# Real-robot evaluation

[← Project overview](../../README.md) · [Training and rollout commands](../training_guide/README.md#9-evaluate-on-the-real-follower)

This protocol supports the README comparison of ACT **with temporal aggregation** and **without temporal aggregation**. Results and videos are pending; the tables are templates for future measurements.

## Matched trials

1. Select one checkpoint and record its filename, code revision, `training_config.json`, and `dataset_stats.pkl`. The current evaluation CLI loads `policy_last.ckpt`.
2. Define a fixed list of marker/cup layouts. Label familiar collection layouts and new test layouts separately; report their results separately.
3. Keep the reset pose, cameras, lighting, chunk size, step limit, and `max_relative_target` identical across modes. Record the actual values.
4. Run each layout in both modes, resetting the scene before each attempt. Alternate which mode runs first. Set the number of repeats before testing and count every started trial, including failures and interventions.
5. Use the [rollout command](../training_guide/README.md#9-evaluate-on-the-real-follower) with `--temporal_agg` for temporal mode and omit that flag for no temporal mode. The CLI does not automatically reset the objects between rollouts; single-rollout invocations make manual scene resets explicit.
6. Score the outcome manually and record video externally. Copy the runner's `result_policy_last.txt` after each invocation because the next invocation overwrites it. That file records rollout completion, not manipulation success.

The no temporal mode executes an entire chunk before querying again. Temporal mode queries every step and averages overlapping predictions. Because both query frequency and averaging change, this protocol compares complete rollout modes.

## Success criteria and measurements

| Measurement | Definition |
| --- | --- |
| Grasp success | Marker is grasped and lifted clear of the surface |
| Task success | Marker is released inside the cup within the step limit, without intervention |
| Task success rate | Successful tasks / all started trials × 100 |
| Grasp success rate | Successful grasps / all started trials × 100 |
| Completion time | Wall-clock seconds from the first policy command after reset to successful release; summarize successful trials and report their count |
| Control rate | Executed policy control steps / rollout wall-clock duration, excluding reset; record a measured rate, rather than assuming the 30 Hz target |
| Failure mode | Missed reach, failed grasp, dropped marker, missed cup, timeout, or manual intervention |

The runner does not currently export per-step timing or outcome annotations. Obtain these from an external recording/timing measurement and state the method; leave unmeasured fields pending.

## Trial log template

Duplicate rows for every layout and repeat. Record checkpoint, chunk size, reset pose, step limit, target-change limit, hardware, and camera configuration with the log.

| Trial | Layout / familiar or new | Mode | Grasp success | Task success | Completion (s) | Control (Hz) | Failure / intervention | Video |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pending | Pending | No temporal | Pending | Pending | Pending | Pending | Pending | Pending |
| Pending | Pending | Temporal | Pending | Pending | Pending | Pending | Pending | Pending |

Summarize both modes in the main README using success counts and denominators. For completion time and control rate, state the statistic used and number of measured trials. Preserve failed trials and label representative videos with their trial IDs.

## Add figures and videos

Create the following asset directories when the files are available. These are planned paths, not existing results:

```text
docs/
├── assets/
│   ├── metrics/
│   │   ├── epoch_loss.png
│   │   ├── epoch_l1.png
│   │   ├── epoch_kl.png
│   │   └── train_step.png
│   └── rollouts/
│       ├── no_temporal.mp4
│       ├── no_temporal.jpg
│       ├── temporal.mp4
│       └── temporal.jpg
└── evaluation/
    ├── README.md
    └── pick_place/                 # Add the selected run's configuration and trial log
```

The main README already contains commented image and video-thumbnail markup. After adding each asset, remove its placeholder text and uncomment the corresponding markup. Clickable thumbnails let GitHub readers open the MP4. A hosted video URL can replace the MP4 path without changing the layout.

For each video, add a caption with checkpoint, mode, layout, trial ID, outcome, and playback speed. For TensorBoard figures, retain the raw trace, axis units, and smoothing setting. Add links to the exact run configuration and trial log once those files exist.
