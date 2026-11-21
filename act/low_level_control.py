from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
import numpy as np
import time
import dm_env
from hardware_constant import FOLLOWER_PORT, FOLLOWER_ID, CONTROL_DT, CAMERA_CONFIGS, MOTOR_NAMES, RESET_POSE
from hardware_utils import disconnect_arm

ROBOT_PORT = FOLLOWER_PORT
ROBOT_ID = FOLLOWER_ID
RESET_POSE = RESET_POSE
DT = CONTROL_DT
MAX_RELATIVE_TARGET = 5.0  # Per-command cap: degrees for arm, 0-100 units for gripper.
CAMERAS = CAMERA_CONFIGS
#TODO:
TASK_CONFIGS = {
    "so101_pick_place": {
        "dataset_dir": "/workspace/data/pick_place_marker",
        "num_episodes": 100,  # Set to your approved, consecutively numbered count.
        "episode_len": 300,  # Evaluation steps; does not truncate training data.
        "camera_names": ["wrist", "front"],
        "state_dim": 6,
    },
}


class RealEnv:
    # Create the real robot
    def __init__(self, port, robot_id, cameras=None, dt=CONTROL_DT, reset_pose=None, max_relative_target=None):
        self.dt = dt
        self.reset_pose = reset_pose
        self._next_observation_at = None

        # SO-ARM configuration
        config = SO101FollowerConfig(
            port = port,
            id = robot_id,
            cameras = cameras or {},
            max_relative_target = max_relative_target,
        )
        
        self.robot = SO101Follower(config)
        try:
            self.robot.connect()
        except BaseException:
            disconnect_arm(self.robot)
            raise

    
    def get_qpos(self, raw_obs):
        qpos = np.array(
            [
                raw_obs[f"{name}.pos"]
                for name in MOTOR_NAMES
            ],
            dtype=np.float32
        )

        return qpos

    def get_images(self, raw_obs):

        images = {}

        for camera_name in self.robot.cameras:
            images[camera_name] = raw_obs[camera_name]

        return images
    
    def get_observation(self):
        """
        Read SO-ARM observation exactly once and convert it into format expected by ACT
        """

        raw_obs = self.robot.get_observation()

        obs = {
            "qpos": self.get_qpos(raw_obs),
            "images": self.get_images(raw_obs),
        }

        return obs

    # Action
    def _vector_to_action_dict(self, action):
        """
        Convert np.ndarray(6,) into 
        { 
            "shoulder_pan.pos": ...,
            ...
            "gripper.pos":....
        }
        """
        action = np.asarray(
            action,
            dtype = np.float32
        )

        if action.shape != (6,):
            raise ValueError(
                "SO-ARM101 action must have "
                f"shape (6,), got {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise ValueError("SO-ARM101 action contains NaN or infinity")
        
        robot_action = {
            f"{name}.pos": float(action[i])
            for i, name in enumerate(MOTOR_NAMES)
        }

        return robot_action

    # Reset
    def _move_to_pose(self, target_pose, move_time=1.5):
        "Smoothly interpolate from current position to target position."

        target_pose = np.asarray(target_pose,dtype=np.float32)
        if target_pose.shape != (6,) or not np.all(np.isfinite(target_pose)):
            raise ValueError(
                "reset pose must contain six finite values"
            )
        
        raw_obs = self.robot.get_observation()

        start_pos = self.get_qpos(raw_obs)

        num_steps = max(1, int(move_time/self.dt))

        for i in range(1, num_steps + 1):
            alpha = i / num_steps

            pose = (1.0 - alpha) * start_pos + alpha * target_pose

            command = self._vector_to_action_dict(pose)

            self.robot.send_action(command)

            time.sleep(self.dt)
        
    def _reset_joints(self):
        # Move robot to configured reset pose
        if self.reset_pose is None:
            raise RuntimeError(
                "A six-joint reset_pose is required before a real-robot rollout. "
                "Refusing to start from an unknown pose."
            )
        
        self._move_to_pose(self.reset_pose)

    # Reward to evaluate
    def get_reward(self):
        # No automatic real-world task reward yet.
        # None explicitly means "not measured"; returning 0 would make every
        # rollout look successful when env_max_reward is also zero.
        return None
    
    def reset(self, fake=False):
        """
        ACT calls this at the beginning
        of an episode.

        fake=True:
            do not physically move robot.

        fake=False:
            move robot to reset_pose if provided.
        """

        if not fake:
            self._reset_joints()

        # Mốc observation tiếp theo, dùng để tính phần thời gian còn lại của DT.
        observation_start = time.perf_counter()
        observation = self.get_observation()
        self._next_observation_at = observation_start + self.dt
        # container TimeStep
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=observation,
        )
    
    def step(self, action):
        # send one 6D ACT action to SO-ARM101
        robot_action = self._vector_to_action_dict(action)
        self.robot.send_action(robot_action)
        # Chỉ nghỉ phần còn lại của chu kỳ DT, có tính cả inference và motor I/O.
        # Nghỉ trước rồi mới đọc observation để policy nhận ảnh/state mới.
        deadline = self._next_observation_at
        if deadline is None:
            deadline = time.perf_counter() + self.dt
        time.sleep(max(0.0, deadline - time.perf_counter()))
        observation_start = time.perf_counter()
        observation = self.get_observation()
        # Nếu xử lý quá DT, bắt đầu chu kỳ mới; không chạy dồn để bù thời gian.
        self._next_observation_at = max(deadline, observation_start) + self.dt

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=observation,
        )

    def close(self):
        disconnect_arm(self.robot)


def make_real_env(port, robot_id, cameras=None, dt=CONTROL_DT, reset_pose=None, max_relative_target=None):
    env = RealEnv(
        port=port,
        robot_id=robot_id,
        cameras=cameras,
        dt=dt,
        reset_pose=reset_pose,
        max_relative_target=max_relative_target,
    )

    return env
