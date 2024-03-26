import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os

import mujoco


class InvertedPendulumEnv(gym.Env):

    def __init__(
        self,
        max_steps=1024,
        record_video=False,
        run_name=None,
        record_video_from=0,
        set_target=False,
        set_target_step=70000,
        x_coordinate=None,
        randomize_mass=False,
        set_mass=0,
    ):
        self.init_qpos = np.zeros(2)
        self.init_qvel = np.zeros(2)
        self.max_steps = max_steps
        self.record_video = record_video
        self.set_target = set_target
        self.randomize_mass = randomize_mass
        self.set_mass = set_mass
        self.step_count = 0
        self.curr_step = 0
        self.rewards_raw = np.zeros([self.max_steps])

        self.model = mujoco.MjModel.from_xml_path("./mjcf.xml")
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 240, 320)

        if not self.set_target:
            self.observation_space = spaces.Dict(
                {
                    "qpos": spaces.Box(
                        low=np.array([-1.0, -1.0, -1.0]),
                        high=np.array([1.0, 1.0, 1.0]),
                        shape=(3,),
                        dtype=np.float32,
                    ),
                    "qvel": spaces.Box(
                        low=np.array([-np.inf, -np.inf]),
                        high=np.array([np.inf, np.inf]),
                        shape=(2,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.target_pos = None
            self.target_appears = set_target_step
            self.x_coordinate = x_coordinate
            self.observation_space = spaces.Dict(
                {
                    "qpos": spaces.Box(
                        low=np.array([-1.0, -1.0, -1.0]),
                        high=np.array([1.0, 1.0, 1.0]),
                        shape=(3,),
                        dtype=np.float32,
                    ),
                    "qvel": spaces.Box(
                        low=np.array([-np.inf, -np.inf]),
                        high=np.array([np.inf, np.inf]),
                        shape=(2,),
                        dtype=np.float32,
                    ),
                    "target": spaces.Box(
                        low=np.array([-1.0, -1.0]),
                        high=np.array([1.0, 1.0]),
                        shape=(2,),
                        dtype=np.float32,
                    ),
                }
            )

        self.action_space = spaces.Box(
            low=np.array([-3.0]),
            high=np.array([3.0]),
            shape=(1,),
            dtype=np.float32,
        )

        if record_video:
            self.video_dir = f"./runs/{run_name}/video/"
            self.frame_dir = self.video_dir + "img/"
            self.curr_eval_img_id = 0
            self.captured_frame = 5
            self.record_video_from = record_video_from
            if not os.path.exists(self.video_dir):
                os.mkdir(self.video_dir)
            if not os.path.exists(self.frame_dir):
                os.mkdir(self.frame_dir)

    def step(self, a):
        self.data.ctrl = a[0]
        mujoco.mj_step(self.model, self.data)

        self.renderer.update_scene(self.data, "fixed")

        reward = np.sin(-self.data.qpos[1] + np.pi / 2)

        if not self.set_target:
            observation = {
                "qpos": np.array(
                    [
                        self.data.qpos[0],
                        np.sin(self.data.qpos[1]),
                        np.cos(self.data.qpos[1]),
                    ],
                    dtype=np.float32,
                ),
                "qvel": self.data.qvel.astype(np.float32),
            }
        else:
            if self.step_count > self.target_appears:
                self.draw_ball()
                target_reward = -np.linalg.norm(
                    self.renderer.scene.geoms[1].pos - self.renderer.scene.geoms[2].pos
                )
                reward += target_reward
            observation = {
                "qpos": np.array(
                    [
                        self.data.qpos[0],
                        np.sin(self.data.qpos[1]),
                        np.cos(self.data.qpos[1]),
                    ],
                    dtype=np.float32,
                ),
                "qvel": self.data.qvel.astype(np.float32),
                "target": np.array(
                    [self.target_pos[0], self.target_pos[2]], dtype=np.float32
                ),
            }

        if self.record_video:
            if (
                self.step_count > self.record_video_from
                and self.step_count % self.captured_frame == 0
            ):
                self.save_frame()

        terminated = False

        if self.curr_step == self.max_steps - 1:
            terminated = True

        info = {}
        self.rewards_raw[self.curr_step] = reward

        self.step_count += 1
        self.curr_step += 1

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos = self.init_qpos
        self.data.qvel = self.init_qvel
        self.data.qpos[1] = np.pi  # Set the pole to be facing down
        
        if self.randomize_mass:
            self.model.body_mass[self.model.geom('cpole').id] = np.random.uniform(low = 3.0, high = 8.0)
            
        if self.set_mass > 0:
            self.model.body_mass[self.model.geom('cpole').id] = self.set_mass

        self.renderer.update_scene(self.data, "fixed")

        if not self.set_target:
            observation = {
                "qpos": np.array(
                    [
                        self.data.qpos[0],
                        np.sin(self.data.qpos[1]),
                        np.cos(self.data.qpos[1]),
                    ],
                    dtype=np.float32,
                ),
                "qvel": self.data.qvel.astype(np.float32),
            }
        else:
            if self.x_coordinate is not None:
                self.target_pos = [self.x_coordinate, 0, 0.6]
            else:
                self.target_pos = [np.random.rand() - 0.5, 0, 0.6]
            self.draw_ball()
            observation = {
                "qpos": np.array(
                    [
                        self.data.qpos[0],
                        np.sin(self.data.qpos[1]),
                        np.cos(self.data.qpos[1]),
                    ],
                    dtype=np.float32,
                ),
                "qvel": self.data.qvel.astype(np.float32),
                "target": np.array(
                    [self.target_pos[0], self.target_pos[2]], dtype=np.float32
                ),
            }

        info = {}
        self.curr_step = 0

        if self.record_video and self.step_count > self.record_video_from:
            os.system(
                f"ffmpeg -y -hide_banner -loglevel error -framerate 20 -i {self.frame_dir}%05d.jpg -c:v libx264 -pix_fmt yuv420p {self.video_dir}/run_{self.step_count}.mp4"
            )

            self.curr_eval_img_id = 0
            os.system(f"rm -r {self.frame_dir}*.jpg 2>/dev/null")

        return observation, info

    def save_frame(self):
        frame = self.renderer.render()
        img_file = self.frame_dir + f"{self.curr_eval_img_id:05d}.jpg"
        cv2.imwrite(img_file, frame)
        self.curr_eval_img_id += 1

    def draw_ball(self):
        self.renderer.scene.ngeom += 1
        mujoco.mjv_initGeom(
            self.renderer.scene.geoms[self.renderer.scene.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.03, 0, 0],
            pos=np.array(self.target_pos),
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 0, 0, 1]),
        )
