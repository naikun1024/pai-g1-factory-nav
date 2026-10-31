"""G1 の学習済み歩行方策(Unitree 公式 unitree_rl_gym の motion.pt)を MuJoCo 上で推論するモジュール。

方策の入力(47次元)= 角速度・重力方向・速度指令(vx, vy, wz)・関節角・関節速度・前回行動・歩行位相
方策の出力(12次元)= 脚 12 関節の目標角(デフォルト姿勢からの差分)
→ PD 制御でトルクに変換して MuJoCo に入れる。
観測の作り方は公式の deploy/deploy_mujoco/deploy_mujoco.py と同じにしている(違うと方策が正しく動かない)。
"""
from __future__ import annotations

import warnings

import mujoco
import numpy as np
import torch
import yaml

from .scene import POLICY_CFG, POLICY_PATH


def gravity_orientation(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class G1Walker:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, frame_cb=None, fps=30):
        with open(POLICY_CFG) as f:
            c = yaml.safe_load(f)
        self.m, self.d = model, data
        self.dt = c["simulation_dt"]
        model.opt.timestep = self.dt
        self.decimation = c["control_decimation"]          # 0.002s × 10 = 50Hz で方策を回す
        self.kps = np.array(c["kps"], np.float32)
        self.kds = np.array(c["kds"], np.float32)
        self.default = np.array(c["default_angles"], np.float32)
        self.cmd_scale = np.array(c["cmd_scale"], np.float32)
        self.ang_vel_scale = c["ang_vel_scale"]
        self.dof_vel_scale = c["dof_vel_scale"]
        self.action_scale = c["action_scale"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.policy = torch.jit.load(POLICY_PATH)
        self.action = np.zeros(12, np.float32)
        self.target = self.default.copy()
        self.obs = np.zeros(c["num_obs"], np.float32)
        self.counter = 0
        self.cmd = np.zeros(3, np.float32)
        self.frame_cb = frame_cb
        self.steps_per_frame = int(round((1 / fps) / self.dt))
        self.pelvis = model.body("pelvis").id

    # ---- 状態 ----
    def pose(self):
        """(x, y, yaw)。実機なら自己位置推定(SLAM 等)で得る値。ここではシミュレータの真値を使う。"""
        return float(self.d.qpos[0]), float(self.d.qpos[1]), yaw_of(self.d.qpos[3:7])

    def fallen(self):
        return self.d.qpos[2] < 0.5

    # ---- 1 ステップ ----
    def step(self):
        d = self.d
        d.ctrl[:] = (self.target - d.qpos[7:]) * self.kps - d.qvel[6:] * self.kds
        mujoco.mj_step(self.m, d)
        self.counter += 1
        if self.counter % self.decimation == 0:
            phase = (self.counter * self.dt) % 0.8 / 0.8
            o = self.obs
            o[:3] = d.qvel[3:6] * self.ang_vel_scale
            o[3:6] = gravity_orientation(d.qpos[3:7])
            o[6:9] = self.cmd * self.cmd_scale
            o[9:21] = d.qpos[7:] - self.default
            o[21:33] = d.qvel[6:] * self.dof_vel_scale
            o[33:45] = self.action
            o[45:47] = (np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase))
            with torch.no_grad():
                self.action = self.policy(torch.from_numpy(o).unsqueeze(0)).numpy().squeeze()
            self.target = self.action * self.action_scale + self.default
        if self.frame_cb and self.counter % self.steps_per_frame == 0:
            self.frame_cb()

    def run(self, seconds, cmd=None):
        if cmd is not None:
            self.cmd[:] = cmd
        for _ in range(int(seconds / self.dt)):
            self.step()
