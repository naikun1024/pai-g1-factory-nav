"""俯瞰カメラの画素座標 ⇔ 床面のワールド座標 の変換(ピンホールカメラモデル)。"""
from __future__ import annotations

import mujoco
import numpy as np


class TopCamera:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name="top_cam", width=640, height=480):
        self.m, self.d = model, data
        self.cid = model.camera(name).id
        self.name = name
        self.W, self.H = width, height
        fovy = np.radians(model.cam_fovy[self.cid])
        self.f = (height / 2) / np.tan(fovy / 2)  # 焦点距離 [px]

    def _pose(self):
        mujoco.mj_forward(self.m, self.d)
        return self.d.cam_xpos[self.cid].copy(), self.d.cam_xmat[self.cid].reshape(3, 3).copy()

    def pixel_to_floor(self, u, v, z=0.0):
        """画素 (u, v) を通る光線と床面 z の交点を返す。"""
        p, R = self._pose()
        ray_cam = np.array([(u - self.W / 2) / self.f, -(v - self.H / 2) / self.f, -1.0])
        ray = R @ ray_cam
        t = (z - p[2]) / ray[2]
        return p + t * ray

    def floor_to_pixel(self, x, y, z=0.0):
        p, R = self._pose()
        c = R.T @ (np.array([x, y, z]) - p)
        return self.W / 2 + self.f * c[0] / -c[2], self.H / 2 - self.f * c[1] / -c[2]

    def norm_to_floor(self, yx_1000):
        """Gemini Robotics ER の [y, x](0〜1000 正規化)を床面座標へ。"""
        y, x = yx_1000
        return self.pixel_to_floor(x / 1000 * self.W, y / 1000 * self.H)[:2]

    def render(self, renderer: mujoco.Renderer):
        renderer.update_scene(self.d, camera=self.name)
        return renderer.render()
