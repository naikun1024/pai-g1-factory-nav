"""ウェイポイント追従:目標点への方位誤差から速度指令 (vx, vy, wz) を作り、歩行方策に渡す。"""
from __future__ import annotations

import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class WaypointFollower:
    def __init__(self, walker, v_max=0.5, w_max=0.6, k_w=1.5, tol=0.25):
        self.w = walker
        self.v_max, self.w_max, self.k_w, self.tol = v_max, w_max, k_w, tol
        self.trace = []  # 実際に歩いた軌跡(評価・描画用)

    def command(self, target):
        x, y, yaw = self.w.pose()
        dx, dy = target[0] - x, target[1] - y
        dist = np.hypot(dx, dy)
        err = wrap(np.arctan2(dy, dx) - yaw)
        wz = float(np.clip(self.k_w * err, -self.w_max, self.w_max))
        # 向きが大きくずれているときはその場で旋回、合ってきたら前進(近づいたら減速)
        vx = 0.0 if abs(err) > 0.7 else self.v_max * np.cos(err) * min(1.0, dist / 0.6 + 0.3)
        return np.array([vx, 0.0, wz], np.float32), dist

    def go(self, waypoints, timeout=60.0, on_tick=None):
        """ウェイポイント列をたどる。戻り値: 'arrived' / 'fallen' / 'timeout'"""
        t_end = self.w.d.time + timeout
        for i, wp in enumerate(waypoints):
            last = i == len(waypoints) - 1
            tol = self.tol if last else 0.35
            while True:
                cmd, dist = self.command(wp)
                if dist < tol:
                    break
                self.w.cmd[:] = cmd
                for _ in range(self.w.decimation):
                    self.w.step()
                self.trace.append(self.w.pose()[:2])
                if on_tick:
                    on_tick()
                if self.w.fallen():
                    return "fallen"
                if self.w.d.time > t_end:
                    return "timeout"
        self.stop(0.8)
        return "arrived"

    def stop(self, seconds):
        self.w.run(seconds, cmd=(0.0, 0.0, 0.0))
