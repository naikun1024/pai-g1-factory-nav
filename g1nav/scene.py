"""工場フロアのシーン(G1・ステーション・障害物・カメラ)を生成するモジュール。"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G1_DIR = os.path.join(ROOT, "third_party", "unitree_rl_gym", "resources", "robots", "g1_description")
POLICY_PATH = os.path.join(ROOT, "third_party", "unitree_rl_gym", "deploy", "pre_train", "g1", "motion.pt")
POLICY_CFG = os.path.join(ROOT, "third_party", "unitree_rl_gym", "deploy", "deploy_mujoco", "configs", "g1.yaml")

# フロアの範囲 [m]
AREA = dict(x=(-0.8, 5.8), y=(-2.6, 2.6))

# ステーション:キー -> (日本語名, 英語名(AIへの指示用), 色)
STATIONS = {
    "parts": ("部品棚", "parts shelf", (0.15, 0.35, 0.90, 1)),
    "inspect": ("検査台", "inspection station", (0.10, 0.70, 0.25, 1)),
    "ship": ("出荷エリア", "shipping area", (0.95, 0.78, 0.10, 1)),
    "charge": ("充電ステーション", "charging station", (0.60, 0.20, 0.80, 1)),
}
MAT_HALF = 0.35          # ステーションのマット半辺 [m]
OBSTACLE_RGBA = (0.62, 0.40, 0.22, 1)  # パレット(茶色)


@dataclass
class Station:
    key: str
    xy: tuple

    @property
    def name_ja(self):
        return STATIONS[self.key][0]

    @property
    def name_en(self):
        return STATIONS[self.key][1]


@dataclass
class Obstacle:
    name: str
    xy: tuple
    half: tuple  # (hx, hy, hz)
    yaw: float


def random_layout(rng: np.random.Generator, n_obstacles=4):
    """ステーションと障害物(パレット)をランダムに配置する。"""
    slots = [(4.8, 1.8), (4.8, -1.8), (2.4, 2.0), (2.4, -2.0), (5.0, 0.0)]
    idx = rng.permutation(len(slots))[: len(STATIONS)]
    stations = [Station(k, slots[i]) for k, i in zip(STATIONS, idx)]
    obstacles = []
    tries = 0
    while len(obstacles) < n_obstacles and tries < 5000:
        tries += 1
        xy = (rng.uniform(1.0, 4.4), rng.uniform(-1.6, 1.6))
        half = (rng.uniform(0.25, 0.45), rng.uniform(0.25, 0.45), 0.25)
        r = max(half[:2]) * 1.42
        if np.hypot(*xy) < 1.0 + r:  # スタート地点の近くは空ける
            continue
        if any(np.hypot(xy[0] - s.xy[0], xy[1] - s.xy[1]) < r + MAT_HALF + 0.5 for s in stations):
            continue
        if any(np.hypot(xy[0] - o.xy[0], xy[1] - o.xy[1]) < r + max(o.half[:2]) * 1.42 + 0.8 for o in obstacles):
            continue
        obstacles.append(Obstacle(f"pallet{len(obstacles)}", xy, half, rng.uniform(0, np.pi)))
    return stations, obstacles


def lookat_xyaxes(pos, target, up=(0, 0, 1)):
    pos, target, up = map(np.asarray, (pos, target, up))
    fwd = (target - pos) / np.linalg.norm(target - pos)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return " ".join(f"{v:.4f}" for v in (*x, *y))


def build_xml(stations, obstacles) -> str:
    items = []
    for s in stations:
        rgba = " ".join(map(str, STATIONS[s.key][2]))
        # 床のマット(ぶつからない)+ 奥に同じ色のラック(障害物として扱う)
        items.append(
            f'<body name="st_{s.key}" pos="{s.xy[0]} {s.xy[1]} 0">'
            f'<geom type="box" size="{MAT_HALF} {MAT_HALF} 0.004" pos="0 0 0.004" rgba="{rgba}" contype="0" conaffinity="0"/>'
            f"</body>"
        )
    for o in obstacles:
        q = (np.cos(o.yaw / 2), 0, 0, np.sin(o.yaw / 2))
        rgba = " ".join(map(str, OBSTACLE_RGBA))
        items.append(
            f'<body name="{o.name}" pos="{o.xy[0]} {o.xy[1]} {o.half[2]}" quat="{q[0]} 0 0 {q[3]}">'
            f'<geom name="{o.name}_geom" type="box" size="{o.half[0]} {o.half[1]} {o.half[2]}" rgba="{rgba}"/>'
            f"</body>"
        )
    cx = (AREA["x"][0] + AREA["x"][1]) / 2
    return f"""
<mujoco model="g1_factory">
  <include file="g1_12dof.xml"/>
  <visual>
    <headlight diffuse="0.45 0.45 0.45" ambient="0.30 0.30 0.30" specular="0 0 0"/>
    <global azimuth="-140" elevation="-25" offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.80 0.84 0.90" rgb2="0.45 0.50 0.58" width="512" height="3072"/>
    <texture type="2d" name="floor" builtin="checker" rgb1="0.72 0.72 0.70" rgb2="0.66 0.66 0.64" width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="2 2" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="{cx} 0 6" dir="0 0 -1" directional="true" diffuse="0.45 0.45 0.45" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
    <!-- 俯瞰カメラ(AI 認識用):画像の右 = +x、上 = +y -->
    <camera name="top_cam" pos="{cx} 0 9.5" xyaxes="1 0 0 0 1 0" fovy="34"/>
    <camera name="overview" pos="-2.2 -3.8 3.6" xyaxes="{lookat_xyaxes((-2.2, -3.8, 3.6), (2.6, 0.0, 0.3))}" fovy="50"/>
    {''.join(items)}
  </worldbody>
</mujoco>
"""


def write_scene(stations, obstacles, filename="_factory_scene.xml") -> str:
    """g1_12dof.xml と同じフォルダに書き出す(メッシュの相対パスを解決するため)。"""
    if not os.path.exists(os.path.join(G1_DIR, "g1_12dof.xml")):
        raise FileNotFoundError(f"G1 モデルが見つかりません: {G1_DIR}\n先に `bash scripts/get_assets.sh` を実行してください。")
    path = os.path.join(G1_DIR, filename)
    with open(path, "w") as f:
        f.write(build_xml(stations, obstacles))
    return path
