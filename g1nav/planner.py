"""タスク理解(どこへ・どの順で行くか)と経路計画。

役割分担(この作品の設計上のポイント):
  - Gemini Robotics ER 2(VLM)… 俯瞰画像 + 日本語指示 から「目的地の順番と位置」「障害物の位置」を認識する
  - A*(古典的な経路計画)… VLM が見つけた障害物を避ける安全な経路を計算する
  → 意味理解は AI、安全性は検証可能なアルゴリズム、という分担にしている。

--planner mock は API キーなしで動作確認するためのモード(シミュレータの真値 + キーワード照合)。
"""
from __future__ import annotations

import heapq
import io
import json
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .scene import AREA, STATIONS

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-robotics-er-2-preview")


@dataclass
class Goal:
    label: str          # AI が付けたラベル
    xy: np.ndarray      # 床面座標 [m]
    point: tuple | None = None  # 画像上の点 [y, x](0-1000)


@dataclass
class Plan:
    goals: list
    obstacles: list = field(default_factory=list)  # 各要素: (xmin, ymin, xmax, ymax) [m]
    obstacle_boxes_px: list = field(default_factory=list)  # 表示用 [ymin,xmin,ymax,xmax](0-1000)
    raw: str = ""
    latency_s: float = 0.0
    source: str = ""


# ---------------------------------------------------------------- Gemini
PROMPT = """You are the task planner of a humanoid robot (Unitree G1) in a factory.
The image is a top-down view of the factory floor. Colored square mats are stations:
blue = parts shelf, green = inspection station, yellow = shipping area, purple = charging station.
Brown boxes are pallets (obstacles). The robot is the small gray figure near the left edge.

Operator instruction (Japanese): "{instruction}"

1. Decide which stations the robot must visit, in order, to follow the instruction.
2. Point to the CENTER of each of those station mats.
3. Detect every brown pallet with a 2D bounding box.

Return ONLY JSON in this format:
{{"goals": [{{"label": "<station name>", "point": [y, x]}}, ...],
  "obstacles": [{{"label": "pallet", "box_2d": [ymin, xmin, ymax, xmax]}}, ...]}}
Points and boxes are normalized to 0-1000, with [y, x] order.
"""


def _parse_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=0)
    return json.loads(text[start:])


def plan_with_gemini(image: np.ndarray, instruction: str, cam, thinking_level="low") -> Plan:
    from google import genai
    from google.genai import types

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("環境変数 GEMINI_API_KEY が設定されていません。")
    client = genai.Client()
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    contents = [types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
                PROMPT.format(instruction=instruction)]
    t0 = time.time()
    try:
        cfg = types.GenerateContentConfig(temperature=0.2,
                                          thinking_config=types.ThinkingConfig(thinking_level=thinking_level))
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=cfg)
    except Exception as e:  # thinking_level 非対応などの場合は設定なしで再試行
        print(f"  [info] 設定付き呼び出しに失敗したため再試行します: {e}")
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents)
    latency = time.time() - t0
    data = _parse_json(resp.text)
    goals = [Goal(g["label"], cam.norm_to_floor(g["point"]), tuple(g["point"])) for g in data.get("goals", [])]
    obstacles, boxes = [], []
    for o in data.get("obstacles", []):
        ymin, xmin, ymax, xmax = o["box_2d"]
        p1 = cam.norm_to_floor((ymin, xmin))
        p2 = cam.norm_to_floor((ymax, xmax))
        obstacles.append((min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1])))
        boxes.append(o["box_2d"])
    return Plan(goals, obstacles, boxes, raw=resp.text, latency_s=latency, source=GEMINI_MODEL)


# ---------------------------------------------------------------- Mock(API なしの動作確認用)
KEYWORDS = {
    "parts": ["部品棚", "部品", "棚", "parts"],
    "inspect": ["検査", "inspection"],
    "ship": ["出荷", "shipping"],
    "charge": ["充電", "charg"],
}


def plan_mock(instruction: str, stations, obstacles) -> Plan:
    """指示文の中で各ステーション名が出てくる順に訪問する(真値を使う)。"""
    found = []
    for key, words in KEYWORDS.items():
        pos = [instruction.find(w) for w in words if instruction.find(w) >= 0]
        if pos:
            found.append((min(pos), key))
    found.sort()
    by_key = {s.key: s for s in stations}
    goals = [Goal(STATIONS[k][1], np.array(by_key[k].xy)) for _, k in found]
    obs = []
    for o in obstacles:
        c, s = np.cos(o.yaw), np.sin(o.yaw)
        corners = np.array([[sx * o.half[0], sy * o.half[1]] for sx in (-1, 1) for sy in (-1, 1)])
        rot = corners @ np.array([[c, s], [-s, c]]) + np.array(o.xy)
        obs.append((*rot.min(0), *rot.max(0)))
    return Plan(goals, obs, source="mock(真値)")


# ---------------------------------------------------------------- A* 経路計画
class GridPlanner:
    def __init__(self, obstacles, res=0.05, inflate=0.40):
        """obstacles: (xmin, ymin, xmax, ymax) のリスト。inflate はロボットの半径 + 余裕 [m]。"""
        self.res = res
        self.x0, self.y0 = AREA["x"][0], AREA["y"][0]
        self.nx = int((AREA["x"][1] - self.x0) / res)
        self.ny = int((AREA["y"][1] - self.y0) / res)
        self.occ = np.zeros((self.nx, self.ny), bool)
        xs = self.x0 + (np.arange(self.nx) + 0.5) * res
        ys = self.y0 + (np.arange(self.ny) + 0.5) * res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        for xmin, ymin, xmax, ymax in obstacles:
            dx = np.maximum(np.maximum(xmin - X, X - xmax), 0)
            dy = np.maximum(np.maximum(ymin - Y, Y - ymax), 0)
            self.occ |= np.hypot(dx, dy) < inflate

    def to_cell(self, xy):
        return (int(np.clip((xy[0] - self.x0) / self.res, 0, self.nx - 1)),
                int(np.clip((xy[1] - self.y0) / self.res, 0, self.ny - 1)))

    def to_xy(self, c):
        return np.array([self.x0 + (c[0] + 0.5) * self.res, self.y0 + (c[1] + 0.5) * self.res])

    def free(self, c):
        return 0 <= c[0] < self.nx and 0 <= c[1] < self.ny and not self.occ[c]

    def _nearest_free(self, c):
        if self.free(c):
            return c
        for r in range(1, 40):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    cc = (c[0] + dx, c[1] + dy)
                    if self.free(cc):
                        return cc
        return c

    def plan(self, start_xy, goal_xy):
        s, g = self._nearest_free(self.to_cell(start_xy)), self._nearest_free(self.to_cell(goal_xy))
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        openq = [(0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        while openq:
            _, c = heapq.heappop(openq)
            if c == g:
                break
            for dx, dy in moves:
                n = (c[0] + dx, c[1] + dy)
                if not self.free(n):
                    continue
                nc = cost[c] + np.hypot(dx, dy)
                if nc < cost.get(n, 1e18):
                    cost[n], came[n] = nc, c
                    heapq.heappush(openq, (nc + np.hypot(g[0] - n[0], g[1] - n[1]), n))
        if g not in came:
            return None
        path, c = [], g
        while c is not None:
            path.append(c)
            c = came[c]
        path = path[::-1]
        return self._shortcut([self.to_xy(c) for c in path], goal_xy)

    def _line_free(self, a, b):
        n = int(np.linalg.norm(b - a) / (self.res / 2)) + 1
        return all(self.free(self.to_cell(a + (b - a) * t)) for t in np.linspace(0, 1, n))

    def _shortcut(self, pts, goal_xy):
        """見通しの良い点同士をつないで、ウェイポイント数を減らす。"""
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._line_free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        out[-1] = np.asarray(goal_xy, float) if self.free(self.to_cell(goal_xy)) else out[-1]
        return out[1:]
