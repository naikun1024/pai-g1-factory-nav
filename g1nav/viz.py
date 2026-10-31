"""動画用の描画(テロップ・タイトルカード・俯瞰図への計画の重ね描き)。"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",   # macOS
    "/System/Library/Fonts/Hiragino Sans GB.ttc",        # macOS
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
    "C:/Windows/Fonts/meiryo.ttc",
]


def font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    print("[warn] 日本語フォントが見つかりません。文字化けする場合は FONT_CANDIDATES にパスを追加してください。")
    return ImageFont.load_default()


COLORS = [(255, 80, 60), (60, 200, 255), (255, 210, 0), (190, 90, 255)]


def card(W, H, title, lines=(), sub=None, bg=(18, 22, 30)):
    """タイトルカード(文字だけの画面)を作る。"""
    img = Image.new("RGB", (W, H), bg)
    dr = ImageDraw.Draw(img)
    s = H / 720
    dr.text((W * 0.07, H * 0.18), title, font=font(int(46 * s)), fill=(255, 255, 255))
    if sub:
        dr.text((W * 0.07, H * 0.18 + 70 * s), sub, font=font(int(26 * s)), fill=(150, 200, 255))
    y = H * 0.18 + 140 * s
    for ln in lines:
        dr.text((W * 0.08, y), ln, font=font(int(28 * s)), fill=(225, 225, 225))
        y += 48 * s
    return np.array(img)


def draw_plan(top_img, cam, plan, paths=None, trace=None, robot_xy=None, active=None):
    """俯瞰画像に AI の認識結果(目的地・障害物)と計画経路・実軌跡を重ねる。"""
    img = Image.fromarray(top_img).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    sx, sy = W / cam.W, H / cam.H
    f = font(max(12, int(16 * W / 640)))
    # 障害物(AI が検出した箱)
    for box in plan.obstacle_boxes_px:
        ymin, xmin, ymax, xmax = box
        dr.rectangle([xmin / 1000 * W, ymin / 1000 * H, xmax / 1000 * W, ymax / 1000 * H], outline=(255, 60, 60, 255), width=3)
    if not plan.obstacle_boxes_px:  # mock の場合は床座標の箱を描く
        for xmin, ymin, xmax, ymax in plan.obstacles:
            u1, v1 = cam.floor_to_pixel(xmin, ymax)
            u2, v2 = cam.floor_to_pixel(xmax, ymin)
            dr.rectangle([u1 * sx, v1 * sy, u2 * sx, v2 * sy], outline=(255, 60, 60, 255), width=3)
    # 計画経路
    if paths:
        for seg in paths:
            pts = [tuple(np.array(cam.floor_to_pixel(*p)) * (sx, sy)) for p in seg]
            if len(pts) >= 2:
                dr.line(pts, fill=(0, 255, 255, 230), width=3)
    # 実際の軌跡
    if trace and len(trace) > 1:
        pts = [tuple(np.array(cam.floor_to_pixel(*p)) * (sx, sy)) for p in trace[::3]]
        dr.line(pts, fill=(255, 255, 255, 255), width=2)
    # 目的地(番号つき)
    for i, g in enumerate(plan.goals):
        u, v = np.array(cam.floor_to_pixel(*g.xy)) * (sx, sy)
        r = 13 * W / 640
        col = (255, 255, 0, 255) if active == i else (255, 255, 255, 255)
        dr.ellipse([u - r, v - r, u + r, v + r], outline=col, width=3, fill=(0, 0, 0, 140))
        dr.text((u - r * 0.45, v - r * 0.8), str(i + 1), font=f, fill=col)
    if robot_xy is not None:
        u, v = np.array(cam.floor_to_pixel(*robot_xy)) * (sx, sy)
        dr.ellipse([u - 7, v - 7, u + 7, v + 7], fill=(255, 40, 160, 255))
    return np.array(img)


def compose(main, inset, caption, status, badge="等倍速 / MuJoCo シミュレーション"):
    """メイン映像 + 右上に俯瞰図 + 下部にテロップ。"""
    img = Image.fromarray(main).convert("RGB")
    W, H = img.size
    iw = int(W * 0.34)
    ih = int(inset.shape[0] * iw / inset.shape[1])
    ins = Image.fromarray(inset).resize((iw, ih))
    img.paste(ins, (W - iw - 12, 12))
    dr = ImageDraw.Draw(img, "RGBA")
    dr.rectangle([W - iw - 12, 12, W - 12, 12 + ih], outline=(255, 255, 255, 255), width=2)
    s = H / 720
    dr.rectangle([0, H - 110 * s, W, H], fill=(0, 0, 0, 170))
    dr.text((20 * s, H - 100 * s), caption, font=font(int(26 * s)), fill=(255, 255, 255))
    dr.text((20 * s, H - 55 * s), status, font=font(int(26 * s)), fill=(120, 230, 255))
    dr.text((14 * s, 12 * s), badge, font=font(int(20 * s)), fill=(255, 255, 255))
    return np.array(img)
