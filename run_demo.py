"""デモ本体:日本語の指示 → AI(Gemini Robotics ER 2)で認識・計画 → A* → G1 の学習済み歩行方策で移動 → 動画出力

使い方:
  python run_demo.py --planner mock                 # API キーなしで動作確認
  python run_demo.py --planner gemini               # 本番(GEMINI_API_KEY が必要)
  python run_demo.py --planner gemini --no-video    # 動画を作らずに速く確認
"""
from __future__ import annotations

import argparse
import json
import os
import time

import imageio.v2 as imageio
import mujoco
import numpy as np

from g1nav import viz
from g1nav.locomotion import G1Walker
from g1nav.navigator import WaypointFollower
from g1nav.planner import GridPlanner, plan_mock, plan_with_gemini
from g1nav.scene import STATIONS, in_hazard, random_layout, write_scene
from g1nav.vision import TopCamera

DEFAULT_INSTRUCTIONS = [
    "部品棚で部品を受け取ってから、検査台に運んでください。",
    "次は出荷エリアに届けて、最後は充電ステーションに戻ってください。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["gemini", "mock"], default="gemini")
    ap.add_argument("--seed", type=int, default=0, help="レイアウトの乱数シード")
    ap.add_argument("--instruction", action="append", help="指示文(複数指定可)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="outputs/demo.mp4")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--thinking", default="low", help="Gemini の思考レベル(low / medium / high)")
    args = ap.parse_args()
    instructions = args.instruction or DEFAULT_INSTRUCTIONS
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    W, H = args.width, args.height

    # ---------- シーン ----------
    stations, obstacles = random_layout(np.random.default_rng(args.seed))
    model = mujoco.MjModel.from_xml_path(write_scene(stations, obstacles))
    data = mujoco.MjData(model)
    obstacle_geoms = {model.geom(o.name + "_geom").id for o in obstacles}
    cam = TopCamera(model, data)

    class FrameWriter:
        """フレームをメモリに溜めずに直接 mp4 へ書く(1280x720 で溜めると数GBになるため)。"""
        def __init__(self):
            self.n = 0
            self.w = None if args.no_video else imageio.get_writer(
                args.out, fps=args.fps, codec="libx264", macro_block_size=8,
                ffmpeg_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"])

        def append(self, img):
            self.w.append_data(img)
            self.n += 1

        def extend(self, imgs):
            for im in imgs:
                self.append(im)

        def __len__(self):
            return self.n

    frames = FrameWriter()
    state = dict(caption="", status="", plan=None, paths=[], active=None, follower=None)
    renderer = None if args.no_video else mujoco.Renderer(model, H, W)
    top_renderer = mujoco.Renderer(model, cam.H, cam.W)
    track = mujoco.MjvCamera()
    track.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    track.trackbodyid = model.body("pelvis").id
    track.distance, track.elevation, track.azimuth = 3.4, -20.0, -150.0
    hits = {"n": 0, "hazard": 0}

    def on_frame():
        if in_hazard(walker.pose()[:2], obstacles):
            hits["hazard"] += 1
        # 障害物との接触チェック(評価用)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 in obstacle_geoms or c.geom2 in obstacle_geoms:
                hits["n"] += 1
                break
        if renderer is None:
            return
        _, _, yaw = walker.pose()
        track.azimuth += 0.03 * ((np.degrees(yaw) + 210 - track.azimuth + 180) % 360 - 180)  # なめらかに追従
        renderer.update_scene(data, camera=track)
        main_img = renderer.render()
        if state.get("top_img") is None or len(frames) % 15 == 0:  # 俯瞰図は0.5秒ごとに更新(高速化)
            state["top_img"] = cam.render(top_renderer)
        top = state["top_img"]
        f = state["follower"]
        inset = viz.draw_plan(top, cam, state["plan"], state["paths"], f.trace if f else None,
                              walker.pose()[:2], state["active"]) if state["plan"] else top
        frames.append(viz.compose(main_img, inset, state["caption"], state["status"]))

    walker = G1Walker(model, data, frame_cb=on_frame, fps=args.fps)
    follower = WaypointFollower(walker)

    def hold(img, sec):
        if renderer is not None:
            frames.extend([img] * int(sec * args.fps))

    # ---------- オープニング ----------
    hold(viz.card(W, H, "言葉で指示できる工場内搬送ヒューマノイド",
                  ["Unitree G1(MuJoCo シミュレーション)",
                   "歩行:学習済み強化学習方策(Unitree 公式 / 推論のみ)",
                   f"認識・計画:{'Gemini Robotics ER 2' if args.planner == 'gemini' else 'mock(真値・動作確認用)'} + A* 経路計画",
                   "映像はすべて等倍速"], sub="PAI 最終課題"), 4)
    hold(viz.card(W, H, "システム構成",
                  ["① 日本語の作業指示 + 俯瞰カメラ画像",
                   "② VLM:訪問するステーションの順番・位置、障害物を認識",
                   "③ A*:障害物を膨張させた地図で安全な経路を計算",
                   "④ 追従制御:経路 → 速度指令 (vx, wz)",
                   "⑤ RL 歩行方策:速度指令 → 脚12関節の目標角(50Hz)"]), 6)

    walker.run(1.0, (0, 0, 0))  # 立ち姿勢を安定させる
    metrics = dict(planner=args.planner, seed=args.seed, missions=[])
    true_xy = {s.key: np.array(s.xy) for s in stations}

    for mi, instruction in enumerate(instructions):
        state.update(caption=f"指示{mi + 1}:{instruction}", status="AI が俯瞰画像を解析中…", paths=[], active=None)
        top = cam.render(top_renderer)
        state["top_img"] = top
        imageio.imwrite(f"outputs/top_{mi + 1}.png", top)
        if args.planner == "gemini":
            plan = plan_with_gemini(top, instruction, cam, thinking_level=args.thinking)
        else:
            plan = plan_mock(instruction, stations, obstacles)
        state["plan"] = plan
        print(f"\n[指示{mi + 1}] {instruction}")
        print(f"  計画({plan.source}, {plan.latency_s:.1f}s): " + " → ".join(g.label for g in plan.goals))
        with open(f"outputs/plan_{mi + 1}.json", "w") as f:
            json.dump(dict(instruction=instruction, source=plan.source, latency_s=plan.latency_s, raw=plan.raw,
                           goals=[dict(label=g.label, xy=g.xy.tolist(), point=g.point) for g in plan.goals],
                           obstacles=plan.obstacles), f, ensure_ascii=False, indent=2)

        # 経路計画
        gp = GridPlanner(plan.obstacles)
        start = walker.pose()[:2]
        paths, ok = [], True
        for g in plan.goals:
            wps = gp.plan(start, g.xy)
            if wps is None:
                print(f"  [error] {g.label} への経路が見つかりません")
                ok = False
                break
            paths.append([start] + wps)
            start = g.xy
        state["paths"] = paths
        ai_img = viz.draw_plan(top, cam, plan, paths)
        imageio.imwrite(f"outputs/ai_view_{mi + 1}.png", ai_img)
        hold(viz.compose(np.array(viz.card(W, H, f"指示{mi + 1}", [instruction, "",
                         "AI の認識結果(右上):番号=訪問順、赤枠=障害物、水色=計画経路"])),
                         ai_img, state["caption"], f"計画: {' → '.join(g.label for g in plan.goals)}"), 5)
        if not ok:
            continue

        mres = dict(instruction=instruction, latency_s=plan.latency_s, goals=[])
        state["follower"] = follower
        for gi, (g, path) in enumerate(zip(plan.goals, paths)):
            state["active"] = gi
            # AI の推定位置と真のステーション位置のずれ(認識誤差)を記録
            key = min(true_xy, key=lambda k: np.linalg.norm(true_xy[k] - g.xy))
            label_ja = STATIONS[key][0]
            state["status"] = f"→ {gi + 1}/{len(plan.goals)} {label_ja}へ移動中"
            t0 = data.time
            result = follower.go(path[1:])
            x, y, _ = walker.pose()
            err_true = float(np.hypot(x - true_xy[key][0], y - true_xy[key][1]))
            percep = float(np.linalg.norm(g.xy - true_xy[key]))
            on_mat = err_true < 0.35  # マット(0.7m 角)の上に立てたか
            mres["goals"].append(dict(label=g.label, station=key, result=result, time_s=round(data.time - t0, 1),
                                      perception_err_m=round(percep, 3), arrival_err_m=round(err_true, 3),
                                      on_mat=bool(on_mat)))
            print(f"  {label_ja}: {result} / 所要 {data.time - t0:.1f}s / 認識誤差 {percep:.2f}m / 到着誤差 {err_true:.2f}m")
            if result != "arrived":
                state["status"] = f"× {label_ja}で停止({result})"
                follower.stop(2.0)
                break
            state["status"] = f"✓ {label_ja}に到着(目標との誤差 {err_true:.2f} m)作業中…"
            follower.stop(1.5)
        metrics["missions"].append(mres)

    metrics["collisions_frames"] = hits["n"]
    metrics["hazard_frames"] = hits["hazard"]
    metrics["sim_time_s"] = round(data.time, 1)
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---------- エンディング ----------
    goals = [g for m in metrics["missions"] for g in m["goals"]]
    n_ok = sum(g["on_mat"] for g in goals)
    lines = [f"到着成功:{n_ok} / {len(goals)} ステーション",
             f"到着誤差(平均):{np.mean([g['arrival_err_m'] for g in goals]):.2f} m" if goals else "",
             f"AI の位置認識誤差(平均):{np.mean([g['perception_err_m'] for g in goals]):.2f} m" if goals else "",
             f"障害物との接触:{'なし' if hits['n'] == 0 else str(hits['n']) + ' フレーム'}",
             f"立入禁止ゾーンへの侵入:{'なし' if hits['hazard'] == 0 else str(hits['hazard']) + ' フレーム'}",
             f"シミュレーション時間:{data.time:.0f} 秒"]
    hold(viz.card(W, H, "結果", lines, sub=f"レイアウト seed={args.seed} / planner={args.planner}"), 5)

    if renderer is not None:
        frames.w.close()
        print(f"\n動画を書き出しました: {args.out}({len(frames) / args.fps:.0f} 秒)")
    print("指標: outputs/metrics.json")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"実行時間 {time.time() - t:.0f} 秒")
