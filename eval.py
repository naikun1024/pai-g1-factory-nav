"""複数レイアウトでの定量評価(動画なし)。レポートの「結果」の表に使う。

  python eval.py --planner mock --n 20
  python eval.py --planner gemini --n 10 --sleep 6   # 無料枠のレート制限に配慮して間隔をあける
"""
from __future__ import annotations

import argparse
import csv
import time

import mujoco
import numpy as np

from g1nav.locomotion import G1Walker
from g1nav.navigator import WaypointFollower
from g1nav.planner import GridPlanner, plan_mock, plan_with_gemini
from g1nav.scene import random_layout, write_scene
from g1nav.vision import TopCamera

INSTRUCTION = "部品棚で部品を受け取り、検査台で検査してから出荷エリアへ運んでください。"
EXPECTED = ["parts", "inspect", "ship"]


def run_one(seed, planner, thinking="low"):
    stations, obstacles = random_layout(np.random.default_rng(seed))
    m = mujoco.MjModel.from_xml_path(write_scene(stations, obstacles))
    d = mujoco.MjData(m)
    w = G1Walker(m, d)
    cam = TopCamera(m, d)
    obs_ids = {m.geom(o.name + "_geom").id for o in obstacles}
    true_xy = {s.key: np.array(s.xy) for s in stations}
    hits = [0]

    def tick():
        for i in range(d.ncon):
            if d.contact[i].geom1 in obs_ids or d.contact[i].geom2 in obs_ids:
                hits[0] += 1
                return

    w.run(1.0, (0, 0, 0))
    if planner == "gemini":
        r = mujoco.Renderer(m, cam.H, cam.W)
        plan = plan_with_gemini(cam.render(r), INSTRUCTION, cam, thinking_level=thinking)
        r.close()
    else:
        plan = plan_mock(INSTRUCTION, stations, obstacles)
    keys = [min(true_xy, key=lambda k: np.linalg.norm(true_xy[k] - g.xy)) for g in plan.goals]
    row = dict(seed=seed, order_correct=keys == EXPECTED, n_goals=len(plan.goals),
               n_obstacles_detected=len(plan.obstacles), n_obstacles_true=len(obstacles),
               latency_s=round(plan.latency_s, 2))
    perc = [float(np.linalg.norm(g.xy - true_xy[k])) for g, k in zip(plan.goals, keys)]
    row["perception_err_m"] = round(float(np.mean(perc)), 3) if perc else None
    gp, f = GridPlanner(plan.obstacles), WaypointFollower(w)
    arr, status = [], "ok"
    for g, k in zip(plan.goals, keys):
        wps = gp.plan(w.pose()[:2], g.xy)
        if wps is None:
            status = "noplan"
            break
        res = f.go(wps, on_tick=tick)
        if res != "arrived":
            status = res
            break
        x, y, _ = w.pose()
        arr.append(float(np.hypot(x - true_xy[k][0], y - true_xy[k][1])))
    row.update(status=status, arrival_err_m=round(float(np.mean(arr)), 3) if arr else None,
               all_on_mat=bool(arr) and len(arr) == len(plan.goals) and max(arr) < 0.35,
               collision_frames=hits[0], sim_time_s=round(d.time, 1))
    row["success"] = row["order_correct"] and row["all_on_mat"] and row["collision_frames"] == 0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["gemini", "mock"], default="mock")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--start", type=int, default=100, help="評価用シードの開始番号(デモと別にする)")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--thinking", default="low", help="Gemini の思考レベル(low / medium / high)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"outputs/eval_{args.planner}{'_' + args.thinking if args.planner == 'gemini' else ''}.csv"
    rows = []
    for s in range(args.start, args.start + args.n):
        try:
            row = run_one(s, args.planner, args.thinking)
        except Exception as e:  # API エラーなども 1 試行の失敗として記録する
            row = dict(seed=s, status=f"error: {e}", success=False)
        rows.append(row)
        print(row)
        if args.sleep:
            time.sleep(args.sleep)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    def stat(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f} (n={len(v)})" if len(v) > 1 else str(v)

    n_ok = sum(bool(r.get("success")) for r in rows)
    print("\n===== まとめ =====")
    print(f"planner={args.planner}  試行数 n={len(rows)}")
    print(f"成功率(順番正しい・全ステーション到着・接触なし): {n_ok}/{len(rows)} = {n_ok / len(rows):.0%}")
    print(f"訪問順の正解率: {sum(bool(r.get('order_correct')) for r in rows)}/{len(rows)}")
    print(f"位置認識誤差 [m]: {stat('perception_err_m')}")
    print(f"到着誤差 [m]:     {stat('arrival_err_m')}")
    print(f"API 応答時間 [s]: {stat('latency_s')}")
    print(f"結果 CSV: {out}")


if __name__ == "__main__":
    main()
