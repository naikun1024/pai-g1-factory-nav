# 言葉で指示できる工場内搬送ヒューマノイド(Unitree G1 × Gemini Robotics ER 2)

PAI 最終課題 日本語の作業指示と俯瞰カメラ画像から、VLM(Gemini Robotics ER 2)が
「どのステーションへどの順で行くか」と「障害物の位置」を認識し、A* で安全な経路を計算、
Unitree 公式の **学習済み強化学習歩行方策** で G1 を歩かせます(MuJoCo シミュレーション)。

## デモ動画
https://youtu.be/3eoXVqPIqbs (等倍速・MuJoCo シミュレーション)

## 工夫した点:立入禁止ゾーン(作業者エリア)の認識と回避
床の黄黒しま模様を「人が作業している区域」とし、Gemini に障害物として検出させ、A* で避けて通る経路を計画する。
意味の理解は AI、安全の担保は検証できるアルゴリズム、という役割分担にした。

## 結果
| 条件 | 試行数 | 成功率 | 立入禁止ゾーンに入った試行 |
| --- | --- | --- | --- |
| Gemini ER 2(ゾーン認識あり) | 10 | 10/10 | 0/10 |
| mock・ゾーン認識あり | 20 | 20/20 | 0/20 |
| mock・ゾーンを無視(比較用) | 20 | 9/20 | 11/20 |

- 同じ20配置で比較。ゾーンを避けることによる所要時間の増加は平均 2.0 ± 1.9 秒。
- 位置認識誤差は 0.005 ± 0.002 m(n=10)。単色の模擬環境のため高精度に出ており、実際の工場画像では悪化すると考えられる。

```
日本語の指示 + 俯瞰画像
   └─▶ Gemini Robotics ER 2 … 訪問先(順番・位置)と障害物を認識   [AI 推論 ①]
         └─▶ A* 経路計画 … 障害物をロボット半径ぶん膨張させた地図で経路を計算
               └─▶ ウェイポイント追従 … 経路 → 速度指令 (vx, wz)
                     └─▶ RL 歩行方策(motion.pt, 50Hz)… 速度指令 → 脚12関節の目標角   [AI 推論 ②]
                           └─▶ PD 制御 → MuJoCo
```

## 動作環境
- macOS(Apple Silicon)/ Python 3.11〜3.12 で動作想定。GPU 不要。
- Linux でも動作確認済み(`MUJOCO_GL=egl`)。

## セットアップ
```bash
python3 -m venv .venv && source .venv/bin/activate   # 仮想環境を作って有効化
pip install -r requirements.txt                       # 依存ライブラリを導入
bash scripts/get_assets.sh                            # G1 モデルと学習済み方策を取得(約80MB)
export GEMINI_API_KEY="(Google AI Studio で発行したキー)"
```

## 実行
```bash
python run_demo.py --planner mock --no-video   # API なしで動作確認(数秒)
python run_demo.py --planner gemini            # 本番:outputs/demo.mp4 を出力
python eval.py --planner mock --n 20           # 定量評価(真値を使った上限性能)
python eval.py --planner gemini --n 10 --sleep 6
```
出力: `outputs/demo.mp4`(動画)、`outputs/ai_view_*.png`(AI の認識結果)、`outputs/plan_*.json`(AI の生の応答)、`outputs/metrics.json`、`outputs/eval_*.csv`

## ファイル構成
| ファイル | 役割 |
|---|---|
| `g1nav/scene.py` | 工場フロア(ステーション・パレット・カメラ)の生成 |
| `g1nav/locomotion.py` | 学習済み歩行方策の推論(観測の作り方は公式 deploy_mujoco.py に準拠) |
| `g1nav/vision.py` | 俯瞰カメラの画素 ⇔ 床面座標の変換 |
| `g1nav/planner.py` | Gemini による認識・タスク計画、A* 経路計画 |
| `g1nav/navigator.py` | ウェイポイント追従制御 |
| `g1nav/viz.py` | 動画のテロップ・重ね描き |
| `run_demo.py` / `eval.py` | デモ動画の生成 / 定量評価 |

## 制約・前提
- 自己位置はシミュレータの真値を使用(実機では SLAM 等が必要)。
- 歩行方策は Unitree 公式の学習済みモデルを推論のみで使用(自前学習はしていない)。
- 12 自由度モデル(脚のみ駆動、腕は固定)のため、物を持つ動作はしない(「到着して作業」の表現にとどめる)。


## クレジット・ライセンス
- G1 モデル・学習済み方策: [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)(BSD-3-Clause)。`scripts/get_assets.sh` で取得し、本リポジトリには含めない。
- Gemini Robotics ER 2: Google Gemini API。
- 本リポジトリのコードは生成 AI(Claude)の支援を受けて作成し、内容を確認・改変しています。
