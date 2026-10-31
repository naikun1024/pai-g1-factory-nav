#!/usr/bin/env bash
# Unitree 公式の unitree_rl_gym から G1 のモデルと学習済み歩行方策だけを取得する(約60MB)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/unitree_rl_gym ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_rl_gym.git third_party/unitree_rl_gym
  git -C third_party/unitree_rl_gym sparse-checkout set resources/robots/g1_description deploy/pre_train/g1 deploy/deploy_mujoco
fi
ls third_party/unitree_rl_gym/deploy/pre_train/g1/motion.pt && echo "OK: G1 アセットを取得しました"
