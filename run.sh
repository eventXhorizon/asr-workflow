#!/usr/bin/env bash
# 一条龙: 视频 -> 日语字幕(srt) -> 中文/双语字幕
# 用法: bash run.sh <视频或目录> [输出目录]
#   例: bash run.sh ./videos ./out
set -euo pipefail

SRC="${1:?用法: bash run.sh <视频或目录> [输出目录]}"
OUT="${2:-}"

# 激活 venv(若存在)
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

OUT_ARGS=()
[ -n "$OUT" ] && OUT_ARGS=(-o "$OUT")
TARGET="${OUT:-$SRC}"   # 翻译阶段去这里找 srt

echo "==> 1/2 ASR 转写(日语)"
python transcribe.py "$SRC" "${OUT_ARGS[@]}" --formats srt,txt

echo "==> 2/2 LLM 翻译成中文(llama.cpp)"
python translate.py "$TARGET" "${OUT_ARGS[@]}" --bilingual

echo "==> 全部完成。"
