#!/usr/bin/env bash
# 启动 llama.cpp 的 llama-server 作为翻译后端。
# 模型路径不硬编码:用第一个参数,或环境变量 LLAMA_MODEL。
#
# 用法:
#   bash start_llama.sh /path/to/model.gguf
#   LLAMA_MODEL=/path/to/model.gguf bash start_llama.sh
#   LLAMA_CTX=16384 LLAMA_PORT=8080 bash start_llama.sh ./models/xxx.gguf
set -euo pipefail

MODEL="${1:-${LLAMA_MODEL:-}}"
[ "$#" -gt 0 ] && shift          # 把模型参数移走,剩余参数透传给 llama-server

if [ -z "$MODEL" ]; then
  echo "用法: bash start_llama.sh <模型.gguf 路径>"
  echo "  或: LLAMA_MODEL=/path/to/model.gguf bash start_llama.sh"
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "找不到模型文件: $MODEL"
  exit 1
fi

# 以下都可用环境变量覆盖,均有默认值
LLAMA_BIN="${LLAMA_BIN:-llama-server}"   # 预编译包用 ./llama.cpp/build/bin/llama-server
HOST="${LLAMA_HOST:-0.0.0.0}"            # 0.0.0.0 = 允许局域网/Mac 远程访问
PORT="${LLAMA_PORT:-8080}"
NGL="${LLAMA_NGL:-99}"                   # GPU 层数,99=全放显存
CTX="${LLAMA_CTX:-8192}"                 # 上下文长度
# Qwen3 等带 thinking 的模型:用 jinja 模板 + 关闭推理,翻译输出更干净更快
EXTRA="${LLAMA_EXTRA:---jinja --reasoning-budget 0}"

echo "==> 启动 llama-server"
echo "    模型: $MODEL"
echo "    地址: $HOST:$PORT  | ngl=$NGL ctx=$CTX | extra: $EXTRA"
# shellcheck disable=SC2086
exec "$LLAMA_BIN" -m "$MODEL" -ngl "$NGL" -c "$CTX" \
     --host "$HOST" --port "$PORT" $EXTRA "$@"
