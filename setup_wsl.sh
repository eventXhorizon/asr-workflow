#!/usr/bin/env bash
# WSL (Ubuntu) + NVIDIA GPU 一键环境配置
# 在那台 Windows WSL 机器上、项目目录里执行: bash setup_wsl.sh
set -euo pipefail

echo "==> 1. 检查 NVIDIA 驱动 (WSL 里能看到 GPU 才行)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "!! nvidia-smi 不存在。请在 Windows 端安装最新 NVIDIA 驱动 (自带 WSL CUDA 支持),"
  echo "   不要在 WSL 里单独装 Linux 显卡驱动。装完重启 WSL: wsl --shutdown"
  exit 1
fi
nvidia-smi

echo "==> 2. 安装系统依赖 ffmpeg + python venv"
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv python3-pip

echo "==> 3. 创建虚拟环境 .venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

echo "==> 4. 安装 Python 依赖"
pip install -r requirements.txt

echo "==> 5. 安装 CUDA 运行库 (faster-whisper/CTranslate2 需要 cuBLAS + cuDNN 9)"
# 不依赖系统 CUDA,直接用 pip wheel 提供的库,最省心
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

echo
echo "==> 完成。每次使用前先激活环境:"
echo "    source .venv/bin/activate"
echo
echo
echo "==> 翻译用的 llama.cpp 需另外构建并启动 llama-server,见 README「部署第 3 步」。"
echo
echo "如果运行时报 'libcudnn ... cannot open shared object file',执行:"
echo '    export LD_LIBRARY_PATH=$(python -c "import nvidia.cublas.lib,nvidia.cudnn.lib,os;print(os.path.dirname(nvidia.cublas.lib.__file__)+\":\"+os.path.dirname(nvidia.cudnn.lib.__file__))"):$LD_LIBRARY_PATH'
echo "(可把上面这行加进 .venv/bin/activate 末尾,长期生效)"
