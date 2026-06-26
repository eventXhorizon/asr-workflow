# jp_asr — 日语视频转中文字幕

**视频 → 日语 ASR → 日文字幕 → 本地 LLM 翻译 → 中文/中日双语字幕**,全程本地、离线。

- ASR:[faster-whisper](https://github.com/SYSTRAN/faster-whisper)(`large-v3` / `kotoba-whisper`)
- 翻译:本地 [llama.cpp](https://github.com/ggml-org/llama.cpp)(`llama-server`,OpenAI 兼容接口)

面向 **Windows WSL2 + NVIDIA GPU(如 3090Ti 24GB)**。

```
video.mp4 ──transcribe.py──> video.srt ──translate.py──> video.zh.srt
   (抽音频 + ASR)              (日文字幕)  (llama.cpp 翻译)  video.bilingual.srt
```

## 快速上手

```bash
# 0. 一次性:装环境(见 DEPLOY_WSL_CUDA.md 处理 llama.cpp 的 CUDA 编译)
bash setup_wsl.sh && source .venv/bin/activate

# 1. 转写(带 BGM 的视频加 --vad-threshold 0.2 防漏段)
python transcribe.py /mnt/d/video.mp4 -o ~/out --model large-v3 --vad-threshold 0.2

# 2. 启动翻译后端(另一个终端),然后翻译
bash start_llama.sh ~/models/你的模型.gguf
python translate.py ~/out/video.srt --bilingual --host http://localhost:8080
```

产出:`*.srt`(日)、`*.zh.srt`(中)、`*.bilingual.srt`(双语)、`*.txt`。

## 文档

| 文档 | 内容 |
|------|------|
| [USAGE.md](USAGE.md) | 详细用法、全部参数、输出格式、性能、推荐工作流 |
| [DEPLOY_WSL_CUDA.md](DEPLOY_WSL_CUDA.md) | WSL2+NVIDIA 从源码编译 CUDA 版 llama.cpp 的完整实录 |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 常见问题(含 VAD 漏段、显存、下载、代理) |

## 脚本一览

| 文件 | 作用 |
|------|------|
| `transcribe.py` | 视频/音频 → 日语字幕(ffmpeg 抽音频 + faster-whisper) |
| `translate.py` | 日语 srt → 中文 / 双语字幕(调本地 llama-server) |
| `run.sh` | 一条龙:依次跑 transcribe + translate |
| `start_llama.sh` | 启动 llama-server 翻译后端(模型路径走参数) |
| `setup_wsl.sh` | WSL 环境一键配置(ffmpeg、venv、CUDA 库) |
