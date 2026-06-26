# 故障排查 / 已知问题

## 转写漏段:大段说话被跳过(VAD 误删)⭐

**症状**:字幕里时间轴出现大跳跃(如 `00:00:35` 直接跳到 `00:01:58`),
中间明明有人说话却没有任何字幕。

**根因**:不是 ASR 模型的问题,是它前面那层 **VAD(Silero 语音活动检测)误判**。
faster-whisper 默认先用 VAD 把"非语音"片段切掉再送去转写;当**语音叠加背景音乐
(BGM)**时(动画、剧集常见),VAD 在默认阈值 `0.5` 下容易把"说话+音乐"判成噪音
**整段删掉**。

**判定方法**:换不同 ASR 模型(如 `kotoba-whisper` 与 `large-v3`)若**漏的是同一段**,
基本可确定是共用的 VAD 干的,而非模型。

**解决**(`transcribe.py` 已支持):
```bash
# 首选:保留 VAD 但调低灵敏度,既补回语音又仍跳过纯静音
python transcribe.py video.mp4 --vad-threshold 0.2     # 还漏可再降到 0.1

# 诊断/兜底:完全关掉 VAD(纯静音处可能出现幻觉文字,不建议长期用)
python transcribe.py video.mp4 --no-vad
```

**经验值**:
- 纯人声、无 BGM:默认 `0.5` 即可。
- 带 BGM / 综艺 / 剧集:`0.2`~`0.3` 比较稳。
- 实在漏得多:`--no-vad`,但要接受静音段偶发幻觉(可后期删)。

---

## 部署 / 环境类

- **`libcudnn ... cannot open shared object file`**
  ctranslate2 找不到 CUDA 库。把 venv 里的库加进搜索路径(`setup_wsl.sh` 末尾也给了):
  ```bash
  export LD_LIBRARY_PATH=$(python -c "import os,nvidia.cublas.lib,nvidia.cudnn.lib;print(os.path.dirname(nvidia.cublas.lib.__file__)+':'+os.path.dirname(nvidia.cudnn.lib.__file__))"):$LD_LIBRARY_PATH
  ```

- **`nvidia-smi` 在 WSL 里找不到**
  Windows 端装/更新 NVIDIA 驱动,然后 PowerShell 执行 `wsl --shutdown` 重启 WSL。
  不要在 WSL 里单独装 Linux 显卡驱动。

- **llama.cpp 编译 / CUDA 部署**:见 [DEPLOY_WSL_CUDA.md](DEPLOY_WSL_CUDA.md)。

---

## 翻译类

- **翻译报"连不上 llama-server"**
  确认服务在跑:`curl http://localhost:8080/health` 应返回 `{"status":"ok"}`。
  没起来就启动 `llama-server`(见 DEPLOY_WSL_CUDA.md)。

- **`request (N tokens) exceeds the available context size`**
  启动 `llama-server` 时上下文太小。加大 `-c`(如 `-c 16384`)并加 `-fa on`
  (flash attention,省显存);或把 `translate.py --batch` 调小。

- **译文里混入推理过程 / `<think>`**
  启动加 `--jinja --reasoning-budget 0` 关掉 Qwen3 的 thinking;
  `translate.py` 也会兜底剥离 `<think>…</think>`。

---

## 显存类

- **ASR 和翻译抢显存**(20GB 翻译模型 + 24GB 卡余量很小)
  分两阶段:先全部 `transcribe.py`(llama-server 不开),再启动 llama-server 批量
  `translate.py`。别让 whisper 和 LLM 同时占显存。

- **`llama-server` OOM**
  换更小量化(`Q4_K_M`)、调小 `-ngl`(部分层放 CPU)、减小 `-c`,或开 `-fa on`。

- **ASR OOM**
  `transcribe.py --compute-type int8_float16`,或换 `large-v3-turbo`。

---

## 下载类

- **HuggingFace 下不动 / `hf-mirror.com` 308 重定向回 huggingface.co**
  用代理直连(WSL mirrored 模式下 Clash 在 127.0.0.1):
  ```bash
  export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
  export NO_PROXY=localhost,127.0.0.1
  unset HF_ENDPOINT
  ```

- **apt 走了坏代理**
  加一个最后生效的覆盖,强制直连(国内镜像不需代理):
  ```bash
  echo 'Acquire::http::Proxy "DIRECT";
  Acquire::https::Proxy "DIRECT";' | sudo tee /etc/apt/apt.conf.d/99-no-proxy
  ```
