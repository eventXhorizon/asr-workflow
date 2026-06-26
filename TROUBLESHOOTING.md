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

- **WSL 看不到某个 Windows 盘(如 F:)**
  `ls /mnt/` 没有该盘、或 `/mnt/f` 为空 = 没挂载(常见于后插入/移动硬盘)。手动挂:
  ```bash
  sudo mkdir -p /mnt/f && sudo mount -t drvfs 'F:\' /mnt/f
  ls /mnt/f
  ```
  开机自动挂:`/etc/fstab` 加 `F: /mnt/f drvfs defaults 0 0`。
  Windows 盘路径换算:`F:\X_下载` → `/mnt/f/X_下载`(盘符小写、无冒号、用 `/`)。

---

## Web 后台(webapp.py)类

- **启动报 `Cannot find empty port in range: 7860`**
  端口被占。换端口:`JP_ASR_WEB_PORT=7861 python webapp.py`。

- **下载产物报 `InvalidPathError: Cannot move ... to the gradio cache dir`**
  Gradio 5 默认禁止下载工作目录/临时目录之外的文件。`webapp.py` 已在 `launch` 加
  `allowed_paths=[OUT_DIR]` 放行产物目录;若改了输出位置,记得同步加进 `allowed_paths`。

- **目录浏览("列出视频")没反应 / 报 `OSError: [Errno 5] Input/output error`**
  drvfs 对中文目录递归列目录会偶发 I/O 错误。`webapp.py` 已改为只列当前层 + 失败重试。
  仍列不出时,直接在"② 路径框"粘贴完整路径即可(单个文件 ffmpeg 能正常读)。

- **大视频(十几 GB)上传只转圈无进度**
  浏览器上传大文件不合适。视频若在服务器本地(含挂载的 Windows 盘),**用路径选择,别上传**
  ——流水线就地读取、不拷贝。

- **任务一直"转写中"像没反应**
  长视频要先从(机械)盘读一遍抽音频,几分钟正常。看实时进度:`tail -f ~/out/<任务ID>/log.txt`,
  或网页点该任务行(日志会每 3 秒自动刷新)。

- **网页 llama-server 显示"运行中"但转写阶段没腾显存**
  webapp 只能停掉**自己启动**的 llama-server。若你**另外手动**起了一个,转写阶段它停不掉
  (会在日志警告),可能与 ASR 抢显存。解决:别手动起,交给 webapp 全权托管。

---

## 翻译类

### 翻译又慢又频繁失败:Qwen3 思考(reasoning)没关 ⭐⭐

**这是本项目踩得最深的坑。** 症状有两种,**根因都是模型在"思考"**:
- 慢:翻一句"こんにちは"生成 426 token(绝大部分是推理),每批 20~30 秒;
- 失败:`批量翻译失败(Expecting value: line 1 column 1 (char 0))`,即返回的 `content` 为空
  (推理把 token 预算用光,还没轮到输出 JSON 就被 `max_tokens` 截断)。

**判定**:对 llama-server 发一条请求看返回:
```bash
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"翻译成中文:こんにちは"}],"chat_template_kwargs":{"enable_thinking":false},"stream":false}' \
  | python3 -m json.tool
```
- 正常(思考已关):`content`="你好",**无 `reasoning_content`**,`completion_tokens` 个位数;
- 异常(思考还开):有大段 `reasoning_content`,`completion_tokens` 几百。

**解决(三道一起上,缺一可能不生效)**:
1. 启动 llama-server 必带 **`--jinja --reasoning-budget 0`**;
2. `translate.py` 每个请求已带 **`chat_template_kwargs={"enable_thinking": false}`**(需服务端 `--jinja`);
3. 仍兜底剥离 `<think>…</think>`。

> 经验:服务端 flag 有时对魔改/微调模型不生效,**客户端的 `enable_thinking=false` 才是关键**
> (本项目即靠它最终关掉)。`--jinja` 是它生效的前提。

### 批量翻译失败:返回被截断(上下文不够)

即使思考已关,若 **`-np` 开太大**导致每槽上下文太小,长批仍可能被截断成非法 JSON。
每槽上下文 = `-c ÷ -np`(如 `16384 ÷ 8 = 2048`)。日志里 `truncated = 1` 即此因。

**解决**:
- `translate.py --batch` 调小(默认已是 10,输出更短);
- 或减小 `-np` / 增大 `-c`,让每槽上下文更宽;
- `translate.py` 已对每请求设 `max_tokens` 上限,异常批快速失败并自动逐条回退,不丢段。

### 翻译太慢(串行)

`translate.py` 默认 **并发** 发批次(`--concurrency`,默认 4),对应 llama-server 的 `-np` 槽位。
想更快:启动加 `-np 8`,翻译用 `--concurrency 8`。串行(并发=1)会让 GPU 空等网络往返。

### 其它

- **翻译报"连不上 llama-server"**:`curl http://localhost:8080/health` 应返回 `{"status":"ok"}`;
  没起来就启动 llama-server(见 DEPLOY_WSL_CUDA.md)。
- **`request (N tokens) exceeds the available context size`**:启动时 `-c` 太小,加大并加 `-fa on`。

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
