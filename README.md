# jp_asr — 日语视频转中文字幕

完整流水线:**视频 → 日语 ASR → 日文字幕 → 本地 LLM 翻译 → 中文/中日双语字幕**

- ASR:[faster-whisper](https://github.com/SYSTRAN/faster-whisper) + **kotoba-whisper-v2.0**(针对日语调优,质量优于原版 large-v3)
- 翻译:本地 **llama.cpp**(`llama-server`,OpenAI 兼容接口),离线、免费,时间轴原样保留

为 **Windows WSL(Ubuntu)+ NVIDIA GPU(如 3090Ti 24GB)** 环境准备。

## 流程概览

```
video.mp4 ──transcribe.py──> video.srt ──translate.py──> video.zh.srt
   (ffmpeg 抽音频 + ASR)        (日文字幕)  (llama.cpp 翻译)  video.bilingual.srt
```

---

## 项目结构

| 文件 | 作用 |
|------|------|
| `transcribe.py` | 视频/音频 → 日语字幕(ffmpeg 抽音频 + faster-whisper 转写) |
| `translate.py` | 日语 srt → 中文 / 中日双语字幕(调本地 llama-server) |
| `run.sh` | 一条龙:依次跑 transcribe + translate |
| `start_llama.sh` | 启动 llama-server 翻译后端(模型路径走参数,不硬编码) |
| `setup_wsl.sh` | WSL 环境一键配置(驱动检查、ffmpeg、venv、CUDA 库) |
| `requirements.txt` | Python 依赖 |
| `README.md` | 本文档 |
| `DEPLOY_WSL_CUDA.md` | WSL2+NVIDIA 编译 CUDA 版 llama.cpp 的完整实录 |

## 环境要求

| 项 | 要求 |
|----|------|
| 系统 | Windows WSL2(Ubuntu),或任意 Linux |
| GPU | NVIDIA,显存 ≥ 6GB;3090Ti 24GB 可上最高质量配置 |
| 驱动 | Windows 端最新 NVIDIA 驱动(自带 WSL CUDA 直通) |
| Python | 3.9+(`setup_wsl.sh` 用系统自带 `python3`) |
| 磁盘 | 模型缓存约 3~6GB(ASR)+ 6~20GB(GGUF 翻译模型,按量化档) |
| 网络 | 仅首次下载模型时需要;之后全程离线 |

---

## 部署到 WSL 机器

1. 把整个 `jp_asr/` 目录拷到那台 WSL 机器(scp / 共享盘 / git 均可)。
2. 在 WSL 里进入目录,一键装环境:

   ```bash
   cd jp_asr
   bash setup_wsl.sh
   ```

   脚本会:检查 `nvidia-smi` → 装 `ffmpeg` 和 venv → 装 Python 依赖 → 装 CUDA 运行库(cuBLAS + cuDNN 9)。

   > 前提:Windows 端装了**最新 NVIDIA 驱动**(自带 WSL CUDA 直通)。不要在 WSL 里单独装 Linux 显卡驱动。

3. 装翻译用的 llama.cpp(只需一次)。来源:<https://github.com/ggml-org/llama.cpp>

   > ⚠️ **官方 Linux 预编译包没有 CUDA 版**(只有 CPU/Vulkan/ROCm…),而 **Vulkan 在
   > WSL 里用不了 NVIDIA**(只有 llvmpipe 软件渲染)。所以 N 卡上**必须自己编译 CUDA 版**。
   > 完整踩坑实录见 **[DEPLOY_WSL_CUDA.md](DEPLOY_WSL_CUDA.md)**,这里给精简版:

   ```bash
   # 工具链(nvidia-cuda-toolkit 提供 nvcc,国内镜像可直连)
   sudo apt install -y build-essential cmake git libcurl4-openssl-dev nvidia-cuda-toolkit
   nvcc --version       # 能打印版本(如 12.4)= 就绪

   # 编译(CUDA_ARCHITECTURES=86 对应 3090Ti 算力 8.6,只编这一档更快)
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86"
   cmake --build build --config Release -j        # 产物在 build/bin/llama-server
   ```
   > 编译中途的 `UI: npm install failed` 是非致命的,会自动从 HF 下预编译 UI 继续;
   > 看到 `[100%] Built target llama-server` 即成功。

   **启动服务**。用本仓库的 `start_llama.sh`,模型路径走参数,不写死:
   ```bash
   # 跑你本地已有的 gguf:
   bash start_llama.sh /path/to/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf

   # 改端口/上下文/二进制路径用环境变量,无需改脚本:
   LLAMA_CTX=16384 LLAMA_BIN=./llama.cpp/build/bin/llama-server \
       bash start_llama.sh /path/to/model.gguf
   ```
   也可以直接手敲(等价):
   ```bash
   llama-server -m /path/to/model.gguf -ngl 99 -c 8192 \
       --host 0.0.0.0 --port 8080 --jinja --reasoning-budget 0
   ```
   - `-ngl 99`:全部层放 GPU(显存够就拉满,最快)
   - `--host 0.0.0.0`:允许局域网/Mac 远程访问
   - `--jinja --reasoning-budget 0`:Qwen3 等带 thinking 的模型,用模板并关推理,
     翻译输出更干净更快(老版二进制不识别就设 `LLAMA_EXTRA="--jinja"` 或留空)
   - 验证就绪:`curl http://localhost:8080/health` → `{"status":"ok"}`

   > **没有现成 gguf 想直接下**:llama-server 也支持 `-hf 仓库:量化档` 自动从
   > HuggingFace 下载,例如 `-hf Qwen/Qwen2.5-14B-Instruct-GGUF:Q5_K_M`。
   > 显存占用粗估:Q4_K_M 的 35B 约 20GB,3090Ti 24GB 够用,但上下文别开太大。

---

## 一条龙(推荐)

视频 → 中文/双语字幕,一条命令搞定(**翻译阶段需要 `llama-server` 已在运行**):

```bash
bash run.sh ./videos ./out      # 处理 ./videos 里的视频,结果放 ./out
bash run.sh video.mp4           # 单个文件,结果与源文件同目录
```

产出:`*.srt`(日文)、`*.zh.srt`(中文)、`*.bilingual.srt`(中日双语)、`*.txt`。

> **从 Mac 远程跑**:`ssh 用户名@WSL机器IP` 进去后在项目目录执行即可。
> 长任务建议套 `nohup` 或 `tmux`,断开 SSH 也不中断:
> ```bash
> tmux new -s asr        # 新建会话;断开后用 tmux attach -t asr 回来
> bash run.sh ./videos ./out
> ```
> WSL 的 IP 在 WSL 里用 `hostname -I` 查;Windows 防火墙可能需放行 22 端口。

---

## 分步使用

每次先激活环境:

```bash
source .venv/bin/activate
```

### 1. 转写日语字幕(transcribe.py)

转写单个视频(默认输出 `srt + txt`,与源文件同目录):

```bash
python transcribe.py /path/to/video.mp4
```

批量转写一个目录,结果统一放到 `out/`,并输出全部格式:

```bash
python transcribe.py ./videos -o ./out --formats srt,txt,vtt,json
```

> 首次运行会自动从 HuggingFace 下载模型(几 GB),需联网,之后走本地缓存。

### 2. 翻译成中文(translate.py)

先确保 `llama-server` 已在运行(见部署第 3 步),再翻译:

```bash
python translate.py video.srt --bilingual          # 出 video.zh.srt + video.bilingual.srt
python translate.py ./out --bilingual              # 批量翻译目录里的所有 srt
python translate.py ./out --host http://192.168.1.50:8080   # 翻译服务在另一台机器
```

> 换翻译模型不在这里改,而是用不同 GGUF 重启 `llama-server`。
> 自动跳过自己生成的 `.zh.srt` / `.bilingual.srt`,不会重复翻译。

---

## 常用参数

### transcribe.py

| 参数 | 说明 | 默认 |
|------|------|------|
| `inputs` | 文件或目录(可多个;目录取其中的媒体文件) | 必填 |
| `-o, --outdir` | 输出目录 | 与输入同目录 |
| `-m, --model` | 模型,见下 | `kotoba-tech/kotoba-whisper-v2.0-faster` |
| `--compute-type` | `float16` / `int8_float16`(省显存) / `float32`(极致质量) | `float16` |
| `--beam-size` | beam 宽度,越大越准越慢 | `5` |
| `--formats` | `srt,txt,vtt,json`(json 含词级时间戳) | `srt,txt` |
| `--language` | 语种,日语=`ja` | `ja` |
| `--overwrite` | 已有结果也重新跑 | 关 |
| `--keep-wav` | 保留抽取的 wav | 关 |

### 模型选择

- `kotoba-tech/kotoba-whisper-v2.0-faster` —— **日语质量最佳(默认)**
- `large-v3` —— 通用最强基线,生态最稳;若 kotoba 版下载/兼容有问题,用它兜底
- `large-v3-turbo` —— 更快,质量略降

3090Ti 24GB 显存充足,追求极致质量可以:

```bash
python transcribe.py video.mp4 --compute-type float32 --beam-size 10
```

### translate.py

| 参数 | 说明 | 默认 |
|------|------|------|
| `inputs` | `.srt` 文件或目录 | 必填 |
| `-m, --model` | 模型名(llama-server 单模型时被忽略,仅记录用) | `local-gguf` |
| `--host` | llama-server 地址 | `http://localhost:8080` |
| `--batch` | 每批翻译多少条字幕 | `20` |
| `--bilingual` | 同时输出中日双语 | 关 |
| `--only-bilingual` | 只出双语,不出纯中文 | 关 |

> 实际用哪个模型由 `llama-server` 启动时加载的 GGUF 决定,不在本脚本里切换。
> 翻译质量:32B(`Q4_K_M`)> 14B(`Q5_K_M`)。日→中翻译 Qwen2.5 系表现很好。
> 远程访问那台 WSL:`--host http://<WSL机器IP>:8080`(服务需 `--host 0.0.0.0` 启动)。
> `--batch` 调大省调用次数但单次更易出格式问题(已有逐条回退兜底)。

---

## 输出文件说明

以 `video.mp4` 为例,跑完一条龙后得到:

| 文件 | 内容 | 用途 |
|------|------|------|
| `video.srt` | 日文字幕,带时间轴 | 原文校对、二次编辑 |
| `video.txt` | 日文纯文本(无时间轴) | 全文阅读、喂给其他工具 |
| `video.zh.srt` | 中文字幕,时间轴同 `video.srt` | 配视频看的中文字幕 |
| `video.bilingual.srt` | 中日双语(中上日下) | 对照学习 / 校对翻译 |
| `video.vtt` | 同 srt,WebVTT 格式 | 网页播放器(需 `--formats` 指定) |
| `video.json` | 含**词级时间戳**的完整结果 | 精修轴、做卡拉OK字幕(需 `--formats` 指定) |

> 播放时把 `.srt` 和视频同名放一起,多数播放器(VLC / PotPlayer / mpv)会自动加载。

## 性能参考

3090Ti(24GB)、半小时视频的**粗略**耗时(实际随语速、配置浮动):

| 阶段 | 配置 | 大致耗时 |
|------|------|----------|
| ASR | `kotoba-whisper` / `float16` / beam 5 | 约 3~6 分钟 |
| ASR | `float32` / beam 10(极致质量) | 约 8~15 分钟 |
| 翻译 | Qwen2.5-14B `Q5_K_M` | 约 2~4 分钟 |
| 翻译 | Qwen2.5-32B `Q4_K_M` | 约 5~10 分钟 |

> 首次运行额外加上模型下载时间。VAD 会跳过静音段,留白多的视频更快。

---

## 故障排查

- **`libcudnn ... cannot open shared object file`**
  说明 CUDA 库没在搜索路径里。`setup_wsl.sh` 末尾给了 `LD_LIBRARY_PATH` 的设置命令,
  把它加进 `.venv/bin/activate` 末尾即可长期生效。

- **`nvidia-smi` 在 WSL 里找不到**
  Windows 端装/更新 NVIDIA 驱动,然后在 Windows PowerShell 执行 `wsl --shutdown` 重启 WSL。

- **显存不够**(同时跑别的任务时)
  改 `--compute-type int8_float16`,或换 `large-v3-turbo`。

- **翻译报"连不上 llama-server"**
  确认服务在跑:`curl http://localhost:8080/health` 应返回 `{"status":"ok"}`。
  没起来就按"部署第 3 步"启动 `llama-server`。

- **`llama-server` 报显存不足 / OOM**
  换更小量化(`Q4_K_M`),或调小 `-ngl`(部分层放 CPU),或减小 `-c` 上下文。

- **ASR 和翻译抢显存**
  分两步跑(先全部 transcribe,再 translate),别让 whisper 和 LLM 同时占显存。
  `run.sh` 就是先转写完再翻译,天然错开;翻译阶段前可临时停掉占显存的进程。

- **kotoba 模型下载慢/失败**
  设镜像:`export HF_ENDPOINT=https://hf-mirror.com` 后重试,或先用 `-m large-v3`。
