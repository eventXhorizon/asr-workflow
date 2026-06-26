# 在 Windows WSL2 + NVIDIA 上部署 llama.cpp(CUDA 版)实录

本文记录在 **Windows WSL2(Ubuntu)+ NVIDIA RTX 3090Ti** 上,把 `llama-server`
跑在 GPU 上的完整过程。结论先行:

> **官方 Linux 预编译包没有 CUDA 版,WSL 里的 Vulkan 也用不了 → 必须自己编译 CUDA 版。**

## 0. 前置确认

WSL 里能看到显卡(Windows 端装了最新 NVIDIA 驱动,自带 WSL CUDA 直通):
```bash
nvidia-smi          # 能看到 RTX 3090Ti 即可
```

---

## 1. 为什么要自己编译(踩坑结论)

试过两条更省事的路,都不通:

- **官方预编译包**:Linux 只提供 CPU / Vulkan / ROCm / OpenVINO / SYCL,**没有 CUDA**
  (CUDA 预编译只有 Windows 版)。
- **Vulkan 版**:下下来 `device_info` 只认到 CPU。`vulkaninfo --summary` 显示只有
  `llvmpipe`(软件渲染),**没有 NVIDIA 设备**——NVIDIA 的 WSL 驱动只透传了 CUDA,
  没有 Linux 侧的 Vulkan ICD。所以 Vulkan 在这套环境里是死路。

`nvidia-smi` 正常 = CUDA 驱动在(`/usr/lib/wsl/lib/libcuda`),因此**编译 CUDA 版**是
唯一稳的路,性能也最好。

---

## 2. 修好网络/代理(否则装不了东西)

本机踩到的坑:apt 配了个连不上的代理 `192.168.10.130:7897`,导致一切下载失败。

**修 apt 代理**(改直连;tuna 等国内镜像本就不需要代理):
```bash
grep -rin proxy /etc/apt/apt.conf.d/ /etc/apt/apt.conf 2>/dev/null   # 找到坏代理在哪
sudo sed -i '/192\.168\.10\.130/s/^/# /' /etc/apt/apt.conf.d/* 2>/dev/null   # 注释掉
sudo apt update                                                       # 不再报代理错即可
```

**git 走代理**(克隆 github 用;mirrored 模式下 Clash 在 `127.0.0.1`):
```bash
curl -x http://127.0.0.1:7897 -I https://github.com   # 先测代理通不通
git config --global http.proxy  http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```
> 连 `127.0.0.1:7897` 也被拒,就去 Clash 打开「Allow LAN / 允许局域网」。

---

## 3. 装编译工具链 + CUDA toolkit

直接用 Ubuntu 仓库的 `nvidia-cuda-toolkit`(提供 `nvcc`),从国内镜像直连可下,
不用折腾 NVIDIA 官方源:
```bash
sudo apt install -y build-essential cmake git libcurl4-openssl-dev nvidia-cuda-toolkit
nvcc --version      # 本机得到 CUDA 12.4 —— 能打印版本号就 OK
```
> 运行时用的 `libcuda` 是 WSL 现成的;`nvcc` 只在编译时用。

---

## 4. 编译 CUDA 版 llama.cpp(关键步骤)

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build --config Release -j
```

- `-DGGML_CUDA=ON`:启用 CUDA 后端(核心开关)。
- `-DCMAKE_CUDA_ARCHITECTURES="86"`:3090Ti 算力 8.6,**只编这一档**,编译快很多。
  (其它卡:4090=89,A100=80,可按需改)
- `-j`:并行编译,会跑满 CPU,耗时几分钟到十几分钟。
- 产物:`~/llama.cpp/build/bin/llama-server`。

**编译过程中的正常现象**:会看到一段 `UI: npm install failed` 的报错——这是 Web UI
资源构建失败,**非致命**,它会自动从 HuggingFace 下载预编译好的 UI 继续。只要最后出现
`[100%] Built target llama-server` 就是成功。

---

## 5. 启动服务

```bash
cd ~/llama.cpp
./build/bin/llama-server \
  -m ~/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  -ngl 99 -c 4096 --host 0.0.0.0 --port 8080 --jinja --reasoning-budget 0
```

启动开头**确认 GPU 被识别**:
```
ggml_cuda_init / device_info:
  - CUDA0 : NVIDIA GeForce RTX 3090 Ti (24563 MiB, 23285 MiB free)
  ... CUDA : ARCHS = 860
```
加载完出现 `server is listening on http://0.0.0.0:8080`。

参数说明:
- `-ngl 99`:全部层放 GPU(显存够就拉满)。
- `-c 4096`:上下文长度(20GB 模型 + 4K 上下文在 24GB 卡上放得下)。
- `--host 0.0.0.0`:允许局域网/其它机器访问(只本机用可省略,默认 127.0.0.1)。
- `--jinja --reasoning-budget 0`:Qwen3 默认开 thinking(日志会显示 `thinking = 1`),
  这两个参数用模板并关掉推理,翻译输出更干净更快。

---

## 6. 模型加载慢的解决:别从 /mnt/d 读

模型放在 Windows 盘(`/mnt/d/...`)时,WSL 通过 drvfs/9P 桥接读取,**带宽很差、加载很慢**。
拷进 WSL 原生文件系统再加载:
```bash
mkdir -p ~/models
cp "/mnt/d/Qwen3.6-...-Q4_K_M.gguf" ~/models/
```
从 `~/models` 加载,且**第二次启动会命中页缓存**(本机 128GB 内存),几乎秒载。
服务是常驻的,加载只在每次启动付一次,别反复重启。

---

## 7. 局域网访问(从 Mac 等其它机器)

本机是 WSL **mirrored 网络模式**(`.wslconfig` 里 `networkingMode=mirrored`),
配合 `--host 0.0.0.0`,其它机器用 Windows 主机 IP 访问:
```bash
# 在 Mac 上:
curl http://192.168.10.130:8080/health      # {"status":"ok"}
```
连不上则在 **Windows PowerShell(管理员)**放行防火墙:
```powershell
New-NetFirewallRule -DisplayName "llama 8080" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8080
# mirrored 模式还有独立的 Hyper-V 防火墙,必要时:
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
```
> 在 WSL 本机用 `curl 192.168.10.130:8080` 测会被 shell 的 `http_proxy` 劫持,
> 要么 `curl --noproxy '*' ...`,要么直接去 Mac 上测才准。
>
> ⚠️ 绑 `0.0.0.0` 等于局域网内无鉴权可调用;需要加锁用 `--api-key <密钥>`。

---

## 8. 验证全链路

```bash
curl http://localhost:8080/health      # GPU 机本机: {"status":"ok"}
nvidia-smi                             # 应看到 llama-server 占 ~20GB 显存
```
就绪后接翻译流水线(见 README):
```bash
cd ~/jp_asr
python translate.py video.srt --bilingual --host http://localhost:8080
```

---

## 速查:从零到跑通

```bash
# 1. 修代理
sudo sed -i '/192\.168\.10\.130/s/^/# /' /etc/apt/apt.conf.d/* 2>/dev/null && sudo apt update
git config --global http.proxy http://127.0.0.1:7897 && git config --global https.proxy http://127.0.0.1:7897
# 2. 工具链
sudo apt install -y build-essential cmake git libcurl4-openssl-dev nvidia-cuda-toolkit
# 3. 编译
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86" && cmake --build build --config Release -j
# 4. 模型拷到本地盘
mkdir -p ~/models && cp "/mnt/d/你的模型.gguf" ~/models/
# 5. 启动
./build/bin/llama-server -m ~/models/你的模型.gguf -ngl 99 -c 4096 \
    --host 0.0.0.0 --port 8080 --jinja --reasoning-budget 0
```
