# 详细用法

> 环境部署见 [DEPLOY_WSL_CUDA.md](DEPLOY_WSL_CUDA.md);遇到问题见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。

每次使用前激活虚拟环境:
```bash
cd ~/jp_asr && source .venv/bin/activate
```

---

## 推荐工作流(两阶段)

因为 ASR 与翻译会**抢显存**(20GB 翻译模型 + 24GB 卡余量很小),长视频建议分两步:

```bash
# 阶段 1:批量转写(llama-server 不开),带 BGM 的视频调低 VAD 阈值防漏段
python transcribe.py /mnt/d/videos -o ~/out --model large-v3 --vad-threshold 0.2

# 阶段 2:启动 llama-server 后,批量翻译
python translate.py ~/out --bilingual --host http://localhost:8080
```

显存富余(小模型)时也可用一条龙 `run.sh`(它先转写完再翻译,天然错开):
```bash
bash run.sh ./videos ./out
```

---

## 1. 转写 transcribe.py

```bash
python transcribe.py video.mp4                              # 单个文件
python transcribe.py ./videos -o ./out --formats srt,txt,vtt,json   # 批量 + 全格式
python transcribe.py video.mp4 --vad-threshold 0.2         # 带 BGM 防漏段
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `inputs` | 文件或目录(可多个;目录取其中媒体文件) | 必填 |
| `-o, --outdir` | 输出目录 | 与输入同目录 |
| `-m, --model` | 模型,见下 | `kotoba-tech/kotoba-whisper-v2.0-faster` |
| `--compute-type` | `float16` / `int8_float16`(省显存) / `float32`(极致) | `float16` |
| `--beam-size` | beam 宽度,越大越准越慢 | `5` |
| `--formats` | `srt,txt,vtt,json`(json 含词级时间戳) | `srt,txt` |
| `--language` | 语种,日语=`ja` | `ja` |
| `--vad-threshold` | VAD 语音判定阈值,越低越不易漏说话段 | `0.5` |
| `--no-vad` | 完全关闭 VAD(纯静音处可能出幻觉) | 关 |
| `--no-condition` | 关 `condition_on_previous_text`,减少长音频跳段/重复 | 关 |
| `--overwrite` | 已有结果也重新跑 | 关 |
| `--keep-wav` | 保留抽取的 wav | 关 |

**模型选择**:
- `kotoba-tech/kotoba-whisper-v2.0-faster` —— 日语快且准,但**长音频偶尔丢段**(蒸馏模型)。
- `large-v3` —— 更稳、长音频完整性更好,**追求质量首选**。
- `large-v3-turbo` —— 更快,质量略降。

> 带 BGM 的视频漏说话段,多半是 VAD 而非模型,见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
> 的「转写漏段」一节。极致质量可加 `--compute-type float32 --beam-size 10`。

---

## 2. 翻译 translate.py

需先启动 `llama-server`(见 DEPLOY_WSL_CUDA.md),再:
```bash
python translate.py video.srt --bilingual                      # 中文 + 双语
python translate.py ./out --bilingual                          # 批量目录
python translate.py ./out --host http://192.168.10.130:8080    # 翻译服务在别的机器
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `inputs` | `.srt` 文件或目录 | 必填 |
| `-o, --outdir` | 输出目录 | 与输入同目录 |
| `-m, --model` | 模型名(llama-server 单模型时被忽略) | `local-gguf` |
| `--host` | llama-server 地址 | `http://localhost:8080` |
| `--batch` | 每批翻译多少条字幕 | `20` |
| `--bilingual` | 同时输出中日双语 | 关 |
| `--only-bilingual` | 只出双语,不出纯中文 | 关 |

> 换翻译模型不在这里改,而是用不同 GGUF 重启 `llama-server`。
> 自动跳过自己生成的 `.zh.srt` / `.bilingual.srt`,不会重复翻译。
> 单批 prompt 太大撞上下文,就把 `--batch` 调小(已有逐条回退兜底)。

---

## 输出文件

以 `video.mp4` 为例:

| 文件 | 内容 | 用途 |
|------|------|------|
| `video.srt` | 日文字幕,带时间轴 | 原文校对、二次编辑 |
| `video.txt` | 日文纯文本(无时间轴) | 全文阅读 |
| `video.zh.srt` | 中文字幕,时间轴同 `video.srt` | 配视频看的中文字幕 |
| `video.bilingual.srt` | 中日双语(中上日下) | 对照学习 / 校对 |
| `video.vtt` | WebVTT 格式(需 `--formats` 指定) | 网页播放器 |
| `video.json` | 含词级时间戳的完整结果(需 `--formats` 指定) | 精修轴 / 卡拉OK字幕 |

> 播放时把 `.srt` 和视频同名放一起,VLC / PotPlayer / mpv 会自动加载。

---

## 性能参考(3090Ti 24GB,半小时视频,粗略)

| 阶段 | 配置 | 大致耗时 |
|------|------|----------|
| ASR | `kotoba-whisper` / `float16` / beam 5 | 约 3~6 分钟 |
| ASR | `large-v3` / `float16` | 约 5~10 分钟 |
| 翻译 | Qwen 14B `Q5_K_M` | 约 2~4 分钟 |
| 翻译 | Qwen 32B `Q4_K_M` | 约 5~10 分钟 |

> 首次运行另加模型下载时间。从 `/mnt/d`(Windows 盘)读取会更慢,模型建议放 WSL 原生盘。

---

## 远程运行(从 Mac 操作那台 WSL 机)

```bash
ssh 用户名@192.168.10.130
tmux new -s asr            # 长任务用 tmux,断开 SSH 也不中断(回来:tmux attach -t asr)
cd ~/jp_asr && source .venv/bin/activate
python transcribe.py ...
```
