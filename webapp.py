#!/usr/bin/env python3
"""jp_asr Web 后台 (Gradio)。

浏览器提交任务 → 后台单 worker 串行处理(显存互斥天然满足):
  阶段1 转写: 关闭 llama-server 腾显存 → 调 transcribe.py
  阶段2 翻译: 拉起 llama-server → 调 translate.py
任务进度/历史持久化到 jobs.json,产物可在网页下载。

启动(在 venv 里):
  LLAMA_MODEL=~/models/xxx.gguf \
  LLAMA_BIN=~/llama.cpp/build/bin/llama-server \
  python webapp.py
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import gradio as gr

# --------------------------------------------------------------------------- #
# 配置(均可用环境变量覆盖)
# --------------------------------------------------------------------------- #
HOME = Path.home()
PROJECT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable                       # 当前 venv 的 python

MODEL_PATH = os.environ.get("LLAMA_MODEL", "")            # 翻译模型 gguf(必填)
LLAMA_BIN = os.environ.get("LLAMA_BIN", "llama-server")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
LLAMA_NGL = os.environ.get("LLAMA_NGL", "99")
LLAMA_CTX = os.environ.get("LLAMA_CTX", "16384")
LLAMA_EXTRA = os.environ.get("LLAMA_EXTRA", "--jinja --reasoning-budget 0 -fa on")
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}"

OUT_DIR = Path(os.environ.get("JP_ASR_OUT", str(HOME / "out")))
VIDEO_ROOT = os.environ.get("JP_ASR_VIDEO_ROOT", "/mnt/d")
JOBS_FILE = OUT_DIR / "jobs.json"
WEB_PORT = int(os.environ.get("JP_ASR_WEB_PORT", "7860"))

OUT_DIR.mkdir(parents=True, exist_ok=True)

STATUS_ZH = {"queued": "⏳ 排队中", "asr": "🎙 转写中", "translate": "🌐 翻译中",
             "done": "✅ 完成", "error": "❌ 失败", "interrupted": "⚠ 已中断"}


# --------------------------------------------------------------------------- #
# 任务模型 + 持久化
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    video: str
    model: str
    vad_threshold: float
    no_vad: bool
    bilingual: bool
    formats: str
    status: str = "queued"
    detail: str = ""
    outputs: list = field(default_factory=list)
    created: str = ""
    finished: str = ""


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
work_q: "queue.Queue[str]" = queue.Queue()


def _save() -> None:
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(j) for j in JOBS.values()], f, ensure_ascii=False, indent=2)


def _load() -> None:
    if not JOBS_FILE.exists():
        return
    for d in json.loads(JOBS_FILE.read_text(encoding="utf-8")):
        JOBS[d["id"]] = Job(**d)


def update(job_id: str, **changes) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        for k, v in changes.items():
            setattr(job, k, v)
        _save()


def jobdir(job_id: str) -> Path:
    return OUT_DIR / job_id


# --------------------------------------------------------------------------- #
# llama-server 托管
# --------------------------------------------------------------------------- #
_llama_proc: subprocess.Popen | None = None
_llama_lock = threading.Lock()


def llama_running() -> bool:
    try:
        with urllib.request.urlopen(f"{LLAMA_URL}/health", timeout=3) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def start_llama(logf) -> None:
    global _llama_proc
    with _llama_lock:
        if llama_running():
            return
        cmd = [LLAMA_BIN, "-m", MODEL_PATH, "-ngl", LLAMA_NGL, "-c", LLAMA_CTX,
               "--host", "0.0.0.0", "--port", str(LLAMA_PORT), *shlex.split(LLAMA_EXTRA)]
        logf.write(f"\n[llama] 启动: {' '.join(cmd)}\n"); logf.flush()
        _llama_proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        for _ in range(180):                      # 最多等 6 分钟(首载较慢)
            if llama_running():
                logf.write("[llama] 就绪\n"); logf.flush()
                return
            if _llama_proc.poll() is not None:
                raise RuntimeError("llama-server 进程退出,见日志")
            time.sleep(2)
        raise RuntimeError("llama-server 启动超时")


def stop_llama(logf) -> None:
    global _llama_proc
    with _llama_lock:
        if _llama_proc and _llama_proc.poll() is None:
            logf.write("[llama] 停止以腾出显存\n"); logf.flush()
            _llama_proc.terminate()
            try:
                _llama_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _llama_proc.kill()
            _llama_proc = None
            time.sleep(2)                         # 等显存释放
        elif llama_running():
            logf.write("[llama] 警告: 检测到外部 llama-server 在跑,无法停止,"
                       "转写可能 OOM。请关掉它或交给本程序托管。\n"); logf.flush()


# --------------------------------------------------------------------------- #
# 后台 worker
# --------------------------------------------------------------------------- #
def run_transcribe(job: Job, jd: Path, logf) -> Path:
    cmd = [PYTHON, str(PROJECT_DIR / "transcribe.py"), job.video,
           "-o", str(jd), "--model", job.model, "--formats", job.formats, "--overwrite"]
    if job.no_vad:
        cmd.append("--no-vad")
    else:
        cmd += ["--vad-threshold", str(job.vad_threshold)]
    logf.write(f"\n[转写] {' '.join(cmd)}\n"); logf.flush()
    subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(PROJECT_DIR), check=True)
    srt = jd / (Path(job.video).stem + ".srt")
    if not srt.exists():
        raise RuntimeError("转写完成但未找到 srt(可能整段无语音)")
    return srt


def run_translate(job: Job, srt: Path, logf) -> None:
    cmd = [PYTHON, str(PROJECT_DIR / "translate.py"), str(srt), "--host", LLAMA_URL]
    if job.bilingual:
        cmd.append("--bilingual")
    logf.write(f"\n[翻译] {' '.join(cmd)}\n"); logf.flush()
    subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(PROJECT_DIR), check=True)


def collect_outputs(jd: Path) -> list[str]:
    return sorted(str(p) for p in jd.iterdir()
                  if p.suffix in {".srt", ".txt", ".vtt", ".json"})


def worker() -> None:
    while True:
        job_id = work_q.get()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job or job.status not in ("queued", "interrupted"):
            work_q.task_done()
            continue
        jd = jobdir(job_id)
        jd.mkdir(parents=True, exist_ok=True)
        with open(jd / "log.txt", "a", encoding="utf-8") as logf:
            try:
                update(job_id, status="asr", detail="转写中")
                stop_llama(logf)                              # 腾显存
                srt = run_transcribe(job, jd, logf)

                update(job_id, status="translate", detail="翻译中")
                start_llama(logf)                             # 拉起翻译后端
                run_translate(job, srt, logf)

                update(job_id, status="done", detail="",
                       outputs=collect_outputs(jd),
                       finished=datetime.now().strftime("%m-%d %H:%M"))
            except subprocess.CalledProcessError as e:
                logf.write(f"\n[ERROR] 子进程失败,退出码 {e.returncode}\n")
                update(job_id, status="error", detail=f"子进程失败({e.returncode})")
            except Exception as e:
                logf.write(f"\n[ERROR] {e}\n")
                update(job_id, status="error", detail=str(e))
        work_q.task_done()


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #
def enqueue(upload, video, fe_path, model, vad, no_vad, bilingual, formats):
    # 优先级:上传的文件 > 手填路径 > 浏览选择
    path = (upload or video or fe_path or "").strip()
    if not path:
        return "⚠ 请上传视频、或填写/选择服务器上的视频路径", *refresh()
    if not Path(path).exists():
        return f"⚠ 文件不存在: {path}", *refresh()
    fmts = ",".join(formats) if formats else "srt,txt"
    job = Job(
        id=datetime.now().strftime("%y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
        video=path, model=model, vad_threshold=float(vad), no_vad=bool(no_vad),
        bilingual=bool(bilingual), formats=fmts,
        created=datetime.now().strftime("%m-%d %H:%M"),
    )
    with JOBS_LOCK:
        JOBS[job.id] = job
        _save()
    work_q.put(job.id)
    return f"✅ 已加入队列: {Path(path).name}  (任务 {job.id})", *refresh()


def refresh():
    with JOBS_LOCK:
        rows = [[j.id, Path(j.video).name, STATUS_ZH.get(j.status, j.status),
                 j.detail, j.created, j.finished]
                for j in sorted(JOBS.values(), key=lambda x: x.created, reverse=True)]
    badge = "🟢 llama-server 运行中" if llama_running() else "⚪ llama-server 已停止"
    return rows, badge


def view_job(job_id):
    job_id = (job_id or "").strip()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return "请选择/填写有效的任务 ID", None
    log_path = jobdir(job_id) / "log.txt"
    log = log_path.read_text(encoding="utf-8")[-8000:] if log_path.exists() else "(无日志)"
    files = job.outputs if job.status == "done" else None
    return log, files


def build_ui() -> gr.Blocks:
    miss = "" if MODEL_PATH else "⚠ 未设置 LLAMA_MODEL 环境变量,翻译阶段会失败!"
    with gr.Blocks(title="jp_asr 字幕后台") as demo:
        gr.Markdown(f"# 🎬 jp_asr 日语视频转中文字幕\n{miss}")
        llama_status = gr.Markdown("⚪ llama-server 已停止")

        with gr.Tab("新建任务"):
            upload = gr.File(label="① 上传视频(从本机/能访问的网络位置选择;适合在别的电脑上的视频)",
                             file_types=[".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m4v"],
                             type="filepath")
            with gr.Row():
                video = gr.Textbox(label="② 或填服务器上的视频路径",
                                   placeholder="/mnt/d/xxx.mp4(可直接粘贴)")
            fe = gr.FileExplorer(label=f"③ 或从 {VIDEO_ROOT} 浏览选择(.mp4)",
                                 root_dir=VIDEO_ROOT, glob="**/*.mp4",
                                 file_count="single", height=200)
            with gr.Row():
                model = gr.Dropdown(
                    ["large-v3", "kotoba-tech/kotoba-whisper-v2.0-faster", "large-v3-turbo"],
                    value="large-v3", label="ASR 模型")
                vad = gr.Slider(0.0, 0.6, value=0.2, step=0.05,
                                label="VAD 阈值(越低越不漏说话段,带 BGM 用 0.2)")
            with gr.Row():
                no_vad = gr.Checkbox(label="完全关闭 VAD", value=False)
                bilingual = gr.Checkbox(label="输出中日双语", value=True)
                formats = gr.CheckboxGroup(["srt", "txt", "vtt", "json"],
                                           value=["srt", "txt"], label="转写输出格式")
            submit = gr.Button("加入队列", variant="primary")
            result = gr.Markdown()

        with gr.Tab("任务队列 / 历史"):
            jobs_df = gr.Dataframe(
                headers=["任务ID", "文件", "状态", "详情", "创建", "完成"],
                datatype=["str"] * 6, interactive=False, wrap=True, label="任务列表")
            with gr.Row():
                job_id_in = gr.Textbox(label="查看任务 ID(从上表复制)", scale=3)
                view_btn = gr.Button("查看日志/产物", scale=1)
            out_files = gr.Files(label="产物下载(完成的任务)")
            log_box = gr.Textbox(label="任务日志(末尾 8000 字)", lines=18, max_lines=18)

        # FileExplorer 选中 → 填到文本框
        fe.change(lambda p: p if isinstance(p, str) else (p[0] if p else ""),
                  inputs=fe, outputs=video)
        submit.click(enqueue,
                     inputs=[upload, video, fe, model, vad, no_vad, bilingual, formats],
                     outputs=[result, jobs_df, llama_status])
        view_btn.click(view_job, inputs=job_id_in, outputs=[log_box, out_files])

        timer = gr.Timer(3.0)
        timer.tick(refresh, outputs=[jobs_df, llama_status])
        demo.load(refresh, outputs=[jobs_df, llama_status])
    return demo


def recover_on_start() -> None:
    """上次未跑完的任务:排队的重新入队,跑到一半的标记中断。"""
    _load()
    for job in JOBS.values():
        if job.status == "queued":
            work_q.put(job.id)
        elif job.status in ("asr", "translate"):
            job.status = "interrupted"
            job.detail = "程序重启,任务中断,可重新提交"
    _save()


def main() -> None:
    recover_on_start()
    threading.Thread(target=worker, daemon=True).start()
    # max_file_size 放开,允许上传大视频(几个 GB)
    build_ui().queue().launch(server_name="0.0.0.0", server_port=WEB_PORT,
                              max_file_size=os.environ.get("JP_ASR_MAX_UPLOAD", "20gb"))


if __name__ == "__main__":
    main()
