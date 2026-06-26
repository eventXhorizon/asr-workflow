#!/usr/bin/env python3
"""日语视频/音频 ASR 转写工具。

流程: 视频 --(ffmpeg)--> 16kHz 单声道 wav --(faster-whisper)--> srt / txt / json

默认使用 kotoba-whisper-v2.0(针对日语调优,质量优于原版 large-v3)。
在 24GB 显存(3090Ti)上可以放心用 float16,beam_size 拉大求质量。
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# 视频/音频后缀(交给 ffmpeg 统一抽轨,所以列得宽一点)
MEDIA_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".ts", ".m4v",
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
}

# 备选模型,方便切换:
#   kotoba-tech/kotoba-whisper-v2.0-faster  日语最佳质量(默认)
#   large-v3                                 通用最强基线,生态最稳
#   large-v3-turbo                           更快,质量略降
DEFAULT_MODEL = "kotoba-tech/kotoba-whisper-v2.0-faster"

log = logging.getLogger("jp_asr")


@dataclass
class Args:
    inputs: list[Path]
    outdir: Path | None
    model: str
    device: str
    compute_type: str
    beam_size: int
    language: str
    formats: set[str]
    keep_wav: bool
    overwrite: bool
    no_vad: bool
    vad_threshold: float
    no_condition: bool


# --------------------------------------------------------------------------- #
# 音频抽取
# --------------------------------------------------------------------------- #
def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        log.error("找不到 ffmpeg。WSL/Ubuntu 安装: sudo apt update && sudo apt install -y ffmpeg")
        sys.exit(1)


def decode_audio(src: Path):
    """ffmpeg 就地读视频,把音频解码进内存(16kHz 单声道 float32),不落临时文件。

    源视频原文件不动、不拷贝;返回的 numpy 数组直接喂给 faster-whisper。
    """
    import numpy as np
    cmd = [
        "ffmpeg", "-nostdin",
        "-i", str(src),
        "-vn",                  # 丢掉视频流
        "-ar", "16000",         # 16kHz
        "-ac", "1",             # 单声道
        "-f", "f32le", "-",     # 裸 float32 PCM 输出到 stdout(管道进内存)
    ]
    log.info("解码音频到内存: %s", src.name)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        log.error("ffmpeg 失败 (%s):\n%s", src.name,
                  proc.stderr.decode("utf-8", "ignore")[-2000:])
        raise RuntimeError(f"ffmpeg failed for {src}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def write_wav(audio, dst: Path) -> None:
    """把内存里的 float32 音频写成 16-bit wav(仅 --keep-wav 时用)。"""
    import wave
    import numpy as np
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16.tobytes())


# --------------------------------------------------------------------------- #
# 输出格式
# --------------------------------------------------------------------------- #
def with_ext(base: Path, ext: str) -> Path:
    """在完整文件名后追加扩展名,避免 with_suffix 误删含点文件名的最后一段。

    例:"a.b.c"(stem)→ "a.b.c.srt",而 with_suffix(".srt") 会得到错误的 "a.b.srt"。
    """
    return base.with_name(base.name + ext)


def fmt_ts(seconds: float, sep: str = ",") -> str:
    """秒 -> HH:MM:SS,mmm (srt) 或 HH:MM:SS.mmm (vtt)。"""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_srt(segments: list[dict], path: Path) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[dict], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{fmt_ts(seg['start'], '.')} --> {fmt_ts(seg['end'], '.')}")
        lines.append(seg["text"].strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(segments: list[dict], path: Path) -> None:
    text = "\n".join(seg["text"].strip() for seg in segments if seg["text"].strip())
    path.write_text(text + "\n", encoding="utf-8")


def write_json(segments: list[dict], info: dict, path: Path) -> None:
    path.write_text(
        json.dumps({"info": info, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# 转写
# --------------------------------------------------------------------------- #
def transcribe_one(model, src: Path, args: Args) -> None:
    out_base = (args.outdir or src.parent) / src.stem
    srt_path = with_ext(out_base,".srt")
    if srt_path.exists() and not args.overwrite:
        log.info("已存在,跳过(用 --overwrite 覆盖): %s", srt_path.name)
        return

    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)

    audio = decode_audio(src)                 # 就地读视频,音频进内存,不落临时文件
    if args.keep_wav:
        write_wav(audio, with_ext(out_base,".wav"))

    log.info("开始转写: %s (vad=%s)", src.name, "off" if args.no_vad else "on")
    seg_iter, info = model.transcribe(
        audio,
        language=args.language,
        beam_size=args.beam_size,
        # VAD 默认开(跳静音);带 BGM 的视频若漏说话段,用 --no-vad 或调低 --vad-threshold
        vad_filter=not args.no_vad,
        vad_parameters=(None if args.no_vad else
                        {"min_silence_duration_ms": 500, "threshold": args.vad_threshold}),
        word_timestamps=("json" in args.formats),
        # 关掉"以前文为条件"可减少长音频跳段/重复(默认开,质量更连贯)
        condition_on_previous_text=not args.no_condition,
    )

    total = round(info.duration, 1)
    log.info("时长 %.1fs,检测语言 %s(p=%.2f),逐段解码中…",
             total, info.language, info.language_probability)

    segments: list[dict] = []
    try:
        from tqdm import tqdm
        bar = tqdm(total=total, unit="s", desc=src.stem[:20], dynamic_ncols=True)
    except ImportError:
        bar = None

    for seg in seg_iter:  # 注意:faster-whisper 是惰性生成器,这里才真正算
        item: dict = {"start": seg.start, "end": seg.end, "text": seg.text}
        if seg.words:
            item["words"] = [
                {"start": w.start, "end": w.end, "word": w.word, "prob": w.probability}
                for w in seg.words
            ]
        segments.append(item)
        if bar:
            bar.n = min(seg.end, total)
            bar.refresh()
    if bar:
        bar.n = total
        bar.close()

    if not segments:
        log.warning("没有解码出任何语音段: %s", src.name)

    info_dict = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "model": args.model,
    }
    written = []
    if "srt" in args.formats:
        write_srt(segments, srt_path); written.append(srt_path.name)
    if "vtt" in args.formats:
        p = with_ext(out_base,".vtt"); write_vtt(segments, p); written.append(p.name)
    if "txt" in args.formats:
        p = with_ext(out_base,".txt"); write_txt(segments, p); written.append(p.name)
    if "json" in args.formats:
        p = with_ext(out_base,".json"); write_json(segments, info_dict, p); written.append(p.name)

    log.info("完成 %s -> %s", src.name, ", ".join(written))


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def collect_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir()
                                if f.is_file() and f.suffix.lower() in MEDIA_SUFFIXES))
        elif p.is_file():
            files.append(p)
        else:
            log.warning("路径不存在,跳过: %s", p)
    return files


def parse_args(argv: list[str]) -> Args:
    ap = argparse.ArgumentParser(
        description="日语视频/音频 ASR 转写 (faster-whisper + kotoba-whisper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", type=Path, help="视频/音频文件或目录(目录递归一层)")
    ap.add_argument("-o", "--outdir", type=Path, default=None, help="输出目录(默认与输入同目录)")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL,
                    help="模型: kotoba-tech/kotoba-whisper-v2.0-faster | large-v3 | large-v3-turbo")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--compute-type", default="float16",
                    help="cuda: float16(默认) / int8_float16(省显存) / float32(极致质量); cpu: int8")
    ap.add_argument("--beam-size", type=int, default=5, help="beam search 宽度,越大越准越慢")
    ap.add_argument("--language", default="ja", help="语种代码,日语=ja")
    ap.add_argument("--formats", default="srt,txt",
                    help="输出格式,逗号分隔: srt,txt,vtt,json(json 含词级时间戳)")
    ap.add_argument("--keep-wav", action="store_true", help="保留抽取的 wav")
    ap.add_argument("--overwrite", action="store_true", help="已有结果也重新转写")
    ap.add_argument("--no-vad", action="store_true",
                    help="关闭 VAD(带 BGM/漏说话段时用;但纯静音处可能出幻觉)")
    ap.add_argument("--vad-threshold", type=float, default=0.5,
                    help="VAD 语音判定阈值,越低越不容易把说话当噪音删(默认 0.5,漏段可试 0.2)")
    ap.add_argument("--no-condition", action="store_true",
                    help="关闭 condition_on_previous_text,减少长音频跳段/重复")
    a = ap.parse_args(argv)

    formats = {f.strip().lower() for f in a.formats.split(",") if f.strip()}
    bad = formats - {"srt", "txt", "vtt", "json"}
    if bad:
        ap.error(f"不支持的格式: {', '.join(bad)}")

    return Args(
        inputs=a.inputs, outdir=a.outdir, model=a.model, device=a.device,
        compute_type=a.compute_type, beam_size=a.beam_size, language=a.language,
        formats=formats, keep_wav=a.keep_wav, overwrite=a.overwrite,
        no_vad=a.no_vad, vad_threshold=a.vad_threshold, no_condition=a.no_condition,
    )


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    ensure_ffmpeg()

    files = collect_inputs(args.inputs)
    if not files:
        log.error("没有找到可处理的媒体文件。")
        return 1
    log.info("待处理 %d 个文件,加载模型 %s (%s/%s)…",
             len(files), args.model, args.device, args.compute_type)

    from faster_whisper import WhisperModel  # 延迟导入:无 GPU 环境也能看 --help

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    failed = 0
    for i, f in enumerate(files, 1):
        log.info("[%d/%d] %s", i, len(files), f.name)
        try:
            transcribe_one(model, f, args)
        except Exception:  # 单个文件失败不影响整体批处理
            log.exception("处理失败: %s", f)
            failed += 1

    log.info("全部结束。成功 %d,失败 %d。", len(files) - failed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
