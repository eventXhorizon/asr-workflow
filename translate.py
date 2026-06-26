#!/usr/bin/env python3
"""把 transcribe.py 产出的 srt 字幕用本地 LLM(llama.cpp)翻译成中文。

输入 xxx.srt -> 输出:
  xxx.zh.srt        纯中文字幕(时间轴原样保留)
  xxx.bilingual.srt 中日双语(中文在上,日文在下)

翻译走本地 llama.cpp 的 llama-server(OpenAI 兼容接口 /v1/chat/completions),
离线、免费。模型由 llama-server 启动时加载的 GGUF 决定,本脚本不管模型文件。
长字幕分批送,保证序号与时间轴严格对齐;单批失败自动回退逐条翻译,不丢段。

先启动服务,例如(-hf 会自动从 HuggingFace 下载并缓存 GGUF):
  llama-server -hf Qwen/Qwen2.5-14B-Instruct-GGUF:Q5_K_M -ngl 99 -c 8192 \
               --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# llama-server 单模型服务:model 字段会被忽略,填什么都行,仅用于日志可读性。
DEFAULT_MODEL = "local-gguf"
DEFAULT_HOST = "http://localhost:8080"  # llama-server 默认端口

log = logging.getLogger("jp_asr.translate")


@dataclass
class Cue:
    index: int
    start: str          # 原始 srt 时间戳字符串,原样回写
    end: str
    text: str           # 原文(可能多行,这里已合并为单行)


# --------------------------------------------------------------------------- #
# srt 解析 / 写出
# --------------------------------------------------------------------------- #
TIME_RE = re.compile(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})")


def parse_srt(path: Path) -> list[Cue]:
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        # 找时间轴行(有的 srt 第一行是序号,有的没有)
        t_idx, m = next(((i, mm) for i, ln in enumerate(lines)
                         if (mm := TIME_RE.search(ln))), (None, None))
        if t_idx is None or m is None:
            continue
        text = " ".join(ln.strip() for ln in lines[t_idx + 1:]).strip()
        cues.append(Cue(index=len(cues) + 1, start=m.group(1), end=m.group(2), text=text))
    return cues


def write_srt(cues: list[Cue], texts: list[str], path: Path) -> None:
    out: list[str] = []
    for i, (c, t) in enumerate(zip(cues, texts), 1):
        out += [str(i), f"{c.start} --> {c.end}", t.strip(), ""]
    path.write_text("\n".join(out), encoding="utf-8")


def write_bilingual_srt(cues: list[Cue], zh: list[str], path: Path) -> None:
    out: list[str] = []
    for i, (c, t) in enumerate(zip(cues, zh), 1):
        out += [str(i), f"{c.start} --> {c.end}", t.strip(), c.text.strip(), ""]
    path.write_text("\n".join(out), encoding="utf-8")


# --------------------------------------------------------------------------- #
# llama.cpp (llama-server) 调用 —— OpenAI 兼容接口
# --------------------------------------------------------------------------- #
def llm_chat(host: str, model: str, system: str, user: str, json_mode: bool,
             max_tokens: int | None = None) -> str:
    payload = {
        "model": model,                 # llama-server 单模型,此字段被忽略
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "temperature": 0.2,
        # 客户端强制关闭 Qwen3 思考(需服务端带 --jinja);避免推理吃光输出
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if max_tokens:                      # 上限,防止模型跑飞撑满上下文被截断
        payload["max_tokens"] = max_tokens
    if json_mode:
        # llama.cpp 支持 OpenAI 风格的 json_object,内部用 GBNF 约束,保证合法 JSON
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return strip_think(data["choices"][0]["message"]["content"])


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """剥掉 Qwen3 等推理模型可能输出的 <think>…</think> 块,只留正文。"""
    return _THINK_RE.sub("", text).strip()


SYSTEM_PROMPT = (
    "你是专业的日译中字幕翻译。把日文台词翻译成自然、口语化的简体中文,"
    "贴合视频字幕语境,简洁不啰嗦。保留专有名词/人名的通用译法。"
    "不要解释,不要加注释,只输出译文。"
)


def translate_batch(host: str, model: str, cues: list[Cue]) -> list[str]:
    """整批翻译,要求模型按编号返回 JSON,严格对齐;失败则逐条回退。"""
    numbered = "\n".join(f"{i + 1}. {c.text}" for i, c in enumerate(cues))
    user = (
        "把下面每一条日文字幕翻译成中文。"
        '严格返回 JSON,格式为 {"1":"译文","2":"译文",...},'
        "键为原编号字符串,值为对应中文译文,数量必须和输入完全一致。\n\n"
        + numbered
    )
    # 译文通常每条几十 token,留足余量即可;上限防止模型啰嗦撑爆每槽上下文
    cap = max(256, len(cues) * 120)
    try:
        raw = llm_chat(host, model, SYSTEM_PROMPT, user, json_mode=True, max_tokens=cap)
        obj = json.loads(raw)
        result = [str(obj.get(str(i + 1), "")).strip() for i in range(len(cues))]
        if all(result):  # 全部有译文才算成功
            return result
        log.warning("批量结果有缺失,回退逐条翻译该批 (%d 条)", len(cues))
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as e:
        log.warning("批量翻译失败(%s),回退逐条翻译该批 (%d 条)", e, len(cues))

    # 回退:逐条翻译,保证不丢段、不错位
    out: list[str] = []
    for c in cues:
        try:
            t = llm_chat(host, model, SYSTEM_PROMPT,
                         f"把这句日文翻译成中文,只输出译文:\n{c.text}",
                         json_mode=False, max_tokens=256)
        except urllib.error.URLError as e:
            log.error("单条翻译失败,保留原文: %s", e)
            t = c.text
        out.append(t.strip())
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def check_server(host: str) -> None:
    try:
        with urllib.request.urlopen(f"{host}/health", timeout=10) as resp:
            status = json.loads(resp.read().decode("utf-8")).get("status")
    except urllib.error.HTTPError as e:
        # llama-server 加载模型时 /health 返回 503
        if e.code == 503:
            log.error("llama-server 正在加载模型(HTTP 503),等加载完再跑。")
        else:
            log.error("llama-server 返回 HTTP %s。", e.code)
        sys.exit(1)
    except urllib.error.URLError:
        log.error("连不上 llama-server (%s)。先在 WSL 启动,例如:\n"
                  "  bash start_llama.sh /path/to/model.gguf",
                  host)
        sys.exit(1)
    if status != "ok":
        log.warning("llama-server 状态异常: %s", status)


def translate_file(src: Path, args) -> None:
    cues = parse_srt(src)
    if not cues:
        log.warning("没有解析到字幕: %s", src.name)
        return
    log.info("翻译 %s,共 %d 条,分批 %d…", src.name, len(cues), args.batch)

    # 分批 + 并发(llama-server 能同时处理多个请求,串行会让 GPU 空等)
    batches = [(i, cues[i:i + args.batch]) for i in range(0, len(cues), args.batch)]
    results: dict[int, list[str]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(translate_batch, args.host, args.model, chunk): i
                for i, chunk in batches}
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += len(results[i])
            log.info("  进度 %d/%d", done, len(cues))
    zh: list[str] = []
    for i, _ in batches:                      # 按原顺序拼回,保证时间轴对齐
        zh.extend(results[i])

    stem = src.name[:-len(".srt")] if src.name.endswith(".srt") else src.stem
    out_dir = args.outdir or src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bilingual:
        p = out_dir / f"{stem}.bilingual.srt"
        write_bilingual_srt(cues, zh, p)
        log.info("写出 %s", p.name)
    if not args.only_bilingual:
        p = out_dir / f"{stem}.zh.srt"
        write_srt(cues, zh, p)
        log.info("写出 %s", p.name)


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(
        description="用本地 llama.cpp(llama-server)把日文 srt 字幕翻译成中文",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help=".srt 文件或目录")
    ap.add_argument("-o", "--outdir", type=Path, default=None)
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL,
                    help="模型名(llama-server 单模型时此字段被忽略,仅记录用)")
    ap.add_argument("--host", default=DEFAULT_HOST, help="llama-server 地址")
    ap.add_argument("--batch", type=int, default=10,
                    help="每批翻译多少条字幕(太大易超出每槽上下文被截断)")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="并发请求数(对应 llama-server 的 -np 槽位,越大越快)")
    ap.add_argument("--bilingual", action="store_true", help="同时输出中日双语字幕")
    ap.add_argument("--only-bilingual", action="store_true", help="只输出双语,不输出纯中文")
    args = ap.parse_args(argv)

    if args.only_bilingual:
        args.bilingual = True

    # 收集 srt(跳过自己生成的 .zh.srt / .bilingual.srt)
    files: list[Path] = []
    for p in args.inputs:
        if p.is_dir():
            files += sorted(f for f in p.iterdir()
                            if f.suffix.lower() == ".srt"
                            and not f.name.endswith((".zh.srt", ".bilingual.srt")))
        elif p.is_file():
            files.append(p)
        else:
            log.warning("路径不存在: %s", p)
    if not files:
        log.error("没有找到 .srt 文件。")
        return 1

    check_server(args.host)

    failed = 0
    for i, f in enumerate(files, 1):
        log.info("[%d/%d] %s", i, len(files), f.name)
        try:
            translate_file(f, args)
        except Exception:
            log.exception("翻译失败: %s", f)
            failed += 1
    log.info("结束。成功 %d,失败 %d。", len(files) - failed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
