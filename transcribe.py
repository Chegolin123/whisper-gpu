#!/usr/bin/env python3
"""Расшифровка аудио/видео через faster-whisper.

    python transcribe.py lecture.mp4
    python transcribe.py lecture.mp4 --model large-v3 --language ru --device cuda

Пишет рядом с исходником: .txt, .srt, .json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


def fmt_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> int:
    ap = argparse.ArgumentParser(description="faster-whisper → txt/srt/json")
    ap.add_argument("media", type=Path, help="аудио или видео файл")
    ap.add_argument("--model", default=os.getenv("WHISPER_MODEL", "medium"))
    ap.add_argument("--model-path", default=os.getenv("WHISPER_MODEL_PATH"),
                    help="локальная папка с моделью, если HF недоступен")
    ap.add_argument("--language", default=None, help="ru/en/...; по умолчанию авто")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute-type", default=None,
                    help="float16 / int8_float16 / int8; по умолчанию под устройство")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if not args.media.exists():
        print(f"нет файла: {args.media}", file=sys.stderr)
        return 1

    compute = args.compute_type or ("float16" if args.device == "cuda" else "int8")

    t0 = time.time()
    model = WhisperModel(
        args.model_path or args.model,
        device=args.device,
        compute_type=compute,
    )
    print(f"[model] {args.model_path or args.model} на {args.device}/{compute} "
          f"за {time.time() - t0:.1f} c", flush=True)

    t1 = time.time()
    segments, info = model.transcribe(
        str(args.media),
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    out_dir = args.out_dir or args.media.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.media.stem

    segs: list[dict] = []
    for i, s in enumerate(segments, 1):
        text = s.text.strip()
        segs.append({"id": i, "start": round(s.start, 2), "end": round(s.end, 2), "text": text})
        print(f"[{s.start:8.2f} -> {s.end:8.2f}] {text}", flush=True)

    (base.with_suffix(".txt")).write_text("\n".join(x["text"] for x in segs), encoding="utf-8")
    with base.with_suffix(".srt").open("w", encoding="utf-8") as f:
        for x in segs:
            f.write(f"{x['id']}\n{fmt_ts(x['start'])} --> {fmt_ts(x['end'])}\n{x['text']}\n\n")
    base.with_suffix(".json").write_text(
        json.dumps({"language": info.language, "segments": segs}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    dt = time.time() - t1
    dur = getattr(info, "duration", 0) or 0
    rtf = dt / dur if dur else 0
    print(f"\n[done] {len(segs)} сегментов за {dt:.0f} c, "
          f"язык {info.language}, RTF {rtf:.2f}")
    print(f"txt: {base.with_suffix('.txt')}\nsrt: {base.with_suffix('.srt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
