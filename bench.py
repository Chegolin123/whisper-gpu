#!/usr/bin/env python3
"""Замер скорости расшифровки: RTF и пропускная способность.

    python bench.py sample.mp3 --model medium --device cuda

RTF < 1 значит «быстрее реального времени».
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from faster_whisper import WhisperModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("media", type=Path)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    compute = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    model = WhisperModel(args.model, device=args.device, compute_type=compute)

    total = 0.0
    for i in range(args.runs):
        t0 = time.time()
        segments, info = model.transcribe(str(args.media), vad_filter=True)
        n = sum(1 for _ in segments)
        dt = time.time() - t0
        total += dt
        rtf = dt / info.duration if info.duration else 0
        print(f"run {i + 1}: {dt:.1f} c, RTF {rtf:.2f}, сегментов {n}, "
              f"язык {info.language}")

    avg = total / args.runs
    print(f"\nitogo: среднее {avg:.1f} c за {args.runs} прогон(ов), "
          f"{compute} на {args.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
