"""Calibration run C: measure the time table on this PC (final-design 14.4, 15.6 "Calibrate", 18).

Per task type and slot type: mean, p5, p95 of the run time, the longest step (decode, preprocess,
inference, postprocess; 11.x d_abort), the dynamic GPU energy per run (nvidia-smi / NVML), and the JPEG
sizes at quality 80. A GPU slot is one inference stream; the F-GPU class has c_gpu streams on one GPU, so
they run concurrently and the dynamic power is shared between them (final-design 12.2). A CPU slot is a
fixed group of cores (torch threads). T4 (detector + text recognition, 14.1): yolov8n finds the vehicles,
then EasyOCR's CRNN recogniser (recognition only) reads up to OCR_CROPS plate-like crops, the lower middle of
each vehicle box (BDD100K has no plate labels).

Not a project dependency: run with an ephemeral overlay, so the lock and the venv stay unchanged:
  uv run --with ultralytics --with nvidia-ml-py python tools/calibrate_yolo.py FRAMES_DIR [out.json]
"""

from __future__ import annotations

import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

QUALITY = 80
TYPES = {  # type -> (weights, frames per run, runs per GPU stream, CPU runs)
    "T1": ("yolov8n.pt", 1, 600, 150),
    "T3": ("yolov8n-seg.pt", 1, 600, 150),
    "T2": ("yolov8l.pt", 60, 20, 0),  # GPU only (config)
    "T4": ("ocr", 1, 300, 100),
}
OCR_CROPS = 3
VEHICLES = {2, 5, 7}  # COCO car, bus, truck
GPU_STREAMS = 2  # hardware.F-GPU.c_gpu
CPU_THREADS = 4  # cores per CPU slot (config comment)
CPU_SLOTS = 2  # hardware.F-GPU.c_cpu: both CPU slots busy at once
WARMUP = 20


def summarize(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs)
    return {"mean": float(a.mean()), "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95))}


def reencode(frames: list[Path]) -> list[bytes]:
    from PIL import Image

    out = []
    for f in frames:
        buf = io.BytesIO()
        Image.open(f).convert("RGB").save(buf, "JPEG", quality=QUALITY)
        out.append(buf.getvalue())
    return out


class PowerSampler:
    """Samples board power (W) every 10 ms; energy = integral of samples over a window."""

    def __init__(self) -> None:
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            w = self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
            self.samples.append((time.perf_counter(), w))
            time.sleep(0.01)

    def mean_w(self, t0: float, t1: float) -> float:
        ws = [w for t, w in self.samples if t0 <= t <= t1]
        return float(np.mean(ws))

    def close(self) -> None:
        self._stop.set()
        self._t.join()


class OcrTask:
    """T4: detect, then recognise text on plate-like crops; predict() mimics the YOLO result's speed dict."""

    def __init__(self, device: str) -> None:
        import easyocr
        from ultralytics import YOLO

        self.det = YOLO("yolov8n.pt")
        self.rec = easyocr.Reader(["en"], gpu=device != "cpu", detector=False, verbose=False)

    def predict(self, img, device: str, verbose: bool = False):
        import cv2

        r = self.det.predict(img, device=device, verbose=False)[0]
        speed = dict(r.speed)
        boxes = [
            b for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), strict=True) if int(c) in VEHICLES
        ]
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:OCR_CROPS]
        t0 = time.perf_counter()
        for x0, y0, x1, y1 in boxes:
            w, h = x1 - x0, y1 - y0
            crop = img[int(y0 + 0.6 * h) : int(y1), int(x0 + 0.25 * w) : int(x1 - 0.25 * w)]
            if crop.size:
                self.rec.recognize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        speed["recognition"] = (time.perf_counter() - t0) * 1000.0

        class R:
            pass

        out = R()
        out.speed = speed  # type: ignore[attr-defined]
        return [out]


def one_run(model, jpegs: list[bytes], i: int, n: int, device: str) -> tuple[float, float]:
    """One task = n frames; returns (run time s, longest step s)."""
    import cv2

    t0 = time.perf_counter()
    longest = 0.0
    for k in range(n):
        d0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(jpegs[(i * n + k) % len(jpegs)], np.uint8), cv2.IMREAD_COLOR)
        dec = time.perf_counter() - d0
        r = model.predict(img, device=device, verbose=False)[0]
        steps = [dec] + [v / 1000.0 for v in r.speed.values()]
        longest = max(longest, *steps)
    return time.perf_counter() - t0, longest


def _stream(weights, n, runs, device, jpegs, s, streams, barrier, q) -> None:
    """One slot = one process (no shared GIL between slots, as on a real node)."""
    import torch
    from ultralytics import YOLO

    if device == "cpu":
        torch.set_num_threads(CPU_THREADS)
    model = OcrTask(device) if weights == "ocr" else YOLO(weights)
    for i in range(WARMUP):
        one_run(model, jpegs, i, 1, device)
    barrier.wait()
    t0 = time.perf_counter()
    res = [one_run(model, jpegs, i * streams + s, n, device) for i in range(runs)]
    q.put((res, t0, time.perf_counter()))


def measure(weights: str, n: int, runs: int, device: str, streams: int, jpegs: list[bytes]):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(streams), ctx.Queue()
    procs = [
        ctx.Process(target=_stream, args=(weights, n, runs, device, jpegs, s, streams, barrier, q))
        for s in range(streams)
    ]
    for p in procs:
        p.start()
    outs = [q.get() for _ in procs]
    for p in procs:
        p.join()
    flat = [x for o in outs for x in o[0]]
    t0, t1 = min(o[1] for o in outs), max(o[2] for o in outs)
    return [x[0] for x in flat], [x[1] for x in flat], t0, t1


def main(frames_dir: Path, out: Path, only: list[str] | None = None) -> None:
    import torch

    frames = sorted(frames_dir.glob("*.jpg"))
    jpegs = reencode(frames)
    sizes_kb = [len(b) / 1000 for b in jpegs]
    from PIL import Image

    report: dict = {
        "gpu": torch.cuda.get_device_name(0),
        "frames": len(frames),
        "resolution": list(Image.open(frames[0]).size),
        "jpeg_q80_kb": summarize(sizes_kb),
        "rows": {},
    }
    ps = PowerSampler()
    time.sleep(5)
    idle_w = ps.mean_w(time.perf_counter() - 5, time.perf_counter())
    report["gpu_idle_w"] = idle_w
    for typ, (w, n, gpu_runs, cpu_runs) in TYPES.items():
        if only and typ not in only:
            continue
        times, longest, t0, t1 = measure(w, n, gpu_runs, "cuda:0", GPU_STREAMS, jpegs)
        e_dyn = (ps.mean_w(t0, t1) - idle_w) * (t1 - t0) / len(times)
        row = summarize(times) | {
            "longest_step": float(np.mean(longest)),
            "energy_j": e_dyn,
            "max": max(times),
        }
        report["rows"][f"{typ}/F-GPU/gpu"] = row
        print(typ, "gpu", {k: round(v, 5) for k, v in row.items()}, flush=True)
        if cpu_runs:
            times, longest, _, _ = measure(w, n, cpu_runs, "cpu", CPU_SLOTS, jpegs)
            row = summarize(times) | {"longest_step": float(np.mean(longest)), "max": max(times)}
            report["rows"][f"{typ}/F-GPU/cpu"] = row
            print(typ, "cpu", {k: round(v, 5) for k, v in row.items()}, flush=True)
    ps.close()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main(
        Path(sys.argv[1]),
        Path(sys.argv[2] if len(sys.argv) > 2 else "calibration.json"),
        sys.argv[3:] or None,
    )
