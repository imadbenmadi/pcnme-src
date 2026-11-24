"""The PCNME dataset for Zenodo: every result, model and log of both maps, zipped per kind, with a datasheet.

    uv run python tools/export_dataset.py                 # -> D:/pcnme-zenodo
    uv run python tools/export_dataset.py --out E:/zenodo --dry-run

Writes one folder with at most 100 files (Zenodo's limit per record): per map a zip for the results (tables,
figures, archived S6), the per-run logs (one zip per scenario), training (models and curves), NSGA-II (plans,
fronts, w*), demonstrations, the testbed, traces (LuST) and the run logs; the calibration; DATASHEET.md,
METRICS.md, LICENSE-DATA.txt and SHA256SUMS.txt. Never included: the BDD100K frames and the YOLO weights
(data/calibration/frames, weights: not ours to publish), the LuST map itself (cite it), smoke runs, the
testbed's _test* folders.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
MAPS = ("vfc", "lust")
SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "S6")
STORED = {".parquet", ".gz", ".png", ".npz", ".zip", ".jpg"}  # already compressed
SKIP_PARTS = {"frames", "weights", "__pycache__"}
SKIP_NAMES = {"points.lp"}  # the InfluxDB feed: a text copy of the Parquet logs (Grafana only)


def files_under(root: Path, skip_prefix: tuple[str, ...] = ()) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root).parts
        skipped = bool(skip_prefix) and bool(rel) and rel[0].startswith(skip_prefix)
        if f.is_file() and f.name not in SKIP_NAMES and not SKIP_PARTS & set(rel) and not skipped:
            yield f


def plan() -> dict[str, list[tuple[Path, Path]]]:
    """zip name -> [(file, its path inside the zip)]."""
    out: dict[str, list[tuple[Path, Path]]] = {}

    def add(zip_name: str, root: Path, skip: tuple[str, ...] = (), only: str | None = None) -> None:
        if not root.exists():
            return
        for f in files_under(root, skip):
            if only and not f.relative_to(root).parts[0].startswith(only):
                continue
            out.setdefault(zip_name, []).append((f, f.relative_to(DATA)))

    for m in MAPS:
        d = DATA / m
        for top in ("manifest.json", "w_star.json", "calibrated.yaml"):
            add(f"{m}-meta.zip", d / top)
        add(f"{m}-results.zip", d / "results", skip=("runs",))
        for s in SCENARIOS:
            add(f"{m}-runs-{s}.zip", d / "results" / "runs", only=f"{s}-")
        add(f"{m}-train.zip", d / "train")
        add(f"{m}-nsga.zip", d / "nsga")
        add(f"{m}-demos.zip", d / "demos")
        add(f"{m}-testbed.zip", d / "testbed", skip=("_test",))
        add(f"{m}-traces.zip", d / "traces")
        add(f"{m}-logs.zip", d / "logs")
        add(f"{m}-archive.zip", d / "_archive")
    add("calibration.zip", DATA / "calibration")
    return out


def write_zip(path: Path, members: list[tuple[Path, Path]]) -> None:
    tmp = path.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for f, arc in members:
            how = zipfile.ZIP_STORED if f.suffix in STORED else zipfile.ZIP_DEFLATED
            z.write(f, arc.as_posix(), compress_type=how)
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("D:/pcnme-zenodo"))
    ap.add_argument("--dry-run", action="store_true", help="print the zips and their sizes only")
    a = ap.parse_args(argv)
    zips = plan()
    total = 0
    for name, members in zips.items():
        size = sum(f.stat().st_size for f, _ in members)
        total += size
        print(f"{name:<24} {len(members):>7} files {size / 1e9:8.2f} GB")
    print(f"{len(zips)} zips, {total / 1e9:.2f} GB before compression")
    if a.dry_run:
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    for name, members in zips.items():
        target = a.out / name
        if target.exists():  # resumable: a finished zip is kept
            continue
        print(f"writing {name} ...", flush=True)
        write_zip(target, members)
    docs = {"DATASHEET.md": REPO / "docs" / "DATASHEET.md", "METRICS.md": REPO / "docs" / "METRICS.md"}
    docs["LICENSE-DATA.txt"] = REPO / "LICENSE-DATA.txt"
    for doc, src in docs.items():
        shutil.copy2(src, a.out / doc)
    sums = [
        f"{sha256(f)}  {f.name}"
        for f in sorted(a.out.iterdir())
        if f.is_file() and f.name != "SHA256SUMS.txt"
    ]
    (a.out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"done: {a.out} ({len(sums) + 1} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
