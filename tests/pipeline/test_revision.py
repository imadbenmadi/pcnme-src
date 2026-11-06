"""Rerunning one scenario of a finished map (pipeline/revision.py; QUESTIONS Q60: S6 with loose deadlines)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pcnme.metrics.compare import BEGIN, END, results_tables, update_doc
from pcnme.paths import MapPaths
from pcnme.pipeline import revision, runinfo
from pcnme.pipeline.evaluate import done_rows

PLAN = {"map": "vfc", "scenarios": ["S1", "S6"], "windows": 6, "population": 40, "generations": 20,
        "budget": 1000000, "workers": None, "checkpoint_every_s": None}  # fmt: skip
ALL = ("S1", "S6")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture()
def finished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MapPaths:
    """A finished map with outputs for S1 and S6."""
    monkeypatch.setenv("PCNME_DATA_DIR", str(tmp_path))
    p = MapPaths("vfc")
    for s in ALL:
        for k in (0, 1):
            (p.nsga / f"{s}-w{k}.json").parent.mkdir(parents=True, exist_ok=True)
            (p.nsga / f"{s}-w{k}.json").write_text("{}", encoding="utf-8")
            (p.demos / f"nsga-{s}-w{k}.parquet").parent.mkdir(parents=True, exist_ok=True)
            (p.demos / f"nsga-{s}-w{k}.parquet").write_bytes(b"x")
        (p.train(s, "pcnme") / "trained.pt").parent.mkdir(parents=True, exist_ok=True)
        (p.train(s, "pcnme") / "trained.pt").write_bytes(b"x")
    write_csv(p.nsga / "fronts.csv", [{"scenario": s, "window": k} for s in ALL for k in (0, 1)])
    write_csv(p.demos / "demos.csv", [{"kind": "nsga", "scenario": s, "window": 0, "plans": 3} for s in ALL])
    p.w_star.write_text(json.dumps({"w_star": 0.414}), encoding="utf-8")
    ident = {"config_hash": "old", "commit": "c1", "plan": PLAN}
    runinfo.write(p.manifest, {"identity": ident, "started": "t0", "finished": "t1"})
    return p


def identity(**plan: object) -> dict[str, object]:
    return {"config_hash": "new", "commit": "c2", "plan": {**PLAN, "scenarios": ["S6"], **plan}}


def test_a_scenario_subset_of_a_finished_map_is_a_revision(finished: MapPaths) -> None:
    assert revision.applies(finished, ["S6"], ALL)
    assert not revision.applies(finished, ALL, ALL)  # the full set is a normal run
    m = json.loads(finished.manifest.read_text(encoding="utf-8"))
    del m["finished"]
    runinfo.write(finished.manifest, m)
    assert not revision.applies(finished, ["S6"], ALL)  # an unfinished map resumes as before


def test_the_revision_archives_only_its_scenario_and_keeps_w_star(finished: MapPaths) -> None:
    rec = revision.start(finished, identity(), "tight")
    dest = Path(rec["archived_to"])
    assert sorted(f.name for f in finished.nsga.glob("*.json")) == ["S1-w0.json", "S1-w1.json"]
    assert sorted(f.name for f in (dest / "nsga").glob("*.json")) == ["S6-w0.json", "S6-w1.json"]
    assert (dest / "train" / "S6-pcnme" / "trained.pt").exists() and finished.train("S1", "pcnme").exists()
    assert {r["scenario"] for r in read_csv(finished.nsga / "fronts.csv")} == {"S1"}
    assert {r["scenario"] for r in read_csv(dest / "nsga" / "fronts.csv")} == {"S6"}
    assert {r["scenario"] for r in read_csv(finished.demos / "demos.csv")} == {"S1"}
    assert revision.kept_w_star(finished) == 0.414
    m = json.loads(finished.manifest.read_text(encoding="utf-8"))
    assert m["identity"]["config_hash"] == "old" and m["revisions"]["S6"]["label"] == "tight"


def test_a_revision_resumes_on_the_same_identity_and_refuses_another(finished: MapPaths) -> None:
    first = revision.start(finished, identity(), "tight")
    (finished.train("S6", "pcnme") / "checkpoint.pt").parent.mkdir(parents=True)
    (finished.train("S6", "pcnme") / "checkpoint.pt").write_bytes(b"x")
    again = revision.start(finished, identity(), "tight")  # resume: nothing archived again
    assert again["archived_to"] == first["archived_to"]
    assert (finished.train("S6", "pcnme") / "checkpoint.pt").exists()
    with pytest.raises(runinfo.Mismatch):
        revision.start(finished, {**identity(), "commit": "c3"}, "tight")
    with pytest.raises(runinfo.Mismatch, match="budget"):
        revision.start(finished, identity(budget=400), "tight")


def test_evaluation_archives_the_old_rows_and_checks_only_its_scenarios(finished: MapPaths) -> None:
    revision.start(finished, identity(), "tight")
    assert revision.label_of(finished, "S6") == "tight" and revision.label_of(finished, "S1") is None
    rows = [{"scenario": s, "method": "pcnme", "seed": 3000, "config_hash": "old", "commit": "c1",
             "run_dir": f"{s}-pcnme-3000-fast"} for s in ALL]  # fmt: skip
    write_csv(finished.results / "summary.csv", rows)
    for r in rows:
        (finished.runs / str(r["run_dir"])).mkdir(parents=True)
    assert revision.archive_results(finished.results, "S6", "tight", "new") == 1
    assert [r["scenario"] for r in read_csv(finished.results / "summary.csv")] == ["S1"]
    assert (finished.results / "archive" / "S6-tight" / "runs" / "S6-pcnme-3000-fast").exists()
    assert revision.archived(finished.results) == [
        (finished.results / "archive" / "S6-tight", "S6 tight (archived)")
    ]
    ident = {"commit": "c2", "config_hash": "new"}
    assert done_rows(finished.results / "summary.csv", ident, ["S6"]) == set()  # S1's old rows do not block
    with pytest.raises(RuntimeError):
        done_rows(finished.results / "summary.csv", ident)


def test_the_results_page_tables_are_rewritten_between_their_markers(tmp_path: Path) -> None:
    rows = [{"scenario": s, "method": m, "seed": "3000", "on_time": str(v), "latency_p99_s": "0.2",
             "energy_per_task_j": "2.0"} for s, m, v in (("S6", "pcnme", 0.9), ("S6", "greedy", 0.8),
                                                         ("S6 tight (archived)", "pcnme", 0.5))]  # fmt: skip
    text = results_tables(rows)
    assert "| PCNME | **90.0** | **50.0** |" in text and "| method | S6 | S6 tight (archived) |" in text
    doc = tmp_path / "RESULTS.md"
    doc.write_text(f"intro\n{BEGIN}\nold\n{END}\noutro\n", encoding="utf-8")
    assert update_doc(doc, rows)
    out = doc.read_text(encoding="utf-8")
    assert out.startswith("intro\n") and out.endswith(f"{END}\noutro\n") and "old" not in out
    assert b"\r\n" not in doc.read_bytes()  # LF on every platform, like the rest of the repository
    plain = tmp_path / "PLAIN.md"
    plain.write_text("no markers\n", encoding="utf-8")
    assert not update_doc(plain, rows) and plain.read_text(encoding="utf-8") == "no markers\n"


def test_a_traces_only_run_is_not_recorded_so_later_training_reuses_the_traces() -> None:
    from pcnme.pipeline.train import records_run

    assert not records_run("traces")
    assert all(records_run(s) for s in ("nsga", "demos", "learning"))
