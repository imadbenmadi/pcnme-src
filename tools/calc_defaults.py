"""Check that the PCNME default numbers fit together (final-design 18).

Reads config/default.yaml, so it can never drift from the config. Placeholders (tag M) must be replaced by
measured values in the calibration run; then rerun this script. Kept independent of the pcnme package on
purpose: tests/test_config.py checks that pcnme.config derives the same numbers.

Run from the repo root: python tools/calc_defaults.py [path/to/config.yaml]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

KB_TO_BITS = 1e3 * 8
NJ = 1e-9
TYPICAL_RADIO_BPS = 600e6  # Friis + Shannon at about 150-200 m (final-design 18)

# YAML 1.2 floats: PyYAML follows YAML 1.1, which reads 40.0e6 or 1e9 (no exponent sign) as strings.
# Needs a dot or an exponent, so integers stay integers.
_FLOAT_12 = re.compile(r"^[-+]?(?:(?:\d[\d_]*\.[\d_]*|\.\d[\d_]*)(?:[eE][-+]?\d+)?|\d[\d_]*[eE][-+]?\d+)$")


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT_12, list("-+0123456789."))


def load(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)


def mean_of(row: float | dict) -> float:
    return row["mean"] if isinstance(row, dict) else row


def p95_of(row: float | dict, factors: dict) -> float:
    """A measured p95 wins; the placeholder factor fills it only when the row has none (QUESTIONS Q2)."""
    if isinstance(row, dict) and "p95" in row:
        return row["p95"]
    return factors["p95"] * mean_of(row)


def main(path: Path) -> None:
    cfg = load(path)
    types = cfg["task_types"]
    tt = {k: v for k, v in cfg["time_table_placeholder"].items() if k != "factors"}
    factors = cfg["time_table_placeholder"]["factors"]
    hw = cfg["hardware"]
    energy = cfg["energy"]
    alpha = cfg["tof"]["alpha"]
    strong = cfg["tof"]["strongest_class"]
    ref = energy["reference_class"]
    zones = len(set(cfg["layout"]["zone_positions_m"]) - set(cfg["layout"]["joining_zones"]))
    c_gpu = hw[ref]["c_gpu"]

    active = {name: t for name, t in types.items() if "share" in t}
    shares = sum(t["share"] for t in active.values())
    assert abs(shares - 1.0) < 1e-9, f"task shares sum to {shares}, not 1"

    # rho_fog uses the reference class's GPU slot over the whole mix, boulders included (final-design 16.2).
    gpu_run = {name: mean_of(tt[name][ref]["gpu"]) for name in active}
    mean_gpu = sum(t["share"] * gpu_run[n] for n, t in active.items())
    capacity = c_gpu / mean_gpu
    print(f"mean GPU run of the mix {mean_gpu * 1e3:.1f} ms; zone GPU capacity {capacity:.1f} tasks/s")

    # Deadlines and the TOF-Broker (final-design 7.1): E_strong = mean run on the strongest class,
    # fastest slot type.
    e_strong = {name: min(mean_of(r) for r in tt[name][strong].values()) for name in active}
    for name, t in active.items():
        for kind in ("loose", "tight"):
            d = t["deadline"][kind]
            if isinstance(d, str):  # "1.1x_E_strong": calibrated rule for T2's tight deadline
                d = float(d.split("x")[0]) * e_strong[name]
            boulder = e_strong[name] >= alpha * d
            print(f"  {name} {kind} D = {d:.3f} s -> {'boulder' if boulder else 'pebble'}")

    # Scenario loads and the one WAN bandwidth (final-design 16.2).
    bits = {n: t["input_kb"] * KB_TO_BITS for n, t in active.items()}
    mean_bits = sum(t["share"] * bits[n] for n, t in active.items())
    soft_bits = sum(t["share"] * bits[n] for n, t in active.items() if t["cls"] == "soft")
    s2_rho = cfg["scenarios"]["S2"]["rho_fog"]
    b_wan = s2_rho * capacity * zones * soft_bits / 0.9
    print(f"mean input {mean_bits / KB_TO_BITS:.0f} KB; B_WAN = {b_wan / 1e6:.0f} Mbps (TI_WAN 0.9 at S2)")
    for s in ("S1", "S2", "S3"):
        rho = cfg["scenarios"][s]["rho_fog"]
        lam_zone = rho * capacity
        ti_wan = lam_zone * zones * soft_bits / b_wan
        per_car = [lam_zone / (n / zones) for n in cfg["cars"]["count"]]
        print(
            f"  {s}: {lam_zone:.1f} tasks/s per zone; per car {per_car[0]:.3f} / {per_car[1]:.3f} "
            f"(at {cfg['cars']['count'][0]} / {cfg['cars']['count'][1]} cars); TI_WAN {ti_wan:.2f}"
        )
    print(f"radio TI per zone at S2: {s2_rho * capacity * mean_bits / TYPICAL_RADIO_BPS:.2f}")

    # Energy ratios (final-design 12). Network energy per bit x network PUE.
    pue_net = energy["pue_network"]
    e_bh = energy["e_bh_nj_per_bit"] * NJ * pue_net
    e_wan = energy["e_wan_nj_per_bit"] * NJ * pue_net
    e_down = energy["e_down_nj_per_bit"] * NJ
    p_fog, p_cl = hw[ref]["p_dyn_w"], hw["cloud"]["p_dyn_w"]
    e_bar = 0.0
    for name, t in active.items():
        s_in, s_out = bits[name], t["result_kb"] * KB_TO_BITS
        rows = tt[name][ref]
        q_ref = min(rows, key=lambda q: mean_of(rows[q]))  # fastest slot type on the reference class (12.3)
        e_ret_here = (2 * e_bh + e_down) * s_out
        e_ref = energy["pue_fog"] * p_fog[f"{q_ref}_slot"] * mean_of(rows[q_ref]) + e_ret_here
        opts = {"fog GPU": energy["pue_fog"] * p_fog["gpu_slot"] * p95_of(rows["gpu"], factors)}
        if "cpu" in rows:
            opts["fog CPU"] = energy["pue_fog"] * p_fog["cpu_slot"] * p95_of(rows["cpu"], factors)
        for k in opts:
            opts[k] += e_bh * s_in + e_ret_here
        opts["cloud"] = (
            energy["pue_cloud"] * p_cl["gpu_slot"] * p95_of(tt[name]["cloud"]["gpu"], factors)
            + (e_bh + e_wan) * s_in
            + (e_wan + e_bh + e_down) * s_out
        )
        ratios = {k: v / e_ref for k, v in opts.items()}
        e_bar = max(e_bar, *ratios.values())
        print(
            f"  {name}: E_ref {e_ref:.3f} J; worst-case ratios "
            + ", ".join(f"{k} {v:.2f}" for k, v in ratios.items())
            + f"; network share {e_bh * s_in / e_ref:.4f}"
        )
    print(f"e_bar (placeholder) = {e_bar:.2f}")

    # Discount horizon at S2 (final-design 11.3).
    gamma, tau = cfg["rl"]["gamma_per_tau"], cfg["rl"]["tau"]
    dt = 1 / (s2_rho * capacity)
    step = gamma ** (dt / tau)
    print(
        f"S2: an anchor decides every {dt * 1e3:.1f} ms; per-step discount {step:.4f}; "
        f"horizon about {1 / (1 - step):.0f} decisions"
    )


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
