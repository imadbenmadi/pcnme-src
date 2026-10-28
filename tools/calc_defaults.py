"""Check that the PCNME default numbers fit together (final-design 18).

Reads config/default.yaml, so it can never drift from the config. Placeholders (tag M) must be replaced by
measured values in the calibration run; then rerun this script.

Run from the repo root: python tools/calc_defaults.py [path/to/config.yaml]
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

KB_TO_BITS = 1e3 * 8
NJ = 1e-9
P95_FACTOR = 1.3  # placeholder p95 = 1.3 x mean (config time_table_placeholder comment)
TYPICAL_RADIO_BPS = 600e6  # Friis + Shannon at about 150-200 m (final-design 18)


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main(path: Path) -> None:
    cfg = load(path)
    types = cfg["task_types"]
    tt = cfg["time_table_placeholder"]
    hw = cfg["hardware"]
    energy = cfg["energy"]
    alpha = cfg["tof"]["alpha"]
    zones = sum(1 for z in cfg["layout"]["zone_positions_m"] if z != "z5")
    c_gpu = hw["F-GPU"]["c_gpu"]

    active = {name: t for name, t in types.items() if "share" in t}
    shares = sum(t["share"] for t in active.values())
    assert abs(shares - 1.0) < 1e-9, f"task shares sum to {shares}, not 1"

    gpu_run = {name: tt[name]["F-GPU"]["gpu"] for name in active}
    mean_gpu = sum(t["share"] * gpu_run[n] for n, t in active.items())
    capacity = c_gpu / mean_gpu
    print(f"mean GPU run of the mix {mean_gpu * 1e3:.1f} ms; zone GPU capacity {capacity:.1f} tasks/s")

    # Deadlines and the TOF-Broker (final-design 7.1).
    for name, t in active.items():
        for kind in ("loose", "tight"):
            d = t["deadline"][kind]
            if isinstance(d, str):  # "1.1x_E_strong": calibrated rule for T2's tight deadline
                d = float(d.split("x")[0]) * gpu_run[name]
            boulder = gpu_run[name] >= alpha * d
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
    p_fog, p_cl = hw["F-GPU"]["p_dyn_w"], hw["cloud"]["p_dyn_w"]
    e_bar = 0.0
    for name, t in active.items():
        s_in, s_out = bits[name], t["result_kb"] * KB_TO_BITS
        e_ret_here = (2 * e_bh + e_down) * s_out
        e_ref = energy["pue_fog"] * p_fog["gpu_slot"] * gpu_run[name] + e_ret_here
        opts = {"fog GPU": energy["pue_fog"] * p_fog["gpu_slot"] * P95_FACTOR * gpu_run[name]}
        if "cpu" in tt[name]["F-GPU"]:
            opts["fog CPU"] = energy["pue_fog"] * p_fog["cpu_slot"] * P95_FACTOR * tt[name]["F-GPU"]["cpu"]
        for k in opts:
            opts[k] += e_bh * s_in + e_ret_here
        opts["cloud"] = (
            energy["pue_cloud"] * p_cl["gpu_slot"] * P95_FACTOR * tt[name]["cloud"]["gpu"]
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
