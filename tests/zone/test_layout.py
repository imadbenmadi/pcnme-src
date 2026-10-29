"""Layout and neighbours (PLAN M3.2; final-design 7.5, 18 Layout)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcnme.config import Config, load_config
from pcnme.zone.layout import NeighbourTable, Site, is_candidate


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def sites(cfg: Config) -> dict[str, Site]:
    r = cfg.layout.zone_radius_m
    return {z: Site(z, x, y, r) for z, (x, y) in cfg.layout.zone_positions_m.items()}


def start(cfg: Config) -> dict[str, NeighbourTable]:
    s = sites(cfg)
    active = [s[z] for z in cfg.layout.zones if z not in cfg.layout.joining_zones]
    return {me.zone_id: NeighbourTable.at_start(me, active, cfg.layout.d_nb_m) for me in active}


def test_the_config_layout_matches_section_18(cfg: Config) -> None:
    lay = cfg.layout
    assert lay.zone_positions_m == {
        "z1": (300, 300),
        "z2": (900, 300),
        "z3": (300, 900),
        "z4": (900, 900),
        "z5": (600, 600),
    }
    assert (lay.zone_radius_m, lay.d_nb_m, lay.joining_zones) == (350, 900, ("z5",))


def test_candidates_overlap_or_lie_within_d_nb(cfg: Config) -> None:
    # Adjacent centres 600 m < 2 x 350: overlap. Diagonal 849 m: no overlap, but within d_nb = 900 m (18).
    s = sites(cfg)
    assert is_candidate(s["z1"], s["z2"], 900) and is_candidate(s["z1"], s["z4"], 900)
    assert not is_candidate(s["z1"], s["z4"], 800)  # 849 > 800 and 849 > 700
    assert is_candidate(s["z1"], s["z2"], 0)  # overlap alone is enough
    assert not is_candidate(s["z1"], s["z1"], 900)


def test_start_up_positions_are_nearest_first_with_ties_by_id(cfg: Config) -> None:
    # z1: z2 and z3 at 600 m (tie, by id), z4 at 849 m; 3 neighbours, the 4th position empty.
    t = start(cfg)
    assert t["z1"].positions == ["z2", "z3", "z4", None]
    assert t["z2"].positions == ["z1", "z4", "z3", None]
    assert t["z3"].positions == ["z1", "z4", "z2", None]
    assert t["z4"].positions == ["z2", "z3", "z1", None]


def test_z5_joins_the_lowest_free_position_of_every_zone_that_counts_it(cfg: Config) -> None:
    # 7.5.2 and 15.7: z5 at (600, 600) is 424 m from every centre, so every zone counts it; each has
    # position 4 free. z5's own table: all four at 424 m, by id.
    t = start(cfg)
    z5 = sites(cfg)["z5"]
    for table in t.values():
        table.on_status(z5, online=True)
    assert [t[z].positions[3] for z in ("z1", "z2", "z3", "z4")] == ["z5"] * 4
    assert t["z1"].positions[:3] == ["z2", "z3", "z4"]  # the others never move
    s = sites(cfg)
    mine = NeighbourTable.at_start(z5, [s[z] for z in ("z1", "z2", "z3", "z4")], cfg.layout.d_nb_m)
    assert mine.positions == ["z1", "z2", "z3", "z4"]


def test_a_zone_with_four_neighbours_keeps_them(cfg: Config) -> None:
    me = Site("a", 0, 0, 350)
    four = [Site(f"b{i}", 100.0 * (i + 2), 0, 350) for i in range(4)]
    t = NeighbourTable.at_start(me, four, 900)
    t.on_status(Site("near", 10, 0, 350), online=True)  # nearer than all, but no free position
    assert t.positions == ["b0", "b1", "b2", "b3"]


def test_an_offline_neighbour_keeps_its_position_and_comes_back(cfg: Config) -> None:
    t = start(cfg)["z1"]
    z3 = sites(cfg)["z3"]
    t.on_status(z3, online=False)
    assert t.positions == ["z2", "z3", "z4", None] and not t.is_online("z3")
    t.on_status(z3, online=True)
    assert t.positions == ["z2", "z3", "z4", None] and t.is_online("z3")


def test_a_zone_joins_when_it_comes_online(cfg: Config) -> None:
    t = start(cfg)["z1"]
    t.on_status(sites(cfg)["z5"], online=False)  # announced, but offline: no position yet
    assert t.positions[3] is None
    t.on_status(sites(cfg)["z5"], online=True)
    assert t.positions[3] == "z5"
    t.on_status(sites(cfg)["z5"], online=True)  # a repeat changes nothing
    assert t.positions == ["z2", "z3", "z4", "z5"]


def test_not_symmetric(cfg: Config) -> None:
    # 7.5.4: B has 4 nearer candidates, so it does not count A; A has room and counts B.
    a, b = Site("a", 0, 0, 350), Site("b", 600, 0, 350)
    near_b = [Site(f"n{i}", 600 + 50.0 * (i + 1), 0, 350) for i in range(4)]
    tb = NeighbourTable.at_start(b, [a, *near_b], 900)
    ta = NeighbourTable.at_start(a, [b], 900)
    assert "a" not in tb.positions and ta.positions[0] == "b"


def test_the_cloud_is_never_a_neighbour_position(cfg: Config) -> None:
    t = start(cfg)["z1"]
    assert (t.position("z4"), t.position("z5")) == (2, None)
    t.on_status(Site("cloud", 300, 300, 0), online=True)  # it announces itself, even at the same spot
    assert t.position("cloud") is None and t.positions[3] is None


# --- over the bus: status messages and the join (5.2, 7.5, 15.7) -------------------------------------------


def test_zones_learn_joins_and_last_wills_from_status_messages() -> None:
    from pcnme.bus.bus import Bus
    from pcnme.net.pipe import FifoPipe
    from pcnme.node.timetable import RunTimeTable
    from pcnme.sim.clock import EventClock
    from pcnme.sim.rng import RandomStreams
    from pcnme.zone.zone import Cloud, Zone
    from tests.zone.harness import Fixed, fixed_delay_config

    cfg = fixed_delay_config()
    clock = EventClock()
    streams = RandomStreams.from_config(cfg, 2000, "training")
    bus = Bus(clock, streams, cfg.links.broker_extra_delay)
    table = RunTimeTable.from_config(cfg)
    wan = FifoPipe(cfg.derived.b_wan)
    cloud = Cloud(clock, streams, cfg, table, bus)
    s = sites(cfg)

    def zone(z: str, active: list[str]) -> Zone:
        others = tuple(s[o] for o in active if o != z)
        return Zone(
            z, clock, streams, cfg, table, wan, cloud, Fixed(None), "loose", bus=bus, start_sites=others
        )

    first = ["z1", "z2", "z3", "z4"]
    zones = {z: zone(z, first) for z in first}
    clock.run_until(0.5)
    assert zones["z1"].neighbours.positions == ["z2", "z3", "z4", None]
    assert zones["z1"].neighbours.is_online("cloud")  # the cloud announced itself too
    zones["z5"] = zone("z5", [*first, "z5"])  # 15.7: z5 joins and publishes online
    clock.run_until(1.0)
    assert [zones[z].neighbours.positions[3] for z in first] == ["z5"] * 4
    assert zones["z5"].neighbours.positions == ["z1", "z2", "z3", "z4"]
    bus.kill("z3")  # its last will: offline, at once
    clock.run_until(1.1)
    assert not zones["z1"].neighbours.is_online("z3") and zones["z1"].neighbours.position("z3") == 1


def test_within_d_nb_includes_d_nb_itself() -> None:
    # Radii small enough not to overlap: exactly at d_nb is "within" (7.5.1).
    a, b = Site("a", 0, 0, 10), Site("b", 900, 0, 10)
    assert is_candidate(a, b, 900) and not is_candidate(a, b, 899.999)


def test_a_join_takes_the_lowest_free_position() -> None:
    me = Site("a", 0, 0, 350)
    t = NeighbourTable.at_start(me, [Site("b", 500, 0, 350), Site("c", 600, 0, 350)], 900)
    assert t.positions == ["b", "c", None, None]
    t.on_status(Site("d", 700, 0, 350), online=True)
    assert t.positions == ["b", "c", "d", None]
