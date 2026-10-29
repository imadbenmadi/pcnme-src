"""The MQTT topics of final-design 5.2, as fixed strings in one place. No wildcards (9.6)."""

from __future__ import annotations


def car_attach(car_id: str) -> str:
    return f"car/{car_id}/attach"


def car_gps(car_id: str) -> str:
    return f"car/{car_id}/gps"


def car_result(car_id: str) -> str:
    return f"car/{car_id}/result"


def car_ack(car_id: str) -> str:
    return f"car/{car_id}/ack"


def zone_state(zone_id: str) -> str:
    return f"zone/{zone_id}/state"


def zone_status(zone_id: str) -> str:
    return f"zone/{zone_id}/status"


def parse(topic: str) -> tuple[str, str, str]:
    """("car" or "zone", the id, the kind), for example ("car", "c01", "result")."""
    scope, ident, kind = topic.split("/")
    return scope, ident, kind
