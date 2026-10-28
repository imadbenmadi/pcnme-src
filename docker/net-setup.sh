#!/bin/sh
# Applies the emulated network for this container, then runs its command (final-design 15.4).
# Shaping parameters come from environment variables set by compose (and later by the runner, per scenario).
# To verify in M9 (final-design 15.4): routing through wan-gw, netem jitter distribution, and that jitter
# does not reorder packets in a way that distorts transfer times.
set -eu

ROLE="${PCNME_ROLE:-none}"

shape() {  # shape <interface> <mean_delay_ms> <jitter_ms> <rate>
  # netem jitter without a named distribution is uniform: delay mean +- jitter = U(mean-jitter, mean+jitter).
  tc qdisc replace dev "$1" root netem delay "${2}ms" "${3}ms" rate "$4"
}

iface_for() {  # iface_for <ip prefix>, e.g. 10.20.0.
  ip -o -4 addr show | awk -v p="$1" '$4 ~ "^"p {print $2; exit}'
}

case "$ROLE" in
  zone)
    # Backhaul: U(1, 5) ms, rate limit on the zone's outgoing interface. Cloud traffic routes via wan-gw.
    shape "$(iface_for 10.20.0.)" "${BACKHAUL_DELAY_MS:-3}" "${BACKHAUL_JITTER_MS:-2}" "${BACKHAUL_RATE:-1gbit}"
    ip route replace 10.30.0.0/24 via 10.20.0.3
    # Radio: base delay only; the Shannon-rate pipe runs in software at the zone's intake.
    shape "$(iface_for 10.10.0.)" "${RADIO_DELAY_MS:-3}" "${RADIO_JITTER_MS:-2}" "10gbit"
    ;;
  wan-gw)
    # The metro-edge router: the one shared WAN pipe, shaped on its cloud side. U(25, 100) ms one way.
    shape "$(iface_for 10.30.0.)" "${WAN_DELAY_MS:-62.5}" "${WAN_JITTER_MS:-37.5}" "${WAN_RATE:-515mbit}"
    # Return direction (cloud -> broker / zones): one backhaul crossing (final-design 12.2: 1 WAN + 1 backhaul).
    shape "$(iface_for 10.20.0.)" "${BACKHAUL_DELAY_MS:-3}" "${BACKHAUL_JITTER_MS:-2}" "10gbit"
    ;;
  cloud)
    # Return WAN crossing: delay only (results are small, no bandwidth term, final-design 4.2).
    shape "$(iface_for 10.30.0.)" "${WAN_DELAY_MS:-62.5}" "${WAN_JITTER_MS:-37.5}" "10gbit"
    ip route replace 10.20.0.0/24 via 10.30.0.3
    ;;
  broker-net)
    # Sidecar sharing the broker's network namespace: each broker -> zone message is one backhaul crossing.
    shape "$(iface_for 10.20.0.)" "${BACKHAUL_DELAY_MS:-3}" "${BACKHAUL_JITTER_MS:-2}" "10gbit"
    ip route replace 10.30.0.0/24 via 10.20.0.3
    ;;
  car)
    shape "$(iface_for 10.10.0.)" "${RADIO_DELAY_MS:-3}" "${RADIO_JITTER_MS:-2}" "10gbit"
    ;;
  *) ;;
esac

exec "$@"
