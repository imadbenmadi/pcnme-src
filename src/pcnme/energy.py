"""Energy accounting (final-design 12).

Decisions use the dynamic energy one attempt causes: the network that carries its data plus the node that runs
it (12.1). For one attempt (12.2):

    E_k = E_fwd + E_exec + E_ret
    E_fwd:  here 0;  neighbour e_bh S;  cloud (e_bh + e_wan) S            (waiting in a pipe costs nothing)
    E_exec: PUE_k p T_run, T_run the real busy time of the slot, an abort's d_abort tail included
    E_ret:  (backhaul crossings e_bh + WAN crossings e_wan + e_down) S_r per delivery; fog 2 backhaul, cloud
            1 WAN + 1 backhaul; each republish pays it again

e_bh and e_wan include PUE_network; e_down does not (QUESTIONS Q3). ê = min(E / E_ref, ē) (12.3). The energy
per attempt follows 12.5; a task's energy is the sum over its attempts. The car's upload is reported only
(12.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pcnme.config import Config, SlotType

if TYPE_CHECKING:
    from pcnme.zone.zone import TaskRecord

Destination = Literal["here", "neighbour", "cloud"]
Executor = Literal["fog", "cloud"]


@dataclass(frozen=True, slots=True)
class AttemptEnergy:
    forwarding: float
    execution: float
    ret: float

    @property
    def total(self) -> float:
        return self.forwarding + self.execution + self.ret


class EnergyModel:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._e = config.energy

    def forwarding(self, destination: Destination, size_bits: float) -> float:
        if destination == "here":
            return 0.0
        if destination == "neighbour":
            return self._e.e_bh * size_bits
        return (self._e.e_bh + self._e.e_wan) * size_bits

    def ret(self, executor: Executor, result_bits: float, deliveries: int = 1) -> float:
        network = self._e.e_wan + self._e.e_bh if executor == "cloud" else 2 * self._e.e_bh
        return (network + self._e.e_down) * result_bits * deliveries

    def e_ref(self, task_type: str) -> float:
        return self._cfg.derived.e_ref[task_type]

    def e_hat(self, task_type: str, energy: float) -> float:
        """ê = min(E / E_ref, ē) (12.3)."""
        return min(energy / self.e_ref(task_type), self._cfg.derived.e_bar)

    def estimate(
        self, task_type: str, destination: Destination, hw: str, slot: SlotType, run: float
    ) -> float:
        """Ê_k before placing (12.4): E_fwd + PUE_k p Ê + E_ret, with `run` the time-table estimate on q*."""
        t = self._cfg.task_types[task_type]
        node = self._cfg.hardware[hw]
        execution = self._e.pue(node) * node.p_dyn_w[slot] * run  # p = slot power for placeholder rows (18)
        executor: Executor = "cloud" if node.is_cloud else "fog"
        return self.forwarding(destination, t.input_bits) + execution + self.ret(executor, t.result_bits)

    def of_record(self, rec: TaskRecord) -> AttemptEnergy:
        """The energy of a task's last attempt, by the 12.5 table (earlier refused attempts: `earlier`)."""
        if rec.option is None:
            return AttemptEnergy(0.0, 0.0, 0.0)  # dropped before any attempt: intake or all masked
        destination = self.destination(rec.option)
        fwd = self.forwarding(destination, rec.task.input_bits)
        report = rec.report
        if rec.reply in ("full", "late") or report is None:
            return AttemptEnergy(fwd, 0.0, 0.0)  # refused: forwarding only
        if report.outcome == "skipped":
            return AttemptEnergy(fwd, 0.0, 0.0)  # skipped in the queue: forwarding only
        execution = report.energy  # PUE p T_run, the abort tail included (executor, 12.2)
        if report.outcome != "done":
            return AttemptEnergy(fwd, execution, 0.0)  # aborted / cancelled: forwarding + execution spent
        executor: Executor = "cloud" if rec.option == "cloud" else "fog"
        return AttemptEnergy(fwd, execution, self.ret(executor, rec.task.result_bits, rec.deliveries))

    @staticmethod
    def destination(option: str) -> Destination:
        return "here" if option == "here" else ("cloud" if option == "cloud" else "neighbour")

    def earlier(self, rec: TaskRecord) -> float:
        """The task's earlier attempts, each refused: forwarding only (12.5)."""
        return sum(self.forwarding(self.destination(e.option), rec.task.input_bits) for e in rec.earlier)

    def car_upload(self, rec: TaskRecord) -> float:
        """Reported only (12.1): the car's airtime, P_tx S / R_car."""
        return self._e.car_tx_w * rec.task.input_bits / rec.rate
