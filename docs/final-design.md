<!-- PCNME — Final Design, version 3 (17 Nov 2026): revision 128 (28 Oct 2026), the later changes in Appendix D, the notation of 15 Nov 2026 (docs/NOTATION.md, Appendix A) and the reorganisation of 17 Nov (overview, a purpose box per section, worked examples, appendices). Section numbers 1–19 are unchanged: the code cites them. This file is the master copy: the claude.ai copy and the printed copy follow it. Do not edit it by hand except through the design-change skill. -->

# PCNME — Final Design

## How to read this document

This is the complete design of PCNME: the system, its mathematical models, its decision maker, its learning pipeline and its evaluation. It is the specification the code implements, and the code cites its section numbers (for example `# final-design 8.4 step 2`), so the numbers of sections 1 to 19 never change.

- **Start with "PCNME on one page"** below. It gives the whole path from the inputs to the outputs.
- **Part I (sections 3–14)** describes PCNME as it would run in a real city. **Part II (15–16)** describes how it is evaluated. **Part III (17–19)** gives the limits, every number and the origin of each decision.
- **Every section opens with a box**: its purpose, its inputs and its outputs.
- **Every model is written the same way**: inputs with units, the equations, the outputs, a worked example with the real numbers, and the source.
- **Symbols** follow one rule: one base letter is one kind of quantity with one unit (T a duration, t an instant, D and d a deadline, W a wait, δ a link delay, L a latency, S a size, B a bandwidth, E and e an energy, P a power, C a cost, n a count, ℓ a length). A hat is an estimate made before the fact, a bar a mean. Appendix A has the full table.
- **Appendices**: A notation, B results, C references, D the history of changes.

## PCNME on one page

**The problem.** Connected cars produce computing tasks, for example object detection on every camera frame, with deadlines of 50 ms to a few seconds. A car cannot run them itself. Each task must be placed on a fog node near the road or in the cloud so that it meets its deadline with the lowest latency and energy, while the car keeps moving from one zone to the next.

**The decision.** The zone a task arrives in (its anchor) decides once where the whole task runs, among 6 options: its own fog node, one of up to 4 neighbour zones, or the cloud.

**The method, in three steps.**

1. A fixed rule, the **TOF-Broker**, sends a task that is too heavy for any fog node straight to the cloud.
2. A learned agent, a **D3QN**, places every other task. It sees 48 numbers: the task, the estimated time and energy of each of the 6 options, and where the car is going.
3. The agent does not start from zero. Offline, **NSGA-II** computes plans that are Pareto-optimal in latency and energy; the **knee point** of each front is taken as the expert; the agent is pretrained on these plans with **DQfD**, then keeps learning online.

**The contribution** is that chain: NSGA-II front → knee point → demonstrations → pretrained D3QN, for task offloading.

### The whole system as inputs and outputs

| Stage | Inputs | Model or rule | Outputs | Section |
| --- | --- | --- | --- | --- |
| Task arrival | Task type, class, size S, deadline D, the car's budget | Intake: deadline bookkeeping | Anchor copy with its deadline d_A | 6, 14 |
| Network | S, bandwidth B, pipe state, link delays δ | Link model | Transfer times T_transfer, estimates T̂_in,k and T̂_ret,k | 4 |
| Queue | Slots c_q, remaining times T^rem, backlog Ω | Water-filling wait estimate | Queue wait Ŵ^queue_k,q | 8 |
| Execution | Task type, hardware class, slot type | Time table | Run time T̄^run_k,q, energy per run | 8, 14 |
| Energy | S, energy per bit, power, PUE, run time | Energy model | Energy E_k of an attempt, estimate Ê_k | 12 |
| Mobility | Position, speed, heading, zone circles | Exit-time predictor | T_exit and the next zone | 10 |
| Decision | All of the above, as 48 numbers | TOF-Broker → masks → D3QN | One of 6 placements | 7, 11 |
| Feedback | Measured latency L and energy E of the closed task | Costs C^time, C^en and the reward r | One training sample | 11 |
| Offline expert | Windows of tasks, the same costs | NSGA-II → knee point → weight w\* | Demonstrations and w\* | 13 |
| Evaluation | 8 methods, 6 scenarios, 20 seeds, 2 maps | Simulation and Docker testbed | On-time share, latency, energy, cold start | 15, 16, Appendix B |

### The components at a glance

| Component | Where | Input | Output | Section |
| --- | --- | --- | --- | --- |
| Car application | Car | Camera frames, GPS | Task uploads with a time budget; GPS every 100 ms; confirmation of each result | 6, 10 |
| Access point | Zone | Radio signal | The upload, at the car's radio rate | 4.2 |
| Intake | Zone | A complete upload | An anchored task with its deadline on the zone's clock | 6 A |
| TOF-Broker | Zone | Task type, D, the time table | Boulder (to the cloud) or pebble (to the agent) | 7.1 |
| Masks | Zone | The state store, the task | The options that can work right now | 7.2 |
| D3QN agent | Zone | 48 numbers | 6 action values; the best unmasked option | 7.3, 11 |
| Fog node | Zone | Tasks placed on it | Results; accept or refuse; its queue state every 100 ms | 8 |
| State store | Zone | Neighbours' state messages, own queues | The numbers behind masks and agent inputs | 5.2, 8.5 |
| Mobility predictor | Zone | The car's last GPS | T_exit and the next zone | 10.2 |
| Anchor table | Zone | Tasks the zone owns | Retries, cancels, the reward of each attempt | 9 |
| Replay buffer and trainer | Zone | Finished attempts (s, a, r, s′, Δt) | Updated network weights | 11.3, 13.5 |
| MQTT broker | Metro edge | Small control messages | Delivery to every subscriber | 5 |
| Cloud | Remote | Boulders and tasks sent there | Results; in phase 2 the averaged model | 3, 13.7 |

### What was run, against what this design plans

The sections describe the full plan. The experiments behind Appendix B used a smaller budget; every difference is listed here.

| Item | Planned in this design | Run for the results |
| --- | --- | --- |
| NSGA-II budget (13.2, 18) | Population 100, 100 generations, 20 windows per scenario | Population 40, 20 generations, 6 windows per scenario |
| Training budget (16.1) | The same for every learning method | 1 million decisions per method and scenario; several learners were still improving at the end |
| Deployed weight w\* (13.3) | Median knee weight | 0.414 on the VFC grid (27 of 36 windows have a knee), 0.344 on LuST (22 of 36) |
| Simulation runs (16.3) | 8 methods × 6 scenarios × 20 seeds | Done on both maps: 960 runs per map, 200 cars |
| Testbed runs (15, 16.3) | 5 seeds, 200–1 000 cars | 5 seeds, 40 cars, 60 s per run, VFC grid only |
| Federated learning, phase 2 (13.7) | Evaluated in simulation | Built and tested in simulation; the comparison of its three variants was not run: future work |
| Ablations and sweeps (16.4), α_TOF sweep (7.1) | The experiments listed in 16.4 | Not run: future work; α_TOF stays 0.8 |
| Topology experiments (15.7) | Zone join, zone kill, broker restart | Not run: future work |
| Task type T5 (14.1) | Optional | Off |

## 1. What PCNME is

PCNME (Predictive Cloud-Native Mobile Edge) places the computing tasks of moving cars across fog zones and the cloud, so that each task meets its deadline with the lowest latency and energy. A fixed rule (the TOF-Broker) sends tasks too heavy for any fog node straight to the cloud. A learned agent (D3QN) in every fog zone places every other task, knowing each nearby zone's load and where the car is going. The agent is pretrained from NSGA-II knee-point plans, so it is good from its first decision, and keeps learning online.

**The contribution.** The main novelty is the chain: an NSGA-II Pareto front over latency and energy, its knee point as the expert, and DQfD-style pretraining of a D3QN for task offloading. No paper found does this for offloading (closest: Wang et al. 2022, offline expert imitated online in vehicular edge computing; Zhang et al. 2025, a fuzzy expert cloned into a DQN; Nahhas et al. 2024, NSGA-III solutions as an imitation expert in factory scheduling). Supporting parts, not claimed as new: the TOF-Broker split (inspired by the pre-classification in Tof-NSGAII, Mokni & Yassa 2024, which splits by fog capacity; PCNME's rule uses the deadline, and the names boulder and pebble are PCNME's own), D3QN, GPS-based proactive placement (MAPSM, PODRL, MATM), DAG tasks. A messaging testbed with a measured gap to the simulation is a further strength, because most papers in this area evaluate in simulation only.

| Layer | What it is | In PCNME |
| --- | --- | --- |
| IoT (vehicle) layer | Connected cars and their sensors | Produce tasks (for example camera frames at 30 per second) and report GPS position, speed and heading |
| Access layer | RSU or 5G base station: the radio only | Receives the car's upload; its radio range is the zone's coverage |
| Fog layer | Fog zones, one per access point | Decides (TOF-Broker, agent) and executes (fog node) |
| Cloud layer | A remote data centre | Runs boulders and loose-deadline tasks; reached through one shared WAN pipe that can congest |

The MQTT broker cluster at the metro edge is messaging infrastructure, like the backhaul, not a fourth layer: the architecture stays IoT–Fog–Cloud.

**Names used everywhere:** fog node = the execution server (the literature also says edge, MEC or VEC server; it maps to an ETSI MEC host at the base station, N7). Fog zone = the whole component around it. Access point = RSU or base station.

## 2. Glossary

| Term | Meaning in PCNME |
| --- | --- |
| Fog zone | One site: an access point, its coverage area, a fog node, the TOF-Broker, the mobility predictor and the D3QN agent |
| Fog node | The execution server inside a fog zone |
| Access point | The RSU or 5G base station: the radio only |
| Anchor zone | The zone a task arrived in. It owns the task until it closes: input copy, decision, retries, result copy, ack, reward. It is never in the result's data path |
| Serving zone | The zone the car is attached to right now |
| Executor | The node that runs an attempt: the anchor's own node, a neighbour, or the cloud |
| Task | One job from a car (for example one camera frame). It may have internal steps (a DAG), but it is placed once and run whole on one node, in one slot |
| Attempt | One placement of a task. Attempt numbers 1, 2, 3 count the first placement and its retries. An attempt is named everywhere by its key (task_id, anchor_id, attempt) (5.5) |
| Pebble / boulder | A task light enough for a fog node / too heavy for any fog node. Boulders go to the cloud by rule |
| TOF-Broker | The fixed first filter at the zone entry: boulder if the mean run time on the strongest fog class's fastest slot type ≥ α_TOF · D |
| Slot, slot type | One place where one task runs at a time. Slot types: GPU slot (one inference stream on a GPU) and CPU slot (a fixed group of cores). A node has c_GPU ≥ 0 and c_CPU ≥ 0 slots |
| Time table | Measured run times (mean, p5, p95), longest step, and dynamic energy per run, per (task type, hardware class, slot type). A missing row = cannot run there |
| Hard / soft task | Hard: a late result is useless (a camera frame), so it is skipped or aborted when it cannot make its deadline. Soft: a late result still has value (analytics), so it is delivered late with a penalty, up to L_max past its deadline |
| D, slack, L_max | D = the task's deadline length from creation on the car. Slack = time left. L_max = D: how late a soft task may still be delivered |
| Budget | A time left, sent instead of a clock time (T^bud_car from the car, T^bud_k to an executor). No decision compares two machines' clocks |
| E_ref | Per task type: the dynamic energy of running it locally on the reference fog class. Energies are divided by it |
| w | The latency weight of the reward (the energy weight is 1 − w), from the NSGA-II knee point |
| Masking | Removing options that cannot work right now before the agent chooses |
| Neighbour position | One of the agent's 4 fixed neighbour outputs; an empty position is always masked |
| Link model | Every bulk transfer = wait for its shared FIFO pipe + size ÷ bandwidth + a random delay |
| Traffic intensity (TI) | Bits arriving per second ÷ a pipe's bandwidth. Below 1 the pipe keeps up; above 1 its queue grows without limit |
| Node load ρ_node | Task arrival rate × mean run time ÷ number of slots; at or above 1 the node's queue explodes |
| WAN pipe | The metro-edge router's uplink to the internet: one pipe shared by all zones. "Cloud congestion" means this pipe is full, not the cloud's compute |
| State store | A zone's in-memory copy of its own state and its neighbours' latest snapshots |
| Snapshot version | A per-zone pair (epoch, counter) on every state snapshot (state messages and task replies); newest version wins, epoch compared first (5.5) |
| Anchor table | A zone's in-memory list of the tasks it owns |
| Car registry | A zone's in-memory record of the cars it serves or anchors: last GPS, speed, heading, current zone |
| Exit time (T_exit) | Predicted time until the car leaves its zone: time until the car, going straight on at its speed and heading, crosses its zone's coverage circle (10.2); infinity when stopped |
| Attach | A car connecting to a zone's access point; the zone announces it on car/{id}/attach. In real 5G the network decides the handover from the car's signal reports |
| MQTT, broker, topic | MQTT is the protocol. The broker (EMQX) delivers each message published on a topic to every subscriber of that topic |
| QoS 0 / 1, retained, last will | At most once / at least once (may arrive twice). Retained = the broker keeps the last message for new subscribers. Last will = the broker publishes "offline" for a client that dies |
| Pareto front, knee point | Plans where no objective improves without the other getting worse; the knee = the front point farthest from the line through the two extremes |
| D3QN | Double + dueling deep Q-network: the agent's network |
| DQfD | Deep Q-learning from Demonstrations: pretraining with a margin loss on the expert's actions plus the TD loss |
| FedAvg | The cloud averages the zones' network weights, weighted by their new experiences |
| Fast mode | The same code on a discrete-event simulated clock with an in-process bus |
| Testbed | Docker containers, real MQTT and HTTP, emulated network (tc netem), real time |

### Notation

Symbols follow one rule: one base letter is one kind of quantity with one unit. The rules, every symbol, the names used before 15 Nov 2026 and the name of each symbol in the code are in Appendix A.

# Part I — PCNME in the real world

## 3. The fog zone and its components

> **Purpose.** Name every part of the system and say what each one holds. **Inputs.** None: this is the structure. **Outputs.** The vocabulary used by every later section; the table "The components at a glance" at the start of the document gives each part's input and output.

A fog zone is one physical site: the access point and the fog node sit together, so deciding at the zone costs no extra network hop. Assumption: one fog server per access point. Every zone has the same parts and runs the same agent network. A zone has up to 4 neighbour zones (section 7.5). All zones and the cloud connect to one MQTT broker cluster at the metro edge, a regional site a few ms from every zone. All zone → cloud traffic leaves through the metro-edge router's uplink.

> **Figure: PCNME in a city.** Cars upload over a radio uplink shared per zone to fog zones A, B, C (each: TOF-Broker, agent, GPU + CPU slots). Zones connect over the backhaul (1–5 ms, one outgoing pipe per zone) to the metro edge, which holds the metro-edge router (the shared WAN uplink) and the EMQX broker cluster (all control messages). The router reaches the cloud (many slots, never fills) over the WAN pipe: 25–100 ms each way, the narrowest link. Every cloud transfer shares that one WAN pipe.

Zones talk to each other over the backhaul and to everything else through the metro edge: control messages through the broker, cloud transfers through the one shared WAN uplink.

| Part | Role |
| --- | --- |
| Access point (RSU / 5G base station) | Radio link to the cars; its range is the zone's coverage. One uplink, shared by all its cars |
| Intake + TOF-Broker | Receives a complete upload, creates the anchor row, sorts the task: boulder → cloud, pebble → agent |
| Car registry + mobility predictor | Last GPS, speed, heading and zone of the cars it serves or anchors; predicts exit time and next zone on every GPS update |
| State store | Own state (exact) and each neighbour's newest snapshot by version, with its local receive time; pending sends; the freshest WAN-wait estimate |
| Neighbour table | Which zone sits in each of the 4 positions, its coverage circle, backhaul bandwidth and mean delay to it |
| D3QN agent | Places each pebble once: here, neighbour 1–4, or cloud (6 outputs), from 48 inputs, after masking |
| Anchor table | Tasks this zone owns: input copy, budget, attempts, pending transition, result copy, ack |
| Executor | A GPU queue and a CPU queue feeding fixed GPU and CPU slots; skip, abort and cancel rules; status replies |
| Online trainer + replay buffer | Pending transitions until their reward is final, then learning; never leaves the zone |
| MQTT client | One client per zone, with a last will on zone/{id}/status; one reference count per topic |
| Federated learning client | Uploads weights and downloads the global model each FedAvg round (phase 2) |

**The cloud** runs the same executor with many slots of each type it offers and a large queue, so it never fills under PCNME's load (assumption 17). Its delay comes from the WAN pipe and its own run time.

**The broker** is an EMQX cluster (for example 3 nodes), so one node failing is not an outage. A full broker outage is out of scope: without it, result delivery to a moved car, GPS to the anchors and state sharing stop, and there is no fallback (assumption 18). Decisions never wait for the broker: the agent reads its local state store.

## 4. Networks and the link model

> **Purpose.** Turn "send S bits over a link" into a time. **Inputs.** Size S (bit), bandwidth B (bit/s), the pipe's state t_free, the link's delay range. **Outputs.** The real transfer time T_transfer, and the estimates T̂_in,k (input to option k) and T̂_ret,k (result back to the car) that the masks and the agent use.

### 4.1 One model for every link that carries task data

```latex
T_{transfer} = W_{pipe} + \frac{S}{B} + \delta, \qquad \delta \sim \mathcal{U}(\delta_{min}, \delta_{max})
```

```latex
W_{pipe} = \max(0,\; t_{free} - t_{now}), \qquad t_{free} \leftarrow \max(t_{now}, t_{free}) + \frac{S}{B}
```

S = the size in bits, B = the pipe's bandwidth, δ = a random delay drawn per transfer, t_free = when the pipe is next free. The delay does not hold the pipe: the next transfer can be pushed while this one travels. Waiting costs time, never energy (section 12). A cloud transfer crosses two pipes in series, the zone's own backhaul pipe and the shared WAN pipe. Its packets flow through both at once, so it reaches the WAN pipe δ_bh after it starts on the backhaul pipe, and the WAN pipe, the narrowest, sets the push time: T = W_bh + δ_bh + W_WAN + S ÷ B_WAN + δ_WAN. Fast mode simulates it this way; on the testbed TCP through two links does it by itself. The radio uplink is the one exception: the cars uploading at once share its airtime instead of queueing (4.2, CHANGES C6).

| Symbol | Meaning | Unit |
| --- | --- | --- |
| S | Size of what is sent | bit |
| B | Bandwidth of the pipe | bit/s |
| δ | Delay drawn for this transfer, uniform between δ_min and δ_max | s |
| t_now, t_free | Present instant; instant the pipe is next free | s |
| W_pipe | Wait until the pipe is free | s |
| T_transfer | Time until the receiver has the whole payload | s |

**Why three terms.** S ÷ B is the time to push the bits, and it grows with the size. δ is the travel and processing time, and it does not depend on the size. W_pipe is the congestion: a pipe serves one transfer at a time, so a second task waits until the first has been pushed. Without W_pipe, sending every task to the cloud would look free; with it, a WAN whose traffic intensity passes 1 builds an ever-growing wait.

**Worked example.** A 76 KB frame is 608 000 bits. To a neighbour over the 1 Gbit/s backhaul: 0.6 ms of push plus a 1–5 ms delay, about 3.6 ms. To the cloud over the 36 Mbit/s WAN: 16.9 ms of push, plus 3 ms and 62.5 ms of mean delay, about 82 ms before any queue.

**Source.** The base paper [1] models the delay of a request as its size times a per-byte transmission delay, for the request and the response, plus a processing time (its Eq. 10–16), with no queueing and no propagation term. PCNME keeps that structure (size ÷ bandwidth for the push) and adds the propagation delay δ and the pipe wait W_pipe. The delay ranges come from the VFC environment (Appendix C [2]). The explicit first-in-first-out wait is PCNME's own addition.

### 4.2 Every link

| Link | Between | Bulk data on it | Delay δ | Bandwidth | Who shares the pipe |
| --- | --- | --- | --- | --- | --- |
| Radio uplink | car → access point | task input | U(1, 5) ms (18) | each upload pushed at that car's Shannon rate R_car from its distance; while n_up uploads are in progress each advances at R_car ÷ n_up | all cars of the zone: one uplink per zone, its airtime shared equally among the cars uploading at once (CHANGES C6) |
| Radio downlink | access point → car | none (results are about 3 KB) | U(1, 5) ms (18) | not modelled | — |
| Backhaul | zone → zone, zone → broker, zone → metro-edge router | task input to a neighbour; the first segment of a cloud transfer | 1–5 ms, random | 1 Gbps, fiber (18) | each zone has one outgoing pipe |
| WAN | metro-edge router → cloud | task input to the cloud; FedAvg weights | 25–100 ms one way, random | the narrowest, about 36 Mbps with the calibrated table (18); one value gives TI 0.5 / 0.9 / 1.5 in S1–S3 (16.2) | one pipe shared by all zones |
| MQTT messages | zone ↔ broker, cloud ↔ broker | none (hundreds of bytes) | the link's delay only | not used | — |

- **"50–200 ms to the cloud" is the round trip.** Each crossing draws 25–100 ms one way: the input pays one crossing (router → cloud), the result one (cloud → broker).
- **Results pay only delays:** they are small, so no bandwidth term. Their path: executor → broker → serving zone → radio.
- **The car's upload** happens before the decision and costs the same for every option, but it eats slack, so it is modelled.
- **The WAN is the narrowest pipe.** Example: 200 KB inputs on 100 Mbps = about 62 tasks per second for the whole city, while a 1 Gbps backhaul pushes about 625 per second per zone. This is the reason fog computing exists, not a flaw of PCNME: the agent decides which tasks deserve the WAN.
- **Connections are persistent and warm** (zones to zones, zones to the cloud), so TCP slow start does not distort transfer times. Stated as an assumption.

**The radio rate.** The rate of a car at distance ℓ from the access point follows from free-space path loss and the Shannon capacity, with f_c the carrier frequency, c the speed of light, G the antenna gains and N₀ the noise density:

```latex
R_{car} = B_{radio} \log_2\!\left(1 + \frac{P_{radio}\, G}{PL(\ell)\, N_0\, B_{radio}}\right), \qquad PL(\ell) = \left(\frac{4 \pi\, \ell\, f_c}{c}\right)^{2}
```

With 40 MHz, 1 W, 5 + 5 dBi, −174 dBm/Hz and 5.9 GHz this gives 930 Mbit/s at 10 m and 540 Mbit/s at 300 m. Source: the radio model of the VFC environment [2] and Shannon [13].

### 4.3 Known and estimated parts

Bandwidths and mean delays are configuration (provisioned in the real world). Only pipe waits change:

- **Own backhaul pipe:** known exactly, because the zone owns it.
- **Shared WAN pipe:** estimated. After each cloud upload the zone measures, on its own clock, send start → accept reply, subtracts its own backhaul part (the exactly known wait in its outgoing pipe + the mean backhaul delay), S ÷ B_WAN, the mean WAN delay and the reply delay, and updates a moving average Ŵ_WAN ← (1 − η) Ŵ_WAN + η · sample. Each zone publishes Ŵ_WAN and the age of its last sample in its state; a zone uses the freshest estimate among its own and its neighbours'. Before any sample, Ŵ_WAN = 0. The cloud does not publish it: it cannot see the router's queue.

Then the agent's network estimates are T̂_in,k = Ŵ^pipe_k + S ÷ B_k + δ̄_k (0 for here; for the cloud Ŵ^pipe = own backhaul pipe wait + Ŵ_WAN, B = B_WAN because the narrowest link sets the rate while packets flow through both links at once, and δ̄ = δ̄_bh + δ̄_WAN) and T̂_ret,k = the sum of mean delays on the result path.

## 5. Messaging and time

> **Purpose.** Fix what is sent, on which channel, and how deadlines survive unsynchronised clocks. **Inputs.** The events of a task's life (upload, placement, result, confirmation) and each node's state. **Outputs.** 8 MQTT topics for small control messages, 6 HTTP and WebSocket calls for bulk data and for the car, time budgets instead of clock times, and identifiers that need no central database.

### 5.1 Why MQTT, and what goes where

Publish/subscribe lets a car's anchor and its serving zone work without knowing each other's address: when the car changes zones, only subscriptions change. Pull (asking each zone) would make every decision wait; a central database would be a single point of failure on every decision. **Small control messages go over MQTT; bulk data (task inputs, model weights) goes over HTTP**. A task hand-off is one POST /tasks with no MQTT notice: the zone/{id}/work topic of the older docs is removed (27 Oct), because every field it carried already reaches the executor inside the POST and nothing ever subscribed to use it. Cars are never MQTT clients: a car talks only to its serving zone (HTTP upload, WebSocket for results).

Two rules apply to every message:

- **No clock time from another machine is used in any decision.** Messages carry budgets, durations and version numbers. A ts field may exist for logs only.
- **Every task message carries the attempt key: task id, anchor id and attempt number (5.5).**

### 5.2 MQTT topics

| Topic | QoS / retained | Publisher → subscribers | Fields |
| --- | --- | --- | --- |
| car/{id}/attach | 1 | Zone the car just attached to → the car's anchors and its previous serving zone | car_id, zone_id |
| car/{id}/gps | 0, retained with an MQTT 5 message expiry | Serving zone, every 100 ms → the car's anchors | car_id, x, y, speed, heading, zone_id, car_time (the car's clock; only differences between its own samples are used) |
| car/{id}/result | 1 | Executor, once → serving zone + anchor | kind = result: task_id, anchor_id, attempt, car_id, executor_id, slot_type, run_ms, energy_J, output (about 3 KB). kind = status: task_id, anchor_id, attempt, executor_id, skipped or aborted, energy_J spent |
| car/{id}/ack | 1 | Serving zone → anchor | task_id, anchor_id and attempt of the copy the car kept, car_id, zone_id, latency_ms measured by the car |
| zone/{id}/state | 0, retained, every 100 ms | Each zone and the cloud → neighbours | zone_id, version (epoch, counter); per slot type with c_q > 0: c, n_max, n_hard, n_soft, n_busy (busy slots), omega_hard_ms, omega_soft_ms, t_rem_ms (the remaining time of each of the c slots, 0 if free); wan_wait_ms and its sample age; load |
| zone/{id}/status | 1, retained, last will = offline | Each zone and the cloud → all zones | zone_id, status, x, y, radius |
| fl/round, fl/global | 1 | Cloud → all zones (phase 2) | round g, global version, upload deadline / version g ready |

### 5.3 HTTP calls

| Call | From → to | Body / reply |
| --- | --- | --- |
| POST /tasks | Car → its serving zone | task_id, car_id, class, type, input_size, T_bud_car_ms, R_car (radio rate, testbed) + the input. Reply: task_id, anchor_id (a repeated upload of a task_id this zone already holds gets the same reply, 9.7) |
| POST /tasks | Anchor → neighbour or cloud | task_id, attempt, car_id, anchor_id, class, type, input_size, budget_ms (T^bud_k) + the input. 200: slot_type, wait_ms, version, state snapshot. 503: reason (full or late), queue info, version |
| DELETE /tasks/{id}?attempt=n&anchor=A | Anchor → executor | Reply: removed, aborted, done or not found, plus energy_J spent |
| GET /tasks/{id}?attempt=n&anchor=A | Anchor → executor | Reply: queued (wait_ms), running (remaining_ms), done, skipped, aborted or not found |
| WebSocket /cars/{id} | Serving zone ↔ car | Zone pushes results (never status messages); the car confirms every copy with task_id, anchor_id, attempt, latency_ms |
| POST /fl/rounds/g, GET /fl/models/g | Zone ↔ cloud (phase 2) | Weights + number of new experiences / the global model |

### 5.4 Clocks: no synchronisation needed

No decision compares times taken on two machines. Allowed: a duration on one clock, a budget, a version number.

| Quantity | Clock used |
| --- | --- |
| Task deadline | Car sends T^bud_car = D − time since creation → anchor sets d_A on its clock → executor gets budget T^bud_k and sets d_k on its clock |
| Executor skip and abort | Executor only |
| Upload timeout, watchdog, probe | Anchor only |
| Age of a neighbour's state | Anchor only: now − local receive time + δ̄_pub |
| Newest snapshot | Version numbers |
| Car speed and heading | Differences between the car's own GPS samples |
| Latency L for the reward | The car: result arrival − task creation, on its clock, sent in its confirm |
| Waste of a failed attempt | Anchor only |
| Per-hop log breakdown | Analysis only: the testbed's shared host clock; NTP or PTP in the real world |

Clock-rate drift changes durations by parts per million, negligible for deadlines of 33 ms to seconds.

### 5.5 Identifiers

There is no central id database: every id is built from parts that are already unique, so no message, round trip or shared store is needed, and fast mode stays reproducible (no random UUIDs).

| Id | Made by | Form | Why |
| --- | --- | --- | --- |
| car_id | Assigned once | Real world: the vehicle's registered identity. Testbed: car container + virtual car number (for example c03-047) | Unique by construction |
| zone_id | Deployment configuration | z1 … z5; the cloud is "cloud". Executor id = zone id | Unique by construction; also the zone's MQTT client id |
| epoch | Each car app and each zone service, kept on its own disk | A counter raised by 1 at every start | A restart never reuses an old task id or snapshot version |
| task_id | The car, when it creates the task | (car_id, epoch, seq), seq counting up per car within an epoch; written c03-047.2.1187 | It exists before any zone has the task, so it stays the same when a cut upload goes to another zone (9.7) and across every attempt; the car's duplicate filter uses it (9.5) |
| attempt | The anchor | 1, 2, 3 per task | First placement and its retries |
| Attempt key | — | (task_id, anchor_id, attempt) | Unique even when two zones anchor the same task (9.7) |
| Snapshot version | Each zone and the cloud | (epoch, counter), compared epoch first | A restarted zone's new snapshots beat its old ones |
| FedAvg round g, global model version | The cloud | Counters | One owner. Every decision logs (global version, local update count) |
| run_id | The runner (testbed) or the experiment script (fast mode) | scenario, method, seed, start | Every log row carries it |

Every message about an attempt carries the attempt key: POST /tasks to an executor and its reply, DELETE and GET (?attempt=n&anchor=A), results, status messages, acks and the car's confirms. The executor keeps (attempt key, outcome) for T_keep. The anchor's pending buffer is local to the anchor, so (task_id, attempt) is enough there.

Why the anchor id is in the key: after a lost intake reply two zones may anchor the same task (9.7), and both call their first placement attempt 1. Keyed by (task_id, attempt) only, an executor chosen by both would treat the second upload as a repeat and ignore it, and one anchor's cancel could abort the other's attempt.

Why the epoch: a zone's snapshot counter restarts at 0 after a crash; without the epoch its neighbours would keep its old, higher version and ignore its new state for good. A car that restarts would reuse task ids the same way.

In fast mode every epoch is 0 and ids are deterministic, so the tie rule "then task id" in 8.2 (task ids compared as the tuple car_id, epoch, seq) gives the same order in every run with the same seed.

## 6. Lifecycle of a task

> **Purpose.** Follow one task from its creation to the reward. **Inputs.** A task created on a car. **Outputs.** A result delivered to the car (or a drop), a confirmation with the measured latency L, and one training sample per attempt. The three deadline equations of this section are the only place where time crosses from one machine to another.

Three loops run all the time: **GPS** (each car → its serving zone every 100 ms; the zone publishes car/{id}/gps), **state** (each zone publishes zone/{id}/state every 100 ms; neighbours keep the newest version), **learning** (agents learn from every closed attempt; FedAvg rounds in phase 2). A task then goes through five phases.

### A. Intake (at the anchor)

1. **Create.** The car creates the task: type (which fixes its time-table rows), class (hard or soft), deadline D, compressed input (JPEG per frame, N10). The car starts its own latency clock at creation.
2. **Upload.** The car uploads to its serving zone over the radio pipe, with T^bud_car = D − time since creation. A task exists only once the whole upload arrived (hole 12): that zone becomes the anchor, stores the input and answers task id + anchor id. A cut upload leaves nothing behind; the car re-uploads to the zone it reaches next.
3. **Anchor deadline.** The whole upload has arrived at t_rx (anchor clock). The anchor sets the task's deadline on its own clock, taking off the upload time: W_radio = the time from the upload's first bit to its last, minus S ÷ R_car (the extra time from sharing the air; known exactly, because the zone runs its uplink; CHANGES C6), S ÷ R_car = its push time alone at the car's rate, δ̄_radio = the radio delay:

```latex
d_A = t_{rx} + T^{bud}_{car} - \hat{T}_{radio}, \qquad \hat{T}_{radio} = W_{radio} + S / R_{car} + \bar{\delta}_{radio}
```

4. **Register.** The anchor table gets the task; the anchor raises its reference count on car/{id}/attach, gps, result and ack (section 9.6).
5. **Classify.** The TOF-Broker: boulder → the cloud (or a drop, section 7.1); pebble → the agent.

### B. Placement (once per attempt)

6. **Predict.** The mobility predictor gives exit time and likely next zone.
7. **Observe and mask.** The anchor builds the 48 inputs from its own memory and removes impossible options (section 7.2).
8. **Decide.** The agent picks one option. If every option is masked, the task is dropped (section 7.3).

### C. Execution (on one node)

9. **Send.** For a neighbour or the cloud, the input goes by HTTP with the budget

```latex
T^{bud}_k = d_A - t_{send} - \hat{T}_{in,k} - \hat{T}_{ret,k}
```

The anchor records the send as pending. The executor sets its local deadline d_k = arrival + T^bud_k. This is exact when the input takes its estimated time; if it takes longer (mainly a WAN-wait error), d_k is late by that error. The cost is bounded: a hard result that reaches the car after D counts as a drop (no ack by the end), so the error can waste a slot but never turns a miss into a success. For here, d_k = d_A − T̂_ret,here.

10. **Accept or refuse.** The executor picks the slot type and answers at once: accepted (slot type, its wait estimate, snapshot) or refused (full or late, snapshot). A refusal leads to one re-decision (section 9.2). After acceptance the anchor starts the watchdog.
11. **Queue and run.** The task waits in the queue of its slot type, then holds one slot for its whole run (section 8). A zone that accepted a task never forwards it.

### D. Delivery

12. **Publish once.** The executor publishes the result on car/{id}/result. The broker delivers it to the serving zone (which pushes it to the car over the radio) and to the anchor (which keeps a copy).
13. **Confirm.** The car keeps the first result per task id and confirms every copy with attempt and latency L. The serving zone publishes car/{id}/ack.
14. **Coverage gap or car moved.** Until the ack, the anchor republishes its copy on every new attach; if no ack comes within T_ack after the result, it republishes once (section 9.5).

### E. Close and learn

15. **Close.** On the ack, or on a drop, the anchor closes the task: cancels any other live attempt, deletes the input and result copy, and lowers its subscription counts G later, so late energy reports still reach it (step 16; CHANGES C8).
16. **Reward.** Every attempt gets its reward when the task is closed plus a grace period G (section 11.4); its transition moves from the pending buffer to the replay buffer.

## 7. The decision maker

> **Purpose.** Choose where a task runs. **Inputs.** The task, the time table, the state store, the car's predicted exit time. **Outputs.** One of 6 placements, or a drop when nothing can work. It has three stages in a fixed order: a rule (TOF-Broker), a filter (masks), and a learned choice (the agent).

Order of checks for every task: intake → TOF-Broker → masks → agent → send and accept → run → delivery → reward. Fixed rules are written in code; only the choice among unmasked options is learned.

### 7.1 TOF-Broker [fixed]

1. **Boulder** if T̄^run_strong ≥ α_TOF · D, where T̄^run_strong is the mean run time of the task type on the strongest fog hardware class, on its fastest slot type. "Strongest fog class" is a deployment constant, so the rule needs no fog state. Inspired by Tof-NSGAII (Mokni & Yassa 2024), which marks a task "intensive" when its cost ÷ the fog node's capacity ≥ 1. PCNME compares with the deadline instead, because a task can fit a fog node and still miss its deadline there.
2. **α_TOF = 0.8 by default**: if even the best fog run takes 80 % of D, less than 20 % is left for queue, transfer and return. The final α_TOF comes from a sweep {0.5, 0.65, 0.8, 1.0} in fast mode on S2 and S4 (best on-time rate).
3. **Boulders go to the cloud**, never to a fog queue. A boulder is dropped at intake when the cloud fails the same "cannot meet the deadline" test as the mask (7.2): T̂_in,cloud + T̄^run_cloud + T̂_ret,cloud > time left for a hard task, > time left + L_max for a soft one (the cloud never queues). No agent decision is made, so no transition is stored.

**The rule as an equation.**

```latex
\text{boulder} \iff \bar{T}^{run}_{strong} \;\ge\; \alpha_{TOF} \cdot D, \qquad \alpha_{TOF} = 0.8
```

**Worked example.** A 60-frame clip (T2) runs 3.87 s on a fog GPU slot. With the loose deadline D = 6 s the threshold is 4.8 s, so it is a pebble and the agent places it. With the tight deadline D = 4.26 s (1.1 × its run time) the threshold is 3.40 s, so it is a boulder and goes to the cloud, where it runs in 0.66 s. A frame (T1: 22.6 ms against D = 50–100 ms) is always a pebble.

**Source.** Sorting tasks by their requirement before the optimiser comes from Tof-NSGAII [4], which splits by fog capacity. Splitting by deadline, and the names boulder and pebble, are PCNME's own.

### 7.2 Masks [fixed]

For each of the 6 options the anchor first finds the usable slot types 𝒬_k (c_q > 0, a time-table row, a free place for the task's class), and assumes the executor will pick the earliest finish:

```latex
q^*_k = \arg\min_{q \in \mathcal{Q}_k} \left( \hat{W}^{queue}_{k,q} + \bar{T}^{run}_{k,q} \right)
```

An option is masked when:

| Mask | Rule |
| --- | --- |
| Empty | No zone in that neighbour position |
| Offline | Its last will said offline |
| Stale | Its newest snapshot is older than 3 × the state period (300 ms at 100 ms), measured from the local receive time |
| Full (with headroom) | No usable slot type has room as the anchor knows it (pending sends included): n_q ≥ n^max_q − n_head for hard, ≥ n^max_q − n_res − n_head for soft. The anchor's own queue is exact (n_head = 0). The cloud never fills |
| Cannot run | 𝒬_k is empty because no slot type there can run the task (for example a GPU-only task, a node without a GPU) |
| Cannot meet the deadline | Hard: T̂_in + Ŵ^queue + T̄^run + T̂_ret on q\* > time left. Soft: the same against time left + L_max |

**The deadline mask as one inequality.** With t_left = d_A − t_now the time left on the anchor's clock, option k stays only if

```latex
\hat{T}_{in,k} + \hat{W}^{queue}_{k,q^*} + \bar{T}^{run}_{k,q^*} + \hat{T}_{ret,k} \;\le\; t_{left} \;(\text{hard}), \qquad \le\; t_{left} + L_{max} \;(\text{soft})
```

Each term comes from one model: T̂_in and T̂_ret from the link model (4.3), Ŵ^queue from the queue model (8.5), T̄^run from the time table (14.4).

### 7.3 The agent's choice [learned]

The agent scores all 6 options and takes the best unmasked one. What a good agent ends up doing (intuition, not code): run here when it finishes in time cheaply (C2); run in the zone the car is heading to when the car would leave first, with a margin (C3); run on a less busy neighbour when it finishes earlier even after the transfer (C4); use the cloud only when it is truly better and the deadline allows the round trip (C5). C3 and C4 are the same action with different reasons; the cases explain why an action is picked, they are not separate mechanisms.

**If every option is masked**, the task is dropped at once, hard or soft, with the drop reward logged; no transition is stored, because no decision was made.

### 7.4 Every case

| Case | Condition | What happens | Who |
| --- | --- | --- | --- |
| C1 Boulder | Mean run on the strongest fog class ≥ α_TOF · D | Cloud, or drop if the cloud cannot make it either | TOF-Broker |
| C2 Run here | The anchor finishes in time, the car stays | Its own node | Agent |
| C3 Car leaving | The result would be ready after the car exits | The zone the car is heading to | Agent + predictor |
| C4 Load balancing | A neighbour finishes earlier, transfer included | That neighbour | Agent |
| C5 Cloud | Every fog is too late or too costly, the deadline allows the round trip | Cloud | Agent |
| C7 Car moved | The car attached elsewhere during the task | The new zone subscribed on attach and delivers; the old one unsubscribes (section 9.6) | Topics + anchor |
| C8 Car stopped | Speed near 0 | Exit time = infinity: local is safe | Predictor |
| C9 Deadline | Cannot or did not finish in time | Hard: refused, skipped or aborted. Soft: delivered late up to L_max, then dropped | Executor + anchor |
| C10 Masked | Full, offline, stale, empty, cannot run, cannot meet | Removed before the choice | Masks |
| C12 Coverage gap | The car is in no zone when the result is ready | The anchor republishes on the next attach; dropped after the deadline (hard) or L_max (soft) | Anchor |
| C13 Two zones | The car is inside two coverage areas | It stays on the stronger access point (hysteresis) | Access layer |
| C14 Failure | Upload error, executor dead, task lost | Retry the whole task (1 hard, 2 soft) if the retry mask allows, else drop | Anchor |

C6 (cloud super task) and C11 (steps split across nodes) were removed on 13 October; the numbers stay so references are stable.

### 7.5 Who is a neighbour

1. Candidates: every zone whose coverage circle overlaps this one or whose centre is within ℓ_nb, from the retained zone/{id}/status messages. The cloud is never a neighbour position.
2. Up to 4 positions, nearest first, fixed at start-up. A zone that joins takes the lowest free position of each zone that counts it as a candidate; a zone with 4 neighbours keeps them. Positions never change during a run, so the inputs keep their meaning.
3. An offline neighbour keeps its position, masked, until it is online again.
4. Not symmetric: A may count B while B does not count A. Any zone accepts tasks from any anchor, so nothing breaks.
5. A joining zone downloads the current global model before it announces "online".
6. In training, which neighbour sits in which position is shuffled per episode (section 13.6), so the network treats the 4 positions alike.

## 8. Queues and execution

> **Purpose.** Say how a node runs the tasks placed on it and how long a new task will wait. **Inputs.** The tasks placed on the node, each with its class, deadline and slot types; the time table. **Outputs.** Which task each free slot takes; accept or refuse; the published queue state (n^busy_q, T^rem_q,j, Ω_q); the wait estimate Ŵ^queue_q that masks and agent use.

Queue waiting is what makes overload real. It comes from fixed slots on every node and from the shared pipes (section 4). Every node, fog or cloud, runs the same scheduler, in fast mode and on the testbed. See it animated in the Fog Node Queue Lab (example numbers).

### 8.1 Slots

1. Each node has c_GPU ≥ 0 GPU slots and c_CPU ≥ 0 CPU slots (a node without a GPU has c_GPU = 0). A CPU slot is a fixed group of cores; a GPU slot is one inference stream (a GPU may give 1 or 2). Times are measured at exactly that size.
2. **One task holds one slot for its whole run.** Its DAG is inside its measured time; steps never move between slots or nodes.
3. **No pause and resume.** A real GPU inference cannot be paused and resumed cleanly, and the real-world layer must be buildable. Aborting for good is allowed (8.6).

### 8.2 Queues

1. One queue per slot type on each node.
2. Order: hard tasks first, then soft. Inside each class, earliest deadline first on the executor's own d_k. Ties: arrival order, then task id (reproducible runs).
3. Capacity n^max_q per queue. The last n_res places are for hard tasks: a soft task is admitted only while the queue holds fewer than n^max_q − n_res.
4. Earliest deadline first cannot starve a task inside its class (a newcomer goes ahead only with an earlier deadline, so a task's wait is bounded by its own fixed deadline, after which it is skipped or, if soft, runs late), so aging is needed only between the classes (8.4, step 2).

### 8.3 Choosing the slot type on arrival

1. For every usable slot type q: estimated finish F_q = Ŵ^queue_q + T̄^run_q (the executor's own estimate, from its exact queue).
2. The task joins the queue with the smallest F_q. The executor chooses by time, not energy; the agent makes the energy trade-off between nodes through w.
3. Refused **"late"**: a hard task whose best F_q misses d_k; a soft task whose best F_q is later than d_k + L_max; a task no slot type here can run. Refused **"full"**: no usable queue has a place for its class.
4. The accept reply carries the chosen slot type and Ŵ^queue_q as a duration.

| Situation | What happens |
| --- | --- |
| No GPU on this node | Only CPU rows are used; a GPU-only task is masked by the anchor, refused "late" if it arrives anyway |
| GPU busy, CPU free | CPU if that finishes earlier, else wait for the GPU |
| CPU busy, GPU free | The same the other way round |
| Everything busy | The queue that finishes earliest; no place → "full"; hard and too late everywhere → "late" |
| One slot type frees early | Work stealing (8.4, step 5) |

### 8.4 Which task a free slot takes

1. **Skip** hopeless heads: a hard task with t + T^run,lo_q > d_k; a soft task with t + T^run,lo_q > d_k + L_max. Each skip is reported at once (8.6).
2. **Soft promotion:** the first soft task i runs first if it waited at least T_age and a check proves every queued hard task still meets its deadline with i taking this slot for T^run,hi_q(i) (the hard queue is list-scheduled on the c_q slots).
3. Otherwise the first hard task.
4. Otherwise the first soft task.
5. **Work stealing:** if this queue is empty, the first task in the other queue's order that can run here and would finish earlier here (t + T̄^run_q < its current estimated finish), and passes the skip rule.
6. Otherwise the slot idles.

Stealing only makes tasks finish earlier, so every estimate the anchor holds stays on the safe side. Promotion never makes a queued hard task miss.

### 8.5 What a node publishes and how the anchor estimates the wait

Per slot type with c_q > 0: c_q, n^max_q, n_q^hard, n_q^soft, Ω_q^hard and Ω_q^soft (sum of the mean run times of the queued tasks), T^rem_q,j (the remaining mean time of the task on each slot j of the c_q slots, 0 for a free slot), n^busy_q (busy slots). The anchor's estimate is 0 while a slot is free (n^busy_q + its own pending sends to q < c_q), and otherwise the water level Ŵ^queue_q at which the class backlog, poured onto the slots in the order they free up, fills them all (CHANGES C4):

```latex
\sum_{j=1}^{c_q} \max(0,\; \hat{W}^{queue,hard}_q - T^{rem}_{q,j}) = \Omega_q^{hard}, \qquad \sum_{j=1}^{c_q} \max(0,\; \hat{W}^{queue,soft}_q - T^{rem}_{q,j}) = \Omega_q^{hard} + \Omega_q^{soft}
```

For c_q = 1 this is T^rem + Ω, exact in the mean; with equal T^rem_q,j it equals the pooled estimate (the sum of the remaining times plus Ω, divided by c_q), which it replaces because one long run (a 3.9 s T2 on one GPU slot) would be spread over every slot and the wait overestimated several times over. It treats queued work as divisible, so it can underestimate by up to one task's run per slot. The hard estimate assumes the new task waits behind every queued hard task. The soft one ignores hard tasks that arrive later (optimistic); the status probe covers that (section 9.3). The anchor's own pending sends (R1) are added to the backlog of their class on their q*. The executor's own estimate (8.3) is the same function on its exact state, without pending sends.

**Worked example.** A slot type has c_q = 2 slots with remaining times T^rem = 10 ms and 30 ms, and a backlog Ω_q = 40 ms. The equation asks for the level Ŵ at which the backlog fills both slots: (Ŵ − 10) + (Ŵ − 30) = 40, so Ŵ = 40 ms. Check: the first slot is free at 10 ms and works on the backlog for 30 ms; the second is free at 30 ms and works for 10 ms; together 40 ms of backlog, finished at 40 ms. Pooling the two slots as one would give (10 + 30 + 40) ÷ 2 = 40 ms here, but with remaining times of 5 ms and 80 ms and the same backlog the water level is 45 ms (the first slot alone clears the backlog) while pooling says 62.5 ms: pooling makes the task wait for a slot it does not need.

**Why an estimate is enough.** The estimate uses mean run times and ignores later arrivals with earlier deadlines. Its error is corrected twice: the executor checks the deadline again on arrival and refuses if it cannot hold (9.2), and the agent learns from the reward what the estimate hides.

### 8.6 Dropping and aborting

An abort stops a task for good: the worker stops at the end of the DAG step it is in (for YOLO: preprocess, inference, post-process) and throws the output away; the model stays loaded. So dropping a GPU task is safe. The slot stays busy for min(remaining, T_abort), where T_abort is the longest step of that task on that slot type.

| Where | Hard task | Soft task |
| --- | --- | --- |
| On arrival | Refused "full" or "late" | Refused "full", or "late" past deadline + L_max |
| Head of the queue | Skipped when t + T^run,lo > d_k | Skipped when t + T^run,lo > d_k + L_max |
| While running | Aborted when t > d_k | Aborted when t > d_k + L_max |
| Anchor cancel | Removed if queued, aborted if running | Same |
| No ack by the end | Drop (deadline) | Drop (deadline + L_max) |

- **Every drop is reported at once** as a status message on car/{id}/result {task id, attempt, skipped or aborted, energy spent}. The anchor subscribes to that topic anyway; the serving zone ignores status messages, so the car never sees them.
- **No leftovers:** the executor deletes the input, frees the place (or the slot after T_abort) and keeps (attempt key, outcome) for T_keep (5.5), to answer a late probe or cancel and to ignore a repeated upload of the same attempt.

### 8.7 Checks

Queue orders compared (cheap experiment): FIFO; classes with FIFO inside; the full rule. Slots, slot choice and stealing stay the same, so only the order differs. One slot type, one class, Poisson arrivals and exponential times must match the M/M/c waiting time (Erlang C); with c = 1, the M/M/1 check W = λ ÷ (μ(μ − λ)).

## 9. Failures, refusals, retries and cleanup

> **Purpose.** Make every failure end in a defined state. **Inputs.** An attempt that was refused, lost, late or duplicated; a zone or a car that disappeared. **Outputs.** A retry, a cancel or a drop, each with its reward, and no task, slot or subscription left behind.

### 9.1 One table of failure rules

Each failure has its own rule because each has a different cause.

| What went wrong | Rule |
| --- | --- |
| Target queue full or too late on arrival | Refused at once; the anchor re-decides once; refused again or no slack → hard: drop; soft: cloud, unless the cloud cannot finish by deadline + L_max either, then drop |
| Upload to the executor fails (error, no reply) | Treated as not placed: re-decide as above (counts as the one re-decision) and cancel the first target, in case only its reply was lost |
| Executor zone dies | Its last will → every task the anchor placed there is handled at once: retry if the retry mask allows, else drop |
| Task silently lost | Watchdog → status probe → retry if the retry mask allows, else drop |
| Deadline cannot be met | Executor skips or aborts (hard at d_k, soft at d_k + L_max) and reports at once |
| Wrong mobility prediction | Nothing to retry: the task runs, the result follows the car, the extra delay is in the reward. A wrong prediction is never a failure |
| Car in a coverage gap | Hold and republish on the next attach; drop after the deadline (hard) or L_max (soft) |
| Upload cut by a handover or gap | No task exists yet; the car re-uploads to the next zone with its new budget (9.7) |

Retries: the whole task from the anchor's stored input, at most once for hard and twice for soft. **Retry mask:** a retry is a new decision with the normal masks; a hard task is retried only if some option passes "can meet the deadline". No separate rule.

### 9.2 Refusal and herding

The anchor decides from neighbour state up to one state period old, plus the broker's delivery delay, and several anchors may pick the same quiet zone at once (herding). Shrinking the state period does not stop herding: it happens even with fresh state, and 10 ms state would raise broker load tenfold. Two zero-message rules do the work:

- **R1, fresh state from replies and own pending sends.** Every zone keeps one snapshot version (epoch, counter; section 5.5) whose counter grows on every snapshot it sends: each state message and each POST /tasks reply. The anchor keeps, per neighbour, the snapshot with the highest version, from either path. A task sent but not yet answered counts as pending in the anchor's copy of that neighbour (its mean run time added to the backlog of its class), until the reply arrives or the upload times out. A rare double count (a state message that already includes a pending task) only overestimates, which is safe.
- **R2, mask with headroom n_head** (section 7.2).
- **R3, event-driven state** (publish when the queue changed by Δ, at most every 10 ms, heartbeat 100 ms) is an extension, tested in the state-period experiment (10 / 100 / 500 ms, R3 on and off; metrics: refusal rate, on-time rate, broker messages per second).

A refusal costs a waste penalty in the reward (section 11.4), which is what teaches the agent to avoid herding; there is no separate herding term. The refusal rate is logged in every run.

### 9.3 Detecting a lost attempt

- **Upload timeout:**

```latex
T^{to}_k = \min\big(\kappa_{up}\,(\hat{T}_{in,k} + \bar{\delta}_{reply,k}),\; d_A - t_{send} + L_{late}\big), \qquad L_{late} = 0 \text{ (hard)},\; L_{max} \text{ (soft)}
```

An upload still being sent is not a failure (CHANGES C7): while the zone's own send is going (the input has not yet reached the executor), only the deadline cap d_A + L_late applies, and reaching it drops the attempt with no re-send. T^to_k is counted from the end of that send; only a reply missing after it counts as "no reply" (9.1).

- **Last will:** zone/{id}/status = offline, immediate.
- **Watchdog**, started at the accept reply (t_acc), using the executor's own wait estimate Ŵ^queue from that reply:

```latex
t_{wd} = t_{acc} - \bar{\delta}_{reply,k} + \hat{W}^{queue} + T^{run,hi}_k + \bar{\delta}_{pub,k} + m_{wd}
```

t_wd is capped at d_A for a hard task and at d_A + L_max for a soft one: after that the task is dropped anyway. The result (or a status report) arriving stops it.

- **Status probe before any retry.** A queued task can move back when hard tasks with earlier deadlines arrive, so a fired watchdog does not prove a loss. If the executor is alive, the anchor sends GET /tasks/{id}?attempt=n&anchor=A and waits at most T_probe: queued → re-arm with the new estimate (a hard task whose new finish is after d_A is retried elsewhere if the retry mask allows, and this attempt cancelled); running → re-arm with the remaining T^run,hi; done → wait δ̄_pub + m_wd; skipped, aborted, not found or no reply → lost.

### 9.4 Retry and cancel

1. Every placement has an attempt number; POST, results, status messages, acks, probes and cancels carry it together with the task id and the anchor id (the attempt key, 5.5).
2. Cancel = DELETE /tasks/{id}?attempt=n&anchor=A from the anchor to the old executor: queued → removed; running → aborted; unknown or finished → ignored. Sending it twice is harmless.
3. **First result wins.** The anchor keeps the first result and cancels any attempt still alive.
4. **Best effort.** A lost cancel costs one wasted slot and a duplicate result that the car filters. Correctness never depends on it.

### 9.5 Duplicate results at the car

Duplicates come from two finished attempts, a republish after an attach, and QoS 1 ("at least once").

1. The car keeps the first result per task id and keeps the id until deadline + L_max.
2. The car confirms every copy (task id, anchor id and attempt of the copy it kept, latency L). A confirm is idempotent, so a lost ack is repaired by the next copy.
3. The serving zone does not push a result again if the car already confirmed that task id in this radio session.
4. The anchor ignores repeat acks for a closed task. If it saw the result but got no ack within T_ack, it republishes once.
5. No ack by the end (deadline for hard, deadline + L_max for soft) counts as a drop for the reward. A car that got the result while every ack was lost is very rare; the testbed counts it from the car's own log.

### 9.6 Subscriptions and cleanup

| Role | Subscribes to | When | Unsubscribes when |
| --- | --- | --- | --- |
| Serving zone | car/{id}/result, car/{id}/attach | The car attaches | Another zone announces an attach for the car, or no GPS for T_leave |
| Anchor | car/{id}/attach, gps, result, ack | Its first open task for the car | G after its last task for the car closes (late energy reports, 11.3.3; CHANGES C8) |
| Executor | Nothing | — | — |

- **One reference count per (zone, topic)**, because one MQTT client serves all roles: subscribe at 0 → 1, unsubscribe at 1 → 0.
- **An early unsubscribe is safe:** the anchor republishes on the next attach. Cleanup can delay a result, never lose it.
- **The car registry follows the serving role**; the anchor drops a car's predictor state when its count reaches 0.
- **Retained GPS expires** (MQTT 5 message expiry), so the broker does not keep every departed car forever.
- **Zones connect with a clean start**, so a restarted zone has no stale subscriptions. **No wildcards** (car/+/result would send every result in the city to every zone).
- **A car always knows whether its serving zone serves it** (testbed transport detail, QUESTIONS Q61). The car repeats its attach until the zone confirms it, and the zone's reply to the car's GPS names the cars it does not serve (T_leave fired, for example on a GPS that arrived late), which then attach again. Without this, a zone that dropped a car still in range never served it again and every later result of that car was lost. Fast mode, where GPS is never late, needs neither.

### 9.7 An upload cut by a handover or a gap

1. A task exists only once its whole upload arrived; a cut upload is discarded with no reward and no transition.
2. The car keeps the frame until a zone accepted it, and re-uploads to the zone it reaches next, with the same task id and its new T^bud_car. That zone is the anchor. If the car is still in the same zone but its connection breaks before the reply, it re-uploads there with the same task id (a reply on a live TCP connection is retransmitted, not lost). No timer: the car cannot see the wait in the zone's radio pipe, so a timer would fire early under load and re-send into the same full pipe; an intake that already holds that task id (open, or closed within T_keep) creates no second row and sends the same reply again.
3. The car gives up locally when the budget is gone (hard: below the fastest possible upload + run + return; soft: past deadline + L_max), logged as a car-side drop.
4. If zone A got the whole upload but its reply was lost and the car has already moved to zone B, B also becomes an anchor (inside one zone a repeated upload is recognised by its task id, step 2). Both results reach the car, which keeps the first; its confirm names the anchor whose copy it kept, so that anchor's attempt is scored as delivered and the other anchor's as "its result came second". Both anchors learn from a real decision, and the attempt key keeps their attempts apart at any executor (5.5). Rare and logged; checking with A first would add a round trip to every re-upload.

### 9.8 A dead anchor

If the anchor zone dies, the tasks it owns are lost: the car may still get a result, but no reward is recorded (assumption 16).

## 10. Mobility and result delivery

> **Purpose.** Deliver the result to a car that has moved, and tell the agent where the car is going. **Inputs.** The car's GPS (position x_car, velocity v) every 100 ms; the zones' coverage circles. **Outputs.** The exit time T_exit and the next zone (6 of the agent's 48 inputs); the result pushed by whichever zone serves the car when it is ready.

The result is published once and the broker delivers it to whichever zone serves the car, so every result can find the car. The prediction only decides where to run a task: a wrong prediction costs time, never the result.

### 10.1 GPS and the anchor

- Every 100 ms the car sends position, speed and heading to its serving zone, which publishes car/{id}/gps. At 50 km/h a car moves about 1.4 m in 100 ms; car GPS receivers give 1–10 positions per second.
- The anchors of the car's open tasks receive it wherever the car is, so they always know its current zone. The car never needs to know where its anchors are.
- The anchor recomputes the prediction on every GPS update. A running task is never moved (live migration is out of scope).
- In a real 5G network the core already knows each car's cell; the anchor's GPS tracking is PCNME's explicit version of that, and GPS is what makes prediction possible.
- At city scale GPS is the largest MQTT load: it grows with cars, not zones (10 000 cars × 10 updates per second = 100 000 messages per second). Ways to reduce it: fewer updates far from a zone edge, or GPS only for cars with open tasks. Cite EMQX cluster benchmarks; do not claim unchecked numbers.

### 10.2 The mobility predictor

Supervised, not RL: where the car went is known afterwards. A swappable module:

1. Motion math: exit time = when the car, going straight on from position x_car with velocity v, crosses its zone's coverage circle (centre x_z, radius ℓ_z): T_exit = (−b + √(b² + |v|² (ℓ_z² − |x_car − x_z|²))) ÷ |v|², with b = (x_car − x_z) · v; infinity when stopped (C8). Inside a circle every heading leaves it, so moving "away from the edge" never gives infinity.
2. Motion math + road grid: at an intersection only 2–3 next zones are possible. **PCNME starts here.**
3. LSTM/GRU on GPS history: an extension, only with realistic routes (real traces or SUMO); random turns cannot be learned. Learned trajectory prediction (MATM, PODRL, MAPSM) is cited, not claimed.

Proactive placement in the next zone (C3) needs a safety margin: the car must clearly leave before the result is ready.

**Where the exit-time formula comes from.** The car at x_car moving with velocity v is at x_car + v·t after t seconds. It leaves the zone when its distance to the centre x_z equals the radius ℓ_z: |x_car + v·t − x_z|² = ℓ_z². This is a quadratic in t, and T_exit is its positive root. With v = 0 there is no root, so T_exit is infinite: a stopped car never leaves.

### 10.3 How a result reaches the car

| Where it ran | Path |
| --- | --- |
| The anchor's node | Anchor publishes → broker → serving zone → car |
| A neighbour | Neighbour publishes → broker → serving zone → car; the anchor gets a copy |
| The cloud | Cloud publishes (one WAN crossing) → broker → serving zone → car; the anchor gets a copy. The cloud has no radio and never predicts |
| Car moved during the run | The same: the new zone subscribed on attach (C7) |
| Car in no zone | No ack: the anchor republishes on the next attach (C12) |
| Car in two zones | The zone it is attached to delivers (C13) |

Why the executor publishes, not the anchor: it saves one backhaul hop (2–5 ms, up to 15 % of a 33 ms deadline), and the anchor still gets the copy from the same topic.

### 10.4 Three levels of handoff

| Level | What | Status |
| --- | --- | --- |
| Before execution | Place the task where the car is heading (C3) | In scope |
| Result delivery | The result follows the car through car/{id}/result (C7) | In scope |
| During execution | Move a half-run task (live migration) | Future work |

## 11. The agent

> **Purpose.** Learn which placement is best in each situation. **Inputs.** The state s: 48 numbers built from the zone's own memory. **Outputs.** 6 action values Q(s, a); the action is the unmasked option with the largest value. **Feedback.** One reward r per attempt, known when the task closes.

One D3QN (double + dueling deep Q-network) per fog zone, all starting from the same pretrained network. No agent on the cars and no global agent. An LSTM is added only if an experiment shows that load history helps.

### 11.1 Inputs: 48 numbers

All built from the zone's own memory, with no network call. Times are divided by the task's D, energies by E_ref, so every input is dimensionless. Time inputs are capped at 2 (= 1 + L_max/D: an option needing longer is masked anyway) and energy ratios at C^en_max (12.3), so a congested pipe cannot blow up the inputs.

**Task (5)**

| Input | Meaning |
| --- | --- |
| input_size | Input size ÷ the largest input size |
| ref_exec | Mean run time on the reference fog class, on its fastest slot type ÷ D |
| slack | (d_A − now) ÷ D |
| hard | 1 = hard, 0 = soft |
| D_norm | D ÷ the largest deadline of any task type: the only input that says how long D is, next to fixed costs like the WAN delay |

**Per option, 6 × 6 = 36** (order: here, neighbour positions 1–4, cloud)

| Input | Meaning |
| --- | --- |
| present | 1 = usable; 0 = empty, offline or stale (also masked) |
| wait | Ŵ^queue on q\* for the task's class (section 8.5), with own pending sends ÷ D |
| exec | Mean run time on q\* ÷ D |
| network | (T̂_in,k + T̂_ret,k) ÷ D: pipe wait + size ÷ bandwidth + mean delay, there and back; for here only the return |
| energy | min(Ê_k ÷ E_ref, C^en_max) ÷ C^en_max (sections 12.3, 12.4) |
| age | Age of that node's snapshot ÷ (3 × state period); 0 for here |

**Mobility (6):** exit = T_exit ÷ D, capped at 1 (a stopped car = 1); next_zone = one-hot over 5 (stays, or neighbour position 1–4).

**Trade-off (1):** w (section 13.3).

A masked option still gets its numbers (zeros where unknown); the mask removes it from the choice. The DAG appears only through ref_exec and exec, which already contain it.

### 11.2 Outputs

6 Q-values in the input's order. The agent takes the highest unmasked one. Boulders never reach it.

| Action | What the zone does |
| --- | --- |
| 0 Here | Its own queue |
| 1–4 Neighbour k | POST /tasks to the zone in position k |
| 5 Cloud | POST /tasks to the cloud through the WAN pipe |

**The network.** Two hidden layers of 256 units feed two heads: a value V(s) and an advantage A(s, a) for each of the 6 actions. They are combined as

```latex
Q(s, a) = V(s) + A(s, a) - \frac{1}{6} \sum_{a'} A(s, a')
```

V says how good the situation is whatever is chosen; A says how much better one option is than the average option. Separating them helps here because in many states (an idle system) every option is fine and only V matters. Source: the dueling network [18].

### 11.3 The RL step

1. One step = one placement decision at an anchor: a task's first decision, or a re-decision after a refusal or a lost attempt.
2. s = the anchor's state at the decision, a = the chosen option, s' = the same anchor's state at its next decision, Δt = the time between the two decisions.
3. The reward comes later than s'. The transition waits in a pending buffer keyed by (task id, attempt) and moves to the replay buffer when its reward is final: when the task closes plus a grace period G for late energy reports.
4. No terminal state: the system runs continuously. A fast-mode episode ends by a time limit, so the last transitions still bootstrap (truncation).
5. Boulders and all-masked drops make no transition. A re-decision gets no extra input: its smaller slack shows it.
6. **Time-based discount:** γ^(Δt/τ) with τ = 100 ms and γ = 0.9 per τ to start (horizon about 1 s, the time a queue needs to drain). A fixed γ per decision would look less far ahead exactly when the load is high and decisions come fast. This is semi-Markov Q-learning. Ablation: γ = 0. To watch in the first fast-mode runs: with thousands of decisions per second in a zone, γ^(Δt/τ) is close to 1 per step, so Q-values grow to the sum of about one second of rewards and change with the load. Only the differences between actions matter, and the dueling head separates them, but if learning is unstable: Huber loss, a smaller learning rate, and γ = 0 as the fallback.
7. **Target (double DQN, masked):**

```latex
y = r + \gamma^{\Delta t/\tau}\; Q_{target}\big(s',\; \arg\max_{a' \in \mathcal{A}(s')} Q(s', a')\big)
```

The argmax and the target use only the actions valid in s', with the same masks as acting.

### 11.4 Rewards: one per attempt

w and 1 − w come from the knee (section 13.3). Every reward is r_ok minus the same per-task cost that NSGA-II minimises (13.2): w · C^time + (1 − w) · C^en, with C^en = min(E/E_ref, C^en_max). r_ok = 1, C_pen = 2 and w3 = 1 are fixed. A drop is charged the worst time cost C_drop and the worst energy C^en_max, so any drop is worse than any delivery for every w > 0 whatever the options' energies, and the agent can never learn to drop tasks to save energy (checked numerically, 27 Oct). The earlier drop reward (−C_pen minus the spent energy) broke this whenever a delivery cost more than 2 · E_ref, which a cloud run easily does.

| Attempt | Reward |
| --- | --- |
| Delivered first, on time | r_ok − w · L/D − (1 − w) · C^en |
| Delivered first, soft and late (L ≤ D + L_max) | r_ok − w · (L/D + w3 · (L − D)/D) − (1 − w) · C^en |
| Refused, lost, cancelled, or its result came second | −(w · T_waste/D + (1 − w) · min(E_waste/E_ref, C^en_max)) |
| Last attempt of a dropped task (skipped, aborted, no result, no ack by the end, retries used up) | r_ok − w · C_drop − (1 − w) · C^en_max, with C_drop = 1 + (1 + w3) · L_max/D + C_pen = 5 |

- L is measured by the car; the attempt delivered first is the one in the car's confirm.
- T_waste = anchor time from sending the attempt to its refusal, loss detection or cancel. E_waste = the energy really spent (section 12.5); a drop's spent energy is logged, not used in its reward.
- Retries, extra hops and transfers need no extra term: their time and energy are already inside L and E.
- Why a drop is charged the worst energy, not what it spent: otherwise giving up early would look cheap (QECO punishes spent energy for the same reason; the cap C^en_max makes the rule hold for any energy). The energy really spent is still logged and reported. Wasted time counts twice, in the waste penalty and inside the delivered attempt's L; this is deliberate, it is what teaches the agent to avoid refusals.
- Herding is penalised through the refusal's waste penalty; a fixed extra refusal penalty is only an optional ablation.

**Latency L** = radio upload + forwarding + queue wait + run on the slot type used + result return, measured end to end by the car from task creation.

**The reward as equations.** Two unit-free costs are computed for each task, and the reward is their weighted sum subtracted from a bonus:

```latex
C^{time} = \begin{cases} \dfrac{L}{D} + w_3\, \dfrac{\max(0,\; L - D)}{D} & \text{delivered} \\[2mm] C_{drop} = 1 + (1 + w_3)\, \dfrac{L_{max}}{D} + C_{pen} = 5 & \text{dropped} \end{cases}
```

```latex
C^{en} = \begin{cases} \min\!\left( \dfrac{E}{E_{ref}},\; C^{en}_{max} \right) & \text{delivered} \\[2mm] C^{en}_{max} & \text{dropped} \end{cases}
```

```latex
r = r_{ok} - w \cdot C^{time} - (1 - w) \cdot C^{en}, \qquad r_{ok} = 1
```

**Property: a drop is never the cheap choice.** The worst delivery is a soft task delivered L_max = D late: C^time = 2 + w3 = 3, and C^en ≤ C^en_max. A drop has C^time = 5 and C^en = C^en_max. So a drop is worse than any delivery in both costs, for every weight w. Neither the optimiser nor the agent can learn to drop tasks to save energy.

**Why divide by D and by E_ref.** A 50 ms frame and a 6 s clip differ by a factor of 120 in time and 250 in energy. Dividing by the task's own deadline and reference energy puts every task on the same scale, so one weight w serves all task types.

**Source.** Charging a failed task for its energy follows QECO [6]. The two unit-free costs, the cap and the property above are PCNME's own.

## 12. Energy

> **Purpose.** Give every placement an energy. **Inputs.** Sizes S and S_res (bit); energy per bit of the links crossed (J/bit); the slot's mean dynamic power P̄ (W); the site's PUE; the run time (s). **Outputs.** The energy E_k of an attempt (J), its estimate Ê_k before placing, and the reference E_ref that makes it unit-free.

### 12.1 Whose energy

1. **Objective = the system energy one task causes:** the network that carries its data plus the node that runs it, as in Paper 1.
2. **Decisions use only dynamic energy** (busy minus idle power, times the time). Nodes and links are switched on anyway, so idle energy is the same whatever the agent picks.
3. **The results also report total energy** (dynamic + idle of every node over the run + messaging overhead), so the numbers compare with Paper 1.

| Left out of the reward | Why |
| --- | --- |
| The car's upload | Every option pays it (the car always uploads to its serving zone). Reported as P_tx · S ÷ R_car (airtime only) |
| The car's JPEG encoding | The same for every option; reported |
| The anchor's input copy (memory) | The same for every option. Testbed storage (a bind-mounted results folder) is test equipment |
| MQTT state, GPS, status | Paid per second, not per task. Reported as overhead |
| The agent's inference | Tiny and the same for every option |

### 12.2 One attempt

```latex
E_k = E^{fwd}_k + E^{exec}_{k,q} + E^{ret}_k
```

| Term | Formula |
| --- | --- |
| Forwarding | here: 0 · neighbour: e_bh · S · cloud: (e_bh + e_wan) · S (S in bits; waiting in a pipe costs no energy) |
| Execution | PUE_k · P̄_k,q · T^run, with P̄ = measured dynamic energy per run ÷ measured run time for that time-table row, and T^run the real busy time of the slot, including the T_abort tail of an abort |
| Return | (backhaul crossings · e_bh + WAN crossings · e_wan + e_down) · S_res. A fog executor: 2 backhaul crossings (executor → broker → serving zone). The cloud: 1 WAN + 1 backhaul. Each republish pays it again |

- **PUE** (total site power ÷ IT power) adds cooling and power losses: one value for fog cabinets, one for the cloud data centre.
- **The cloud's execution energy is counted.** Without it the cloud would look free and the agent would push work there.
- **No WAN energy penalty factor:** the WAN's own per-bit energy makes the cloud cost more. (This also removes the clash with Paper 1's α, which means something else.)
- **Energy per run, not data-sheet power:** with 2 slots on one GPU each stream draws part of its power; measuring at the real slot setup captures that. Real world: RAPL (CPU) and nvidia-smi (GPU). Fast mode and testbed: from the time table.

**Worked example: one T1 frame (S = 608 kbit, S_res = 24 kbit).**

| Option | Forwarding | Execution | Return | E_k | E_k ÷ E_ref |
| --- | --- | --- | --- | --- | --- |
| Own node, GPU slot | 0 | 1.5 × 0.914 J = 1.371 J | 0.0001 J | 1.37 J | 1.00 |
| Own node, CPU slot | 0 | 1.5 × 41.7 W × 69.5 ms = 4.35 J | 0.0001 J | 4.35 J | 3.17 |
| Neighbour, GPU slot | 0.0001 J | 1.371 J | 0.0001 J | 1.37 J | 1.00 |
| Cloud, GPU slot | 0.0002 J | 1.1 × 150 W × 9.1 ms = 1.50 J | 0.0001 J | 1.50 J | 1.10 |

Execution dominates: the network terms are about 0.01 % of the total. The conflict between latency and energy therefore does not come from distance. It comes from the slot type (a free CPU slot finishes sooner than a busy GPU queue, at three times the energy) and from the cloud.

**Source.** The structure follows the base paper [1], Eq. 1–9: the energy of a request is the transmission energy of its bytes to the executor, the processing energy, the share forwarded to the cloud, and the energy to send the response back. PCNME keeps the per-bit transmission terms and replaces the per-byte processing energy of [1] (a mix of idle and busy power, its Eq. 4 and 9) with the measured dynamic power of the slot times its run time; idle power is left out of the decision and reported separately (12.6). The per-bit energies, the cloud power and the PUE values come from Behbehani et al. [3] and the Uptime survey [14]. The fog slot energies are measured.

### 12.3 E_ref

```latex
E_{ref}(\text{type}) = PUE_{fog} \cdot \bar{P}_{ref} \cdot \bar{T}^{run}_{ref} + E^{ret}_{here}
```

The energy of running that task type locally on the reference fog class, on its fastest slot type. E/E_ref = 1 means "as costly as running it here on a standard fog node". It is fixed, so the reward's scale never drifts. C^en_max = the largest Ê_k ÷ E_ref over every task type and option, computed with the p95 run time from the time table and the link energies. It caps every energy ratio in the inputs, the reward and NSGA-II; being the worst case, it only ever clips rare retries.

### 12.4 What the agent sees before placing

```latex
\hat{E}_k = E^{fwd}_k + PUE_k \cdot \bar{P}_{k,q^*_k} \cdot \bar{T}^{run}_{k,q^*_k} + E^{ret}_k
```

The same estimate is used by NSGA-II and the experts.

### 12.5 Energy of each attempt in the reward

| Attempt | Energy counted |
| --- | --- |
| Delivered first | Its E_k with the real run time and bytes, every republish included |
| Refused | Forwarding only |
| Lost, cancelled or second | Forwarding + the execution really spent (reported by the executor) |
| Skipped in the queue | Forwarding only |
| Aborted | Forwarding + execution up to the abort + the T_abort tail |

A task's energy is the sum over its attempts, so retries are never free. In the reward the last attempt of a dropped task is charged C^en_max instead (11.4); what it really spent is reported.

### 12.6 Reported

Dynamic energy per task split into forwarding, execution and return; car upload energy; total system energy; the share of energy spent on wasted attempts.

## 13. The learning pipeline

> **Purpose.** Give the agent a good policy before its first real decision, then keep improving it. **Inputs.** Windows of tasks from each scenario; the costs C^time and C^en; the simulator. **Outputs.** A Pareto front of plans per window; its knee point; the weight w\*; demonstrations (s, a, r, s′, Δt); the pretrained network; in phase 2 a shared model.

### 13.1 Overview

Offline, in fast mode: NSGA-II plans windows of tasks → the knee plan and the whole front become demonstrations → DQfD-style pretraining of one global D3QN. Online, in every zone: the same network decides and keeps learning; FedAvg averages the zones in phase 2.

### 13.2 NSGA-II plans

1. **Window:** all pebbles arriving in the whole city during T_win of one scenario, with its seed (arrivals, run times, delays, mobility fixed).
2. **Chromosome:** one gene per task = the agent's action index (here, Nb1–Nb4, cloud) relative to the task's anchor. A gene pointing at an empty position is repaired to "here".
3. **Evaluation = one fast-mode run of the window** with the genes as decisions and the same seed for every plan (common random numbers). Draws are keyed by task, not by event order, so a task keeps the same luck in every plan wherever it runs: its run time is a uniform u keyed by (seed, task id, attempt), mapped through the inverse CDF of the time-table row it actually runs on; each delay is keyed by (seed, task id, attempt, hop). Arrivals and mobility do not depend on decisions and stay sequential per car (CHANGES C3). Queues, refusals, skips, aborts and retries happen as in a real run, so a node's load grows with what is put on it. Tasks arriving just after the window are placed by the greedy expert, so the last window tasks see realistic load. Before the window, T_warmup of greedy decisions fills the queues and pipes, so the first window tasks see realistic load too.
4. **Objectives**, both minimised and averaged per task:

```latex
f_1 = \frac{1}{N_{task}}\sum_{i} C^{time}_i, \qquad f_2 = \frac{1}{N_{task}}\sum_{i} C^{en}_i
```

C^time = L/D + w3 · max(0, L − D)/D for a delivered task; for a dropped task C^time = C_drop = 1 + (1 + w3) · L_max/D + C_pen, the worst delivery plus C_pen. C^en_i = min(E_i/E_ref,i, C^en_max) for a delivered task, with E_i over all its attempts; C^en_i = C^en_max for a dropped task. So a drop is worse than any delivery in both objectives, the front cannot end in a "make everything fail cheaply" plan, and the reward (11.4) uses the same per-task cost.

5. **No hard constraint:** under overload drops are unavoidable, and a feasibility constraint would collapse the front onto the fewest-drop plans. The drop cost keeps "any drop is worse than any delivery", as in the reward.
6. **Operators:** tournament selection, crossover, and a mutation that moves a task to another valid option (no arithmetic on node ids). Plain NSGA-II in pymoo. **Initial population** (CHANGES C9): the greedy plan, an energy-greedy plan (each window task's first decision takes the open option with the lowest Ê, 12.4, i.e. the cheapest that meets the deadline), and mutated copies of both (the same mutation operator) fill half of it; the rest are random valid plans. NSGA-II is elitist, so no point of the final front is dominated by the greedy plan. MMDE / DE mutation (used in Paper 1) is not used: it builds new genes by arithmetic, V = X₁ + F · (X₂ − X₃), which needs genes on a numeric scale, while PCNME's genes are categorical actions relative to each task's anchor, so a difference such as "Nb3 − Nb1" has no meaning.

**The problem in one statement.** For one window with N_task tasks, find the placements a = (a₁, …, a_N) that minimise both objectives at once:

```latex
\min_{a \in \{0,\dots,5\}^{N_{task}}} \; \big( f_1(a),\; f_2(a) \big), \qquad f_1 = \text{mean } C^{time}, \quad f_2 = \text{mean } C^{en}
```

No single plan is best in both, so the answer is a set: the Pareto front, the plans where one objective cannot improve without the other getting worse. The two objectives are the same two costs as in the reward (11.4), so the expert and the agent pursue one goal.

### 13.3 The knee point and the weight w

Knees as a concept: Branke et al. 2004, whose angle measure also uses the slopes to the two neighbours. The farthest-from-the-line rule is Das 1999, as used in KnEA (Zhang, Tian, Jin 2015, Eq. 4–5), which keeps only points on the convex side; KnEA applies it inside small neighbourhoods, PCNME takes the single knee of the final front. Normalise the final front: f̃_i = (f_i − f_i^min) ÷ Δf_i, with Δf_i = f_i^max − f_i^min. The knee is the front point farthest from the line through the two extremes, which on normalised axes is the point with the smallest f̃₁ + f̃₂. The local slope at the knee, from its two neighbours on the front sorted by f̃₁ (one-sided at an end), gives the weight on normalised axes:

```latex
\sigma = \frac{\tilde f_2(p^-) - \tilde f_2(p^+)}{\tilde f_1(p^+) - \tilde f_1(p^-)}, \qquad \tilde w = \frac{\sigma}{1+\sigma}
```

At that weight the knee is the local optimum of w̃ · f̃₁ + (1 − w̃) · f̃₂. The reward uses the raw per-task costs C^time and C^en, so the weight is taken back through the ranges:

```latex
w = \frac{\tilde w / \Delta f_1}{\tilde w / \Delta f_1 + (1-\tilde w)/\Delta f_2}
```

- **Deployment value:** w\* = the median of the knee weights over all training windows; the online reward and the agent's w input use it.
- **Every front point p** gets its own w_p by the same formulas, for training conditioned on w (the supervisor's request).
- **Limitation:** if the front is not convex around the knee, no weighted sum selects exactly that point; the reward then asks for the nearest trade-off while the margin loss still teaches the knee's actions.

**Reading the formulas.** σ is the exchange rate at the knee: how many units of normalised energy the front gives up for one unit of normalised latency. A weighted sum w̃·f̃₁ + (1 − w̃)·f̃₂ has its minimum where the front's slope equals w̃ ÷ (1 − w̃); setting that equal to σ gives w̃ = σ ÷ (1 + σ). The last formula undoes the normalisation, because the reward uses the costs themselves.

**What the runs gave.** w\* = 0.414 on the VFC grid (27 of 36 windows have a knee) and 0.344 on LuST (22 of 36). The window weights are widely spread, with quartiles near 0.0005 and 0.72 on the grid and about a third below 0.01, where the front is almost flat in energy. One median weight is a simplification and is listed as a limitation.

**Source.** NSGA-II: Deb et al. [7], run with pymoo [21]. The knee as the point farthest from the line through the extremes: Das [8], as used in KnEA [9]; the concept: Branke et al. [20]. Turning the slope at the knee into the reward weight is PCNME's own step.

### 13.4 From plans to demonstrations

1. Replay the knee plan in fast mode with the agent's input builder on. At each decision record s (48 inputs with w = the knee's weight), a = the plan's gene, then r, s' and Δt from the same replay with the online reward function.
2. The replay is the real simulator, so s holds exactly what the agent would see: stale state, pending sends, masks.
3. A refused planned action is re-decided by the greedy expert; a planned action that is masked at replay time is replaced by the greedy choice. Those transitions are stored without the expert flag (no margin loss).
4. Every front plan is replayed the same way with its own w_p.
5. The dataset covers many windows across S1–S6, seeds and zone on/off patterns. Boulders are not in it.

### 13.5 Pretraining and online learning

- **DQfD** pretraining (Hester et al. 2018) (decided 27 Oct, N19): loss = J_DQ + λ1 · J_n + λ2 · J_E + λ3 · L2. J_DQ = the 1-step double-DQN loss of 11.3 on all transitions. J_n = the same loss with an n-step target over the same anchor's next n_step = 10 decisions: Σ_{j\<n_step} γ^((t_j − t_0)/τ) · r_j + γ^((t_{n_step} − t_0)/τ) · Q_target(s_{n_step}, a\*), with a\* the best valid action in s_{n_step} by the online network; an n-step transition is built only once all its n_step rewards are final. J_E = the large-margin loss on flagged expert transitions only: max over valid a of [Q(s, a) + l(a_E, a)] − Q(s, a_E), with l(a_E, a) = m_E = 0.8 for a ≠ a_E and 0 for a_E; it pushes the expert's action above every other valid action. L2 = weight decay on the network. Start: λ1 = λ2 = 1, λ3 = 10⁻⁵. This removes the cold start; Tang & Wong and QECO need about 450–650 episodes from random play.
- **Online:** every zone keeps learning from its own closed attempts. The expert transitions stay in the replay buffer for good, in a part that is never overwritten. Prioritized replay (α_PER = 0.4, β_PER = 0.6, priority bonus ε_d = 1.0 on expert and ε_a = 0.001 on own transitions) decides how often they are drawn, so their share follows how much they still teach. The margin loss applies to them only. DQfD's own ablations show that dropping the n-step or the margin loss each hurts badly, and Maus et al. 2025 found that preloading demonstrations into a DQN's replay alone gave nothing. The earlier rule (expert share shrinks, margin fades, close to Zhang et al.'s Fading-BC) is kept as an ablation (16.4). Online learning fixes the distribution shift (states the expert never showed).
- IQL (offline RL) stays as an ablation.
- Limitation: the expert is clairvoyant. With common random numbers an NSGA-II plan sees the window's future arrivals, run times and delays, which the agent never sees, so some expert actions have reasons that are not in s. The TD and n-step losses and online learning correct those; the greedy-expert baseline and the expert filter (16.4) measure the effect (Maus et al. 2025).

**The loss as an equation.**

```latex
J = J_{DQ} + \lambda_1 J_n + \lambda_2 J_E + \lambda_3 J_{L2}, \qquad J_E = \max_{a \in \mathcal{A}(s)} \big[ Q(s, a) + m_E \cdot \mathbb{1}(a \ne a_E) \big] - Q(s, a_E)
```

| Term | What it does |
| --- | --- |
| J_DQ | The one-step loss on the target of 11.3: makes the values consistent with the rewards |
| J_n | The same with a 10-step return: carries a late reward back faster |
| J_E | The margin loss: pushes the expert's action a_E above every other action by at least m_E = 0.8. It is zero once the expert's action is already the best by that margin |
| J_L2 | Weight decay: keeps the network from fitting the small demonstration set too closely |

Without J_E the network would learn values only for the actions the expert took and could rate the others arbitrarily high. Without J_DQ it would copy the expert without learning what the actions are worth, and could not improve online. Source: Hester et al. [10].

### 13.6 Neighbour order in training

In fast mode every episode draws a random permutation of which neighbour sits in which position (next_zone follows it). The network cannot attach a meaning to "position 2", so averaged weights mean the same in every zone. Online, positions are fixed (section 7.5). A shared encoder for the neighbour slots would give the symmetry by construction, but would make PCNME's network differ from the baselines' plain 48-input network; it is an extension.

### 13.7 Federated learning (phase 2)

1. The cloud's FedAvg service starts a round on a timer: every T_fl (real world for example 1 hour; testbed a few minutes; fast mode T_fl of simulated time). It publishes fl/round = {g, global version, upload deadline}.
2. Each zone uploads its weights and its number of new experiences by HTTP (POST /fl/rounds/g), through the shared WAN pipe like any cloud transfer. A zone with no new experiences sends nothing.
3. The cloud averages the weights by share of new experiences and announces version g on fl/global; zones download it by HTTP (GET /fl/models/g).
4. Late or offline zones are left out of that round; the round never waits. A returning zone installs the newest model.
5. A zone installs a model between two decisions, never during one, and resets its target network. Replay buffers never leave the zone, so car data stays private. Every decision logs the model version it used.

| Variant (ablation) | Model that decides | Model uploaded |
| --- | --- | --- |
| No federated learning | The zone's own model, never averaged | Nothing |
| FedAvg | The shared model: the global model trained online since the last round | The shared model |
| FedAvg + local fine-tuning | A personal copy: the new global model fine-tuned for K steps on the zone's replay buffer, then trained online | The shared model, never the personal copy |

In the third variant every transition trains both copies; the personal copy takes over only after its K steps, so decisions never wait. Fine-tuning helps because a quiet suburb and a busy downtown see different traffic, and one average fits neither perfectly.

**Where it is evaluated (16 Nov, QUESTIONS Q64).** Phase 2 is evaluated in fast mode only. On the testbed each learning zone's network lives in its trainer process (15.5), so FedAvg's upload, install and fine-tuning would run there; that is future work.

**The averaging rule.** With θ_z the weights of zone z after local training and n_z its number of new samples in the round:

```latex
\theta^{(g+1)} = \frac{\sum_z n_z\, \theta_z}{\sum_z n_z}
```

## 14. Workloads and tasks

> **Purpose.** Define the tasks and how long they take. **Inputs.** Real camera frames (BDD100K) and real models (YOLOv8) on real hardware (GTX 1070). **Outputs.** The task types with their class, deadline, size and share; and the time table: mean run time, 5 % and 95 % quantiles and energy per run for each (task type, hardware class, slot type).

### 14.1 Task types

| Type | Model | Class and deadline | Usual sort | Role |
| --- | --- | --- | --- | --- |
| T1 Real-time detection, one frame | YOLO (for example v8 or 11), nano or small | Hard, 33–100 ms | Pebble | First scenario |
| T2 Clip analytics, a bundle of frames | The same YOLO, medium or large, as one batch | Soft, seconds | Boulder or a fog GPU | Bundling at the car |
| T3 Lane or drivable-area segmentation | A light segmentation network | Hard, about 100 ms | Medium | A task of its own |
| T4 Plate or sign reading | Detector + text recognition: two steps inside one task | Soft | Medium | A task with internal steps |
| T5 3D detection from LiDAR | PointPillars-type network | Hard, about 100 ms | Heavy | Optional stress test |
| T6 Route planning | Graph search on the road map | — | — | Not offloading: an ordinary cloud service; out of scope |

One detector in several sizes gives a natural light-to-heavy mix with real costs, far easier to defend than invented MI values. The task mix per scenario is a number for the numbers step.

### 14.2 A task's internal steps

A task may be a small DAG, for example one frame: step 1 decode (CPU) → step 2 detect objects (GPU) and step 3 detect lanes (CPU) → step 4 track and fuse → result (about 3 KB). The DAG only describes the work inside the task: the whole task is placed once, holds one slot, and its measured run time already contains all its steps. Steps are never split across nodes: dependencies, partial failures and intermediate data (often bigger than the input) cost more than they gain. The longest step sets T_abort.

### 14.3 Inputs and bundles

- **Each frame is compressed on the car as JPEG,** on its own. Video coding is rejected: it makes frames depend on each other, so one task could not be decoded alone. Input size = the compressed size; step 1 decodes it on the executor. The encode time is inside L (the car's clock starts at creation).
- **Raw video is** far too heavy: one uncompressed 720p frame is about 2.8 MB (1080p about 6.2 MB). As JPEG (quality 80) a BDD100K 720p frame is 76 KB on average (5–95 %: 34–127 KB, measured on 300 frames in the calibration run), and even then 30 frames per second is about 18 Mbps per car (8–30 Mbps). Compression makes every pipe carry more tasks per second, so the bandwidths are chosen with the compressed sizes.
- **Bundles at the car** (for example 60 frames = 2 s of video, 6–18 MB as JPEG) fit soft analytics only: the first frame already waited 2 s. For real-time work the usual trick is detecting every Nth frame and tracking in between.

### 14.4 The time table

One row per (task type, hardware class, slot type): mean, p5 and p95 of the run time, the longest step, and the mean dynamic energy per run. Measured by running real YOLO many times (or taken from published benchmarks), with the compressed input sizes beside them. In the main experiments a slot holds a task for a time drawn from this measured distribution (fixed seed); no real YOLO runs. The draw is lognormal, fitted to the row's mean and p95, which both stay exact; its p5 is about 0.75 × the mean for the placeholder rows, and for measured rows the calibration run reports the fitted p5 next to the measured one (CHANGES C5). The quantiles give T^run,lo and T^run,hi.

# Part II — The evaluation environment

## 15. Fast mode and the Docker testbed

> **Purpose.** Evaluate the same code in two ways. **Inputs.** A configuration, a scenario, a method and a random seed. **Outputs.** In fast mode (a discrete-event simulation), a bit-identical event log per seed in seconds of computing: used for all training and for the 960 evaluation runs per map. On the Docker testbed (real MQTT and HTTP over shaped networks), the same metrics in real time: used to measure the gap between simulation and a running system.

### 15.1 Real world vs evaluation

| Part | Real world (Part I) | Evaluation |
| --- | --- | --- |
| Zones | N zones, up to 4 neighbours each | 5 zone positions: 4 active, the 5th joins in the topology experiment |
| Cars | Thousands | 10 car containers, each simulating 20–100 virtual cars (200–1 000 cars) |
| Broker | EMQX cluster at the metro edge | One EMQX container standing in for the cluster |
| WAN pipe | The metro-edge router's uplink | A wan-gw container shaped on its cloud side |
| Radio | 5G or RSU | The VFC radio model (Shannon rate) in a software pipe per zone |
| Execution | Real inference | Slots wait for a time drawn from the measured time table |
| Clocks | Each machine its own | Shared host clock (testbed) or one simulated clock (fast mode); the design relies on neither |

### 15.2 One code base, two runtimes

The zone service, the car app and the cloud service are written once in Python (the agent is PyTorch, NSGA-II is pymoo, the VFC code is Python) against four interfaces:

| Interface | Fast mode | Testbed |
| --- | --- | --- |
| Bus | In-process topic table; each delivery pays the modelled broker delay | MQTT client to EMQX |
| Calls | Direct call on the target object after the modelled transfer time | HTTP between containers |
| Clock and timers | Discrete-event simulated clock (one event queue, no waiting) | Real clock |
| Links | The link-model formulas in code | tc netem delays and rate limits; the radio pipe in software |

Every random draw comes from seeded generators, one per purpose (arrivals, run times, link delays, mobility), so the same seed gives the same run. Every drawn time (run times, link delays, arrival gaps) is rounded to whole nanoseconds, and every radio rate to whole bit/s, where it is made: log, exp, sin and cos may differ in the last bit between operating systems, and one bit can flip a scheduling decision, so without it the same seed could give different runs on Windows and Linux. No quantity in the design is near 1 ns or 1 bit/s. Fast mode is used for NSGA-II, pretraining, baseline training and sweeps; the testbed for the final evaluation of every method. Training on the testbed alone is impossible: about 240 real-time runs (6 scenarios × 8 methods × 5 seeds) plus training and thousands of NSGA-II evaluations would take weeks.

### 15.3 What comes from the VFC code

Taken: the radio model (path loss + Shannon rate), the city grid and its mobility with random turns (extended with traffic lights and random stops), sensible parameter values, and its PPO agent as a baseline algorithm. Not taken: its environment loop, its state and actions, its delay equations (three errors found: follow its code, not its equations), its device-side decision, its parked service vehicles. Check its license before copying code; otherwise reimplement from the paper and cite it.

**The radio uplink:** one pipe per zone whose airtime the cars uploading at once share equally: while n uploads are in progress each advances at that car's current Shannon rate R_car ÷ n (CHANGES C6). On the testbed tc cannot give every virtual car its own rate on one interface, so the pipe runs in software at the zone's intake with the fast-mode code (the car app sends R_car with the upload), and netem adds only the radio's base delay. Coverage is enforced in the car app: a car talks only to the zone covering its simulated position, and to nobody in a gap.

### 15.4 Testbed containers and networks

| Container | Count | Contents |
| --- | --- | --- |
| zone-1 … zone-5 | 4 active + 1 that joins | Zone service: MQTT client, intake + radio pipe + TOF-Broker, car registry + predictor, state store, agent, online trainer, executor with GPU and CPU queues and slots, FedAvg client, logger |
| car-1 … car-10 | 10 | Each simulates 20–100 cars: movement on the city map with traffic lights and stops, zone choice with hysteresis, GPS every 100 ms, task creation, JPEG sizes, radio rate, duplicate filter, confirms |
| broker | 1 | EMQX |
| wan-gw | 1 | The metro-edge router: all zone → cloud traffic routes through it; shaped on its cloud side (one shared WAN pipe). Fallback if routing fails: a reverse proxy (worse: it splits TCP) |
| cloud | 1 | Cloud executor (many slots), FedAvg service, model registry |
| storage | 0 | None: datasets, models and results are a bind-mounted folder, data/ (test equipment; QUESTIONS Q55) |
| metrics | 1 | InfluxDB + Grafana (test equipment, never on the decision path) |
| runner | 1 | Starts each scenario, sets delays and seeds, collects results, tears down |

| Network | Connects | Emulated |
| --- | --- | --- |
| radio_net | cars ↔ zones | netem base delay; the Shannon-rate pipe in software at each zone |
| backhaul_net | zones ↔ zones ↔ broker ↔ wan-gw | netem 1–5 ms, rate limit per zone's outgoing interface |
| wan_net | wan-gw ↔ cloud | netem 25–100 ms each way with jitter, the WAN rate limit; carries tasks, results, the cloud's MQTT and FedAvg weights |
| mgmt_net | every container ↔ storage, metrics | Unshaped: test-equipment traffic never uses the shaped pipes |

To verify when coding: the routing through wan-gw, netem's jitter distribution and that jitter does not reorder packets in a way that distorts transfer times, and that each container allows net.ipv4.tcp_slow_start_after_idle=0 (connections are kept open and warm).

### 15.5 Compute

A fog node's size comes from its slot counts and its time table. --cpus only limits the zone's own work (agent, queues, MQTT). Optionally one zone runs real YOLO in one scenario to validate the table and T_abort.

A zone whose agent learns online (13.5) gets a second CPU on the testbed, for its **trainer**: a separate process that holds the replay buffer and makes the updates. The zone turns its decisions and transitions into samples as in fast mode and sends them to the trainer; the trainer sends its newest weights back (testbed.trainer_sync_s) and the zone installs them only between two decisions. Learning in the zone's own loop would delay its decisions, messages and timers by the length of each update (QUESTIONS Q62). Fast mode has no compute time and learns in place.

### 15.6 The pipeline

1. **Calibrate:** measure YOLO (nano to large) per hardware class and slot type, and the JPEG sizes → the time table.
2. **Plan:** NSGA-II in fast mode over many windows → fronts, knees, w\* → demonstrations in storage.
3. **Pretrain:** DQfD-style on the demonstrations → models/global_v0.
4. **Baselines:** train VFC PPO, Tang & Wong, QECO, D3QN from scratch and D3QN + greedy expert in fast mode with the same budget.
5. **Testbed runs:** zones download global_v0 (deployment, not FedAvg yet), cars drive, agents decide and keep learning; in phase 2, FedAvg rounds.
6. **Results:** every task logged (CSV/Parquet) to storage, live metrics to InfluxDB; notebooks compare methods and fast mode against the testbed on the same scenarios and seeds.

### 15.7 Topology experiments

- **Training:** 5 positions, zones switched on and off at random per episode, neighbour order shuffled.
- **Zone join:** zone 5 is switched on during a testbed run: it downloads the global model, publishes online and its state, takes free neighbour positions; the runner adds its coverage circle. Measure how load spreads and how fast agents use it (expect a short learning dip).
- **Zone kill:** 2 or 3 of the 5 zones stop. Last wills mask them; tasks other anchors placed there are retried; tasks they anchored are lost; their cars attach to an overlapping zone or are in a gap. Plot on-time rate, latency and drops before, during and after.
- **Broker restart:** measures recovery.

Later and optional: a live map dashboard (cars, zone circles coloured by load) for the defence, not a source of results. Future environments: one stronger machine with zones pinned to cores and a shared GPU (Docker, not VMs); a real hardware testbed with edge boards.

## 16. Experiments

> **Purpose.** Say what is compared, under which load, and how it is measured. **Inputs.** 8 methods, 6 scenarios, 2 maps, 20 random seeds. **Outputs.** On-time share, latency, energy per task, refusal rate and decisions until stable, each as a mean with a 95 % interval and paired tests. The numbers are in Appendix B.

### 16.1 Methods compared

Every learning method runs its own algorithm inside PCNME's zone agent, with the same 48 inputs, 6 actions, masks, reward and time-based discount, and the same training budget in decisions. These are algorithm-level reimplementations in PCNME's setting, and the thesis says so: the original papers decide on the device.

| Method | What it is here | Shows |
| --- | --- | --- |
| PCNME | D3QN, DQfD-style pretraining on the knee expert, then online | The proposal |
| D3QN from scratch | The same network, no pretraining | What pretraining adds |
| D3QN + greedy expert | The same pretraining, greedy expert as teacher | That the knee is a better teacher (the novelty claim) |
| VFC PPO | PPO actor-critic; masked actions get probability 0 | Open-source vehicular baseline; delay only, so no energy claim against it |
| Tang & Wong | DQN with an LSTM over the last H inputs, dueling + double | A well-known offloading DQN |
| QECO | Its D3QN + LSTM design, QoE- and energy-aware, on our inputs and reward | The main opponent on energy |
| Greedy | Earliest estimated finish among unmasked options; ties by energy | A strong rule; also the greedy expert |
| Random | Uniform among unmasked options | The floor |
| NSGA-II knee plan (reference) | The offline knee plan replayed on the same windows and seeds, fast mode only. Clairvoyant, so a reference line, not a competitor | How close the online agent gets to a planner that knows the window's future |

Fairness: the same scenarios, seeds and online budget on the testbed for every method; learning baselines get the same fast-mode training budget as PCNME's online phase (they have no expert, as in their papers).

### 16.2 Scenarios

Designed by load, not by number of cars. Two exact definitions: ρ_fog = the city-mean arrival rate per zone × the mix's mean GPU run on F-GPU ÷ c_GPU (the load if every task of the whole mix, boulders included, ran on its anchor's GPU; so S2 and S4–S6 have the same arrival rate and differ only in deadlines, mobility or fog hardware, and in S4–S6 the real fog load is lower because T2 goes to the cloud), and TI_WAN = the cloud-eligible bits per second (soft tasks and boulders) ÷ B_WAN (the WAN load if all of them went to the cloud). One B_WAN, set so that TI_WAN = 0.9 at ρ_fog = 0.9, gives TI_WAN = ρ_fog in S1–S3, because both scale with the arrival rate. The arrival rate per car is divided by the share of task creations made with coverage, measured once per map by a short positions-only calibration run (tag C, QUESTIONS Q29), so that the creations that reach a zone meet ρ_fog; a frame made in a coverage gap is reported as uncovered, apart from the method's drops. The hotspot (40 % of cars through z1) puts z1 at 1.6 × the mean load in every scenario: a share h_hot of the cars stays on the roads between z1's intersections and the rest roam the whole map, with h_hot set by the same calibration run so that z1 holds 40 % of the covered load (tag C, QUESTIONS Q30). With these deadlines a hard task cannot reach the cloud in time (T1: about 140 ms mean round trip against D = 100 ms), so the cloud serves soft tasks and boulders only, which is the reason fog exists.

| Scenario | Traffic intensity | Deadlines | Mobility | Shows |
| --- | --- | --- | --- | --- |
| S1 Light | about 0.5 | Loose | Low | Sanity: every method works |
| S2 Critical | about 0.9 | Loose | Low | Queues start to matter |
| S3 Overload | 1.5 to 2 | Loose | Low | The system is overloaded: under Greedy the fog GPUs saturate (97–100 %); under cloud-only placement the WAN queue grows without limit (M2 test) |
| S4 Tight deadlines | about 0.9 | Tight | Low | Feasibility separates methods |
| S5 High mobility | about 0.9 | Tight | High | The mobility component earns its place |
| S6 Mixed fog | about 0.9 | Loose | Low | Placement intelligence on unequal nodes (with and without GPUs) |

City maps (decided 27 Oct, N19): the VFC grid, simple and complex (traffic lights, random stops), for development and the S1–S3 checks; a LuST crop (SUMO, 1.2 × 1.2 km of central Luxembourg, with the same zone layout as the grid, 18) for the final results. SUMO runs once per scenario; its trajectories (FCD output, 100 ms) are replayed in fast mode and by the car containers, with no live coupling, so both runtimes see the same cars. A peak window for S3 and S5, off-peak for S1. Fallback: TAPASCologne or Bologna. Real routes also make the LSTM predictor option possible (10.2). Run S1–S3 first: if the WAN pipe does not congest in S3, stop and debug the queueing.

**The two load numbers as equations.** With λ_zone the task arrival rate of one zone, T̄^run_mix the mean GPU run time of the task mix and c_GPU the number of GPU slots:

```latex
\rho_{fog} = \frac{\lambda_{zone} \cdot \bar{T}^{run}_{mix}}{c_{GPU}}, \qquad TI_{WAN} = \frac{\text{bits sent to the cloud per second}}{B_{WAN}}
```

Below 1 the queue keeps up; at or above 1 it grows without limit. With c_GPU = 2 and the calibrated mix, ρ_fog = 0.9 means 38.9 tasks per second per zone.

### 16.3 Metrics and statistics

Shares delivered on time, late, dropped and failed; mean, p99 and worst-1 % latency; dynamic and total energy; refusal rate; wasted-attempt energy; delivery success after a zone change; next-zone acc@1 and T_exit error; decisions until the on-time rate is stable; for NSGA-II the hypervolume of the fronts; load balance as Jain's fairness index over zone loads, with a per-zone load chart; the agent's mean (latency, energy) point on the NSGA-II front of the same windows and its distance to it; the demonstration up-sample ratio (how often expert transitions are drawn compared with uniform sampling, as in DQfD's Fig. 1); early-stage performance = the mean over the first N_early online decisions (N_early = 1 000) after deployment (Zhang et al.). Evaluation seeds are never used for planning that any method learns from, for demonstrations or for training (Maus et al.); the knee reference (16.1) plans on its evaluation seeds because it is a reference line that no method learns from, and runs only for S2 and S4 on the first 5 evaluation seeds (CHANGES C10). Testbed: 5 seeds per scenario and method, mean ± 95 % confidence interval (Student t with 4 degrees of freedom, t = 2.78). Fast mode: 20 seeds (t = 2.09 with 19 degrees of freedom; N19). Fast mode and testbed are both reported on the same scenarios and seeds (the simulation-to-testbed gap).

**Two formulas.** Load balance is Jain's index over the zone loads x_z; every reported mean has a Student t interval over the n = 20 seeds with standard deviation s:

```latex
J = \frac{\left(\sum_z x_z\right)^2}{N_z \sum_z x_z^2}, \qquad \bar{x} \pm t_{0.975,\, n-1} \cdot \frac{s}{\sqrt{n}}
```

### 16.4 Ablations and extra experiments

| Experiment | Question |
| --- | --- |
| State period 10 / 100 / 500 ms, R3 on and off | How stale state and herding hurt; is event-driven state needed |
| Queue order: FIFO, classes + FIFO, full rule | Does the queue rule matter |
| α_TOF sweep {0.5, 0.65, 0.8, 1.0} | The TOF threshold |
| γ = 0 vs γ > 0 | Does looking ahead help, or is placement a bandit problem |
| No FL / FedAvg / FedAvg + fine-tuning | Federated learning (phase 2) |
| IQL instead of DQfD | Offline RL vs demonstrations |
| Fixed refusal penalty (only if the refusal rate stays high) | Herding |
| 60 inputs (both slot types per option) | Is one assumed slot type enough |
| Zone join, zone kill, broker restart | Robustness |
| Pretrain only (no online learning) | Is imitation alone enough (Zhang et al.'s Only-BC; Yu et al.'s offline imitation) |
| Expert data kept with priorities (default) / shrinking share + fading margin / cleared after pretraining | How the expert data should be handled online (Hester et al.; Zhang et al.'s Fading-BC and Norm-BC) |
| No n-step loss; no margin loss | Which DQfD part matters here (Hester et al.'s own ablation) |
| Expert filter: keep a window only if its knee plan beats greedy by ≥ x % | Does noisy demonstration data hurt (Maus et al.) |
| No neighbours (here + cloud only); no predictor (exit = ∞, next zone = stays); no result forwarding (result only to the anchor); no TOF-Broker (α_TOF = ∞) | What each mechanism buys (Fan et al.'s ablation-style baselines) |
| w sweep: the w-conditioned agent run at the front's w_p values | Does the agent follow the trade-off it is given (Wu et al.'s weight sweep) |
| Sensitivity sweeps, fast mode: one parameter at a time from 50 % to 150 % of its default (arrival rate, deadline D, input size, number of cars, fog capacity, WAN bandwidth); on-time and drop shares, latency and energy per point | How each method reacts to each condition: the standard figure in Fan, Tang & Wong, QECO and Wang |
| Random-plan expert (a random valid NSGA-II individual as teacher) | How much the teacher's quality matters, next to greedy and knee (Lin et al., Fig. 4) |

### 16.5 Verification before experiments

In the repository under tests/, all passing before the big runs: a units table (every symbol with its unit; every equation balances); a hand-calculation test (one task, empty system, fixed numbers, simulator within 1 %, once per destination); the queueing test (M/M/1 and M/M/c); sanity asserts (no negative delays, total delay = sum of its parts, energy ≥ 0, shares between 0 and 1); a front check: in S2 the first NSGA-II fronts must hold at least 5 distinct points. If they do not, the latency–energy conflict is too weak and the energy model is revisited before any training. The risk is real: network energy is about 0.1 % of E_ref, so here and a neighbour cost the same energy, and the conflict comes only from GPU vs CPU slots and from the cloud (2.9 × E_ref for T1 with the placeholders).

# Part III — Limits and numbers

## 17. Assumptions and out of scope

One sentence each in the thesis.

**Assumptions**

- One fog server per access point.
- The broker cluster is always available; a full broker outage is out of scope (mobility support and state sharing would stop, with no fallback).
- The cloud is always available and never fills under PCNME's load.
- A dead anchor loses the tasks it owns (the car may still get a result, but no reward is recorded).
- Connections between zones and to the cloud are persistent and warm (no TCP slow start).
- Bandwidths and mean link delays are known configuration; only pipe waits are estimated.
- The radio uplink shares its airtime equally among the cars uploading at once, each at its own Shannon rate (fluid processor sharing; real 5G NR and Wi-Fi 6 schedulers share airtime among users within a millisecond). The rates come from free-space path loss (the VFC model), about 540–930 Mbps at 10–300 m, so the radio is not the bottleneck (radio TI about 0.05 in S2), and a 76 KB frame is never held behind a 4.6 MB T2 clip (CHANGES C6); every option pays the same upload.
- Execution times follow the measured table; a slot is one inference stream or one core group.
- Every fog node has every task type's model loaded; only the task's input moves, never a container or a model (proactive service migration, as in MAPSM, moves hundreds of MB and is out of scope).
- Security and privacy of messages and data are not covered.

**Out of scope (future work)**

- Moving a half-run task between zones (live migration).
- Splitting one task's steps across nodes.
- The cloud super task (merging tasks) and GPU micro-batching at the fog.
- Pausing and resuming a running task (abort is in scope).
- Fog node failures beyond retries and last-will masking.
- Cost as a third objective.
- Learned trajectory prediction beyond the extension option.
- Parked cars as extra compute (the VFC paper's service vehicles).
- A full real-world deployment: the zone service is a design and a prototype.
- Route planning (an ordinary cloud service, not offloading).

**Simplifications stated in the paper**

- The executor chooses the slot type by time only; the energy trade-off is made between nodes by the agent.
- If the front is not convex around the knee, the reward's weighted sum cannot select exactly that point.
- Two anchors for one task are possible after a lost upload reply (rare, logged, harmless).
- A result counted as a drop that the car actually received (every ack lost) is possible but very rare; it is counted from the car's log.
- The NSGA-II expert is clairvoyant within its window (13.5).

## 18. Numbers

Every number now has a default; together they form config/default.yaml. Tag: **P** = from a paper, **R** = chosen by the reason given, **C** = calibrated by the rule given (computed when the config loads, or in fast mode), **M** = measured in the calibration run (the value shown is a placeholder until then). The arithmetic that checks these values fit together is in calc_defaults.py; its results are quoted in the last column.

| Group | Number | Default | Tag | Source or reason |
| --- | --- | --- | --- | --- |
| Layout | Area; zone centres; radius ℓ_z | 1.2 × 1.2 km; z1–z4 at (300, 300), (900, 300), (300, 900), (900, 900) m, z5 (joins) at (600, 600); ℓ_z = 350 m | P / R | Yin et al. 350 m. Adjacent centres are 600 m apart, so circles overlap by 100 m for handover. The centre is a gap until z5 joins (the join experiment fills it) |
| Layout | ℓ_nb; handover hysteresis | 900 m; 20 m | R | 900 m makes diagonal zones (849 m) neighbours, so every zone has ≤ 4. The 20 m hysteresis stops ping-pong in the 100 m overlap |
| Cars | Count; speeds | 200–1 000; low U{10, 20, 25, 40} km/h, high U{30, 40, 50, 60} km/h | P / R | VFC speeds for low mobility; high = urban arterial |
| Cars | Hotspot | 40 % of cars routed through z1 (1.6 × the mean load): a share h_hot of the cars stays on z1's roads, h_hot = 0.200 (low mobility) and 0.175 (high) on the VFC grid | R / C | Like Fan et al.'s high-load RSUs: without uneven load, neighbour offloading (C4) is never needed. h_hot from tools/calibrate_map.py so that z1 holds 40 % of the covered load (QUESTIONS Q30) |
| Radio | Bandwidth, power, gains, noise, carrier | 40 MHz, 1 W, 5 + 5 dBi, −174 dBm/Hz, 5.9 GHz; free-space path loss + Shannon | P | VFC. The carrier is not in the VFC paper: check its code. Gives 540–930 Mbps at 10–300 m |
| Radio | Uplink and downlink delay | U(1, 5) ms each | R | 5G NR user-plane latency |
| Links | Backhaul bandwidth and delay | 1 Gbps per zone outgoing pipe; U(1, 5) ms | P | Section 4.2 |
| Links | WAN delay; WAN bandwidth | U(25, 100) ms one way; B_WAN = the cloud-eligible bits per second at S2 ÷ 0.9 (about 36 Mbps with the calibrated time table and sizes) | P / C | VFC (50–200 ms round trip). One B_WAN serves S1–S3: arrivals scale with ρ_fog, so TI_WAN = 0.5 / 0.9 / 1.5 follows (16.2) |
| Links | Broker processing | 1 ms on top of the backhaul delay | R | EMQX in-memory routing |
| Workload | Task mix (by count) | T1 74.2 %, T3 15.2 %, T4 10.2 %, T2 0.4 %; T5 off | R | Real-time frames dominate; T2 is rare but heavy: about 1/3 of the fog GPU time (33.5 %) and 19 % of the bits. T1 : T3 : T4 keep the ratio 73 : 15 : 10 |
| Workload | Input sizes; result sizes | 76 KB per JPEG frame (T1, T3, T4), T2 = 60 frames = 4.57 MB; results 3 KB, T2 10 KB | M | JPEG quality 80 on BDD100K 720p frames: mean 76 KB (5–95 %: 34–127 KB) over 300 frames (14.3) |
| Workload | Deadlines D (loose / tight) | T1 100 / 50 ms; T3 100 / 70 ms; T4 1 / 0.5 s; T2 6 s / 1.1 × T̄^run_strong(T2) | R / C | Hard vision tasks inside 33–100 ms (14.1). T2 is a pebble with the loose D (3.87 < 0.8 × 6 s) and a boulder with the tight D (3.87 ≥ 0.8 × 4.26 s), so the TOF-Broker is exercised in S4 and S5 |
| Workload | Arrival rate per car | λ_car = ρ_fog × c_GPU ÷ (mean GPU run of the mix × cars per zone × coverage): S2 = 38.9 tasks/s per zone, 0.78 per car at 200 cars, 0.16 at 1 000, before the coverage; coverage = the share of task creations with coverage, 0.863 (low mobility) and 0.849 (high) on the VFC grid, at the calibrated hotspot share | C | Sets the scenario load (16.2). Coverage from tools/calibrate_map.py (positions only, 120 s, the smallest fleet); frames made in a gap are reported as uncovered (QUESTIONS Q29) |
| Time table | Mean run, p5, p95, longest step, energy per run | Measured on the GTX 1070 (FP32, 2 streams and 2 CPU slots busy at once; run = decode + pre + inference + post). F-GPU GPU slot, mean (p95): T1 22.6 (25.7) ms, T3 26.5 (29.9) ms, T4 97.8 (142.7) ms, T2 3.87 (3.91) s; energy per run 0.91 / 1.12 / 3.05 / 225.5 J. CPU slot: T1 69.5, T3 93.3, T4 382 ms, T2 none. Cloud GPU: Ultralytics' A100 TensorRT inference (v8n 0.99, v8n-seg 1.21, v8l 2.39 ms) + the measured decode, pre- and post-processing; cloud CPU: Ultralytics' CPU ONNX inference (v8n 80.4, v8n-seg 96.1 ms) + the same measured steps; T4 in the cloud: that YOLO part + the recogniser as measured on the GTX 1070 (no published number, no speed-up assumed). Placeholder factors (p5 0.75 ×, p95 1.3 ×, longest step 0.5 × the mean) fill only what a row does not give; run times lognormal, fitted to the mean and p95 | M | tools/calibrate_yolo.py, data/calibration/. T4 = yolov8n + EasyOCR's CRNN recogniser on up to 3 plate-like crops. GPU slot at PCIe x4 |
| Nodes | Classes and slots | F-GPU (reference and strongest): c_GPU = 2, c_CPU = 2 (4 cores each). F-CPU (S6 only): c_GPU = 0, c_CPU = 2. Cloud: 32 + 32 | R | One GTX 1070 (FP32, no TensorRT) with 2 streams. S6 alternates F-GPU and F-CPU zones |
| Nodes | n^max_q; n_res; n_head | n^max_GPU = 32, n^max_CPU = 16; n_res = 4; n_head = 2 | R | 32 queued T1 = 128 ms of work, more than D, so "late" binds before "full" for hard tasks. Soft tasks are admitted below 28 (GPU), and masked at 26 |
| Scheduler | T_age; T_probe; T_keep | 200 ms; 50 ms (fog), 300 ms (cloud); 10 s | R | T_age = 2 × the loose hard D. T_probe covers about 10 backhaul round trips, or the WAN round-trip maximum plus margin. T_keep ≥ longest D + L_max = 6 s |
| Timing | κ_up; m_wd; quantiles; η; T_ack; T_leave; GPS expiry | 2; 20 ms (fog), 50 ms (cloud); 5 % and 95 %; 0.2; 200 ms; 300 ms; 1 s | R | T_ack ≫ result + radio + confirm (about 20 ms) |
| Timing | Encode time (the car's JPEG encode, 14.3) | 0 s (placeholder) | M | `timing.encode`. Inside L: the upload starts at creation + encode, and T^bud_car = D − time since creation already holds it, so it is not subtracted again (QUESTIONS Q14) |
| State | State period; staleness | 100 ms; 300 ms | P | Section 9.2 |
| Reward | r_ok, C_pen, w3, C_drop; w; G; C^en_max | 1, 2, 1, 5; median knee weight; G = 500 ms; C^en_max computed from the time table | P / C / R | G > the slowest status report (cloud one way 100 ms + broker) + T_abort. C^en_max ≈ 9.2 with the calibrated table (T4 on a CPU slot at p95) |
| RL | γ per τ; τ | 0.9 per 100 ms | P | Tang & Wong and QECO (0.9 per 0.1 s slot). At S2 an anchor decides every 13.5 ms, so the per-step factor is 0.986 and the horizon about 70 decisions |
| RL | Network; optimiser; learning rate; batch | 48 → 256 → 256 → dueling heads (V: 1, A: 6); Adam; 1e−4; 64 | P | GA-DRL (2 × 256, lr sweep); Ruan et al. (batch 64) |
| RL | Replay; target update; update ratio; ε | 100 000 own transitions + the expert part, never overwritten; hard copy every 1 000 updates; 1 update per 4 new transitions; ε = 0.01 online, baselines 1 → 0.01 over the first 20 % | P / R | Ruan et al. (buffer); DQfD (ε 0.01); Tang & Wong (ε schedule) |
| RL | Loss; LSTM history H; PPO | MSE (Huber if unstable, 11.3); H = 10; clip 0.2, lr 3e−4, GAE 0.95, 10 epochs, minibatch 64 | P / R | H only for the Tang & Wong and QECO baselines; PPO = Schulman et al. defaults |
| Retries | Hard; soft | 1; 2 | P | Section 9.1 |
| TOF-Broker | α_TOF; strongest class | 0.8, then the sweep; F-GPU | P | Section 7.1 |
| Energy | Dynamic power per time-table row | P̄ = measured energy per run ÷ mean run (12.2): F-GPU GPU rows 40.5 (T1), 42.4 (T3), 31.2 (T4), 58.3 (T2) W. Rows without a measurement take the slot's power: F-GPU 41 W per GPU slot, 41.7 W per CPU slot; cloud 150 W per GPU slot, 30 W per CPU slot | M / P | GPU: NVML in the calibration run. CPU slot: estimated, (TDP 140 W − idle 15 W) × 4 ÷ 12 threads (no RAPL on Windows; data/calibration/cpu-power.md). Cloud: Behbehani et al. |
| Energy | Idle power (reporting only) | Fog node 83.4 W; cloud server 57 W | P | iFogSim default (Jafari & Rezvani); Behbehani et al. |
| Energy | PUE_fog; PUE_cloud | 1.5; 1.1 | R / P | Fog: between Behbehani's 1.0 for an open edge site and the Uptime 2023 average. Cloud: Behbehani |
| Energy | e_bh; e_wan; e_down | 0.141, 0.228 and 4.2 nJ/bit, the network ones × PUE 1.5 | P | Behbehani Table 7 (aggregation switch + router port; core + cloud router ports + cloud switch); e_down = AP 28 dBm ÷ 150 Mbps, Behbehani's method |
| Energy | Car P_tx; reference class | 0.158 W (reporting only); F-GPU | P | Behbehani (DSRC 22 dBm) |
| NSGA-II | Population; generations; crossover; mutation | 100; 100; 0.9 (uniform on genes); 1 ÷ number of genes | P | Mokni & Yassa (100 × 100); Deb et al. 2002 |
| NSGA-II | T_win; T_warmup; windows | 2 s (about 600 genes at S2); 5 s of greedy before each window; 20 windows per scenario | R | The warm-up makes the window start loaded (13.2) |
| Pretraining | n_step, λ1, λ2, λ3, m_E, PER, updates | 10, 1, 1, 1e−5, 0.8; α_PER 0.4, β_PER 0.6 → 1, ε_d 1.0, ε_a 0.001; 200 000 updates, stop early when held-out expert agreement stops rising | P / R | Hester et al. (750 000 updates on larger data, scaled down) |
| Fast mode | Episode length; seeds | 60 s; seed ranges 1000–1999 NSGA-II and demonstrations, 2000–2999 training, 3000–3999 evaluation (20 evaluation seeds, 5 on the testbed) | P | VFC (1 min); Maus et al. (disjoint seeds) |
| FedAvg | T_fl; upload deadline; K | 60 s fast mode, 300 s testbed, 1 h real world; 30 s; 500 steps | R | One round per fast-mode episode |

Reminders for that step: write every formula with its real-world and its simulation version; node load ρ_node and link TI both appear; the compressed frame sizes go into the time table next to the run times.

## 19. Where each decision came from

Fix Holes holds the full reasoning, the rejected options and the real-world vs simulation tables for every item below.

| Section | Source |
| --- | --- |
| 1, 3 Layers, fog zone, broker | Architecture (decisions of 8–18 Oct); N1, N6, N7, N11 |
| 4 Link model, WAN pipe, metro-edge router | Hole 22; holes 1 + 2 Part D (WAN wait estimate); N4, N9 |
| 5 Messages and fields | Older proposals, approved 25 Oct (final form); holes 1 + 2, 13, 15, 20, 23, 25 |
| 5.4 Clocks | Hole 13 |
| 6 Lifecycle, budgets | Architecture; holes 1 + 2 Part D, 3, 12, 13 |
| 7.1 TOF-Broker | Hole 10 |
| 7.2 Masks, q\* | Holes 3 (R2), 23 Part J; freshness rule |
| 7.3 All masked | Older proposals, approved 25 Oct |
| 7.5 Neighbours | Hole 11 |
| 8 Queues, slots, drops | Hole 23 (Parts A–J); hole 21 (L_max) |
| 9.1 Failure table | N14, holes 1 + 2, 3, 12, 23 |
| 9.2 Refusal, herding | Hole 3 (R1, R2, R3); hole 4 Part D |
| 9.3–9.4 Detection, retry, cancel | Holes 1 + 2; hole 23 Part F |
| 9.5 Duplicates | Hole 15 |
| 9.6 Subscriptions | Hole 20 |
| 9.7 Cut uploads | Hole 12 |
| 5.5 Identifiers; 9.7 repeated upload to the same zone | Hole 26 |
| Math check of every formula (27 Oct): 4.1, 4.3, 6, 7.1, 8.2, 8.5, 9.3, 9.7, 10.2, 11.1, 11.3, 11.4, 12.3, 13.2, 14.3, 15.2, 16.3 | Fix Holes N17, N18 |
| 10 Mobility | Architecture; N5 |
| 11.1 Inputs | Older proposals, approved 25 Oct |
| 11.3–11.4 RL step, discount, rewards | Hole 4; Architecture (r_ok, C_pen, w3) |
| 12 Energy | Hole 24 (includes hole 9) |
| 13.2, 13.4 NSGA-II and demonstrations | Hole 5 |
| 13.3 Knee and w | Hole 14 |
| 13.5 Pretraining | Older proposals, approved 25 Oct; Architecture (fading term, IQL) |
| 13.6 Neighbour order | Hole 8 |
| 13.7 FedAvg | Hole 25, local fine-tuning |
| 14 Workloads | Architecture; N10 |
| 15 Fast mode, testbed | Hole 6; hole 22 (wan-gw, mgmt_net); Architecture (environment plan) |
| 16 Experiments | Hole 7; Tracker (scenarios, verification); hole 3 (state-period experiment); hole 23 Part H |
| 17 Assumptions | Holes 16–19; Architecture (out of scope); N15 |
| Literature reading (27 Oct): 1, 7.1, 13.3, 13.5, 16.1–16.4, 17, 18 | Fix Holes N19; PCNME — Literature Review |

# Appendix A — Notation

One base letter is one kind of quantity with one unit. Approved on 15 Nov 2026; the same table is `docs/NOTATION.md` in the repository.

### Rules

One base letter is one kind of quantity with one unit. The subscript says which instance.

| Letter | Always means | Unit |
| --- | --- | --- |
| T | A duration that is computed, estimated or configured | s |
| t | An instant on one machine's clock | s |
| D, d | Deadline: D = its length, d = its instant on one clock | s |
| W | A waiting time (in a pipe or in a queue) | s |
| δ | A link delay | s |
| L | Latency measured by the car | s |
| S | A size | bit |
| B | A bandwidth or bit rate of a link | bit/s |
| E, e | Energy: E = of an attempt or task, e = per bit | J, J/bit |
| P | Power | W |
| C | A cost inside the objective and the reward | none |
| n | A count (tasks, slots, places) | none |
| w | The latency weight of the reward | none |
| r, Q, s, a, γ | Reward, Q-value, state, action, discount | none |
| ℓ | A length on the map | m |

| Mark | Always means |
| --- | --- |
| x̂ (hat) | An estimate made before the fact |
| x̄ (bar) | A mean from the configuration or the time table |
| x^lo, x^hi | The 5 % and 95 % quantiles from the time table |
| x̃ (tilde) | Normalised to the range of the Pareto front |
| x* | The chosen one (q* the assumed slot type, w* the deployed weight) |

| Index | Runs over |
| --- | --- |
| k | The 6 placement options: here, neighbour 1–4, cloud |
| q | Slot types: GPU, CPU |
| i | Tasks |
| j | Slots of one type on a node; steps of the n-step return |
| z | Zones |
| g | FedAvg rounds |
| p | Points of a Pareto front |

Names are not symbols: S1–S6 (scenarios), T1–T6 (task types), C1–C14 (cases), z1–z5 (zones), R1–R3 (herding
rules), F-GPU and F-CPU (hardware classes) never appear inside an equation and keep their names.

### Old → new

"Sections" are the sections of `docs/final-design.md` where the symbol appears.

#### Run times, budgets and timers

| Old | New | Meaning | Unit | Sections |
| --- | --- | --- | --- | --- |
| Ê_k,q | T̄^run_k,q | Mean run time of the task's type on option k, slot type q (time table) | s | 7.2, 8.3, 8.4, 11.1, 12.4 |
| Ê_strong, Ê_cloud | T̄^run_strong, T̄^run_cloud | The same on the strongest fog class / on the cloud, fastest slot type | s | 7.1 |
| Ê^low, Ê^high | T^run,lo, T^run,hi | 5 % and 95 % quantile of the run time | s | 8.4, 8.6, 9.3 |
| T_run | T^run | The real busy time of the slot for one attempt | s | 12.2 |
| B_car | T^bud_car | Time budget the car sends: D − time since creation | s | 5.3, 5.4, 6A, 9.7 |
| B_k (budget) | T^bud_k | Time budget the anchor sends to executor k | s | 5.3, 6C |
| Û_up | T̂_radio | Time the upload took on the radio, taken off the budget | s | 6A |
| T_up,k | T^to_k | Upload timeout for a send to option k | s | 9.3 |
| d_abort | T_abort | Longest step of the task on that slot type | s | 8.6, 12.2, 12.5, 14.2 |
| A_max | T_age | Wait after which a soft task may be promoted | s | 8.4, 18 |
| ℓ (in 9.3) | L_late | Lateness a class may have: 0 for hard, L_max for soft | s | 9.3 |
| t_s | t_send | Instant the anchor starts a send | s | 6C, 9.3 |

#### Links, waits and delays

| Old | New | Meaning | Unit | Sections |
| --- | --- | --- | --- | --- |
| d, d_min, d_max | δ, δ_min, δ_max | Random delay drawn per transfer and its bounds | s | 4.1 |
| d_bh, d_WAN | δ_bh, δ_WAN | The drawn delay on the backhaul / the WAN | s | 4.1 |
| δ̂_radio, δ̂_reply, δ̂_pub | δ̄_radio, δ̄_reply, δ̄_pub | Configured mean delay of a link or path | s | 5.4, 6A, 9.3 |
| ŵ_WAN | Ŵ_WAN | Moving-average estimate of the wait in the shared WAN pipe | s | 4.3, 5.2 |
| ŵ_k | Ŵ^pipe_k | Estimated pipe wait on the way to option k | s | 4.3 |
| Ŵ_k,q^κ, Ŵ_q | Ŵ^queue_k,q, Ŵ^queue_q | Estimated queue wait for the task's class; the class is written as a word (hard, soft) | s | 7.2, 8.3, 8.5, 9.3, 11.1 |
| λ (WAN estimate) | η | Smoothing factor of the moving average | none | 4.3, 18 |
| S_r | S_res | Size of a result (S stays the input size) | bit | 12.2 |
| R_z | ℓ_z | Radius of a zone's coverage circle | m | 10.2, 18 |
| l_nb | ℓ_nb | Largest centre distance that makes two zones neighbours | m | 7.5, 18 |
| p, c (in 10.2) | x_car, x_z | Position of the car / centre of the zone | m | 10.2 |

#### Slots and queues

| Old | New | Meaning | Unit | Sections |
| --- | --- | --- | --- | --- |
| U_k | 𝒬_k | Set of slot types usable for this task on option k | none | 7.2 |
| u_q | n^busy_q | Busy slots of type q | none | 5.2, 8.5 |
| Q_q | n^max_q | Capacity of the queue of slot type q | none | 7.2, 8.2, 18 |
| r (in 8.2) | n_res | Queue places kept for hard tasks | none | 7.2, 8.2, 18 |
| h (in 7.2) | n_head | Headroom: places the mask keeps free at a neighbour | none | 7.2, 9.2, 18 |
| B_q^hard, B_q^soft | Ω_q^hard, Ω_q^soft | Backlog: sum of the mean run times of the queued tasks of that class | s | 5.2, 8.5 |
| r_q,i | T^rem_q,j | Remaining mean time of the task on slot j (0 if free) | s | 5.2, 8.5 |
| n (in 4.2) | n_up | Uploads in progress on a zone's radio | none | 4.1, 4.2 |
| "the soft task S" | the soft task i | A task is always i | none | 8.4 |

#### Energy, cost and reward

| Old | New | Meaning | Unit | Sections |
| --- | --- | --- | --- | --- |
| c^time | C^time | Time cost of a task: L/D plus the lateness term, or C_drop | none | 11.4, 13.2 |
| ê | C^en | Energy cost of a task: min(E / E_ref, C^en_max) | none | 11.1, 11.4, 13.2 |
| ē | C^en_max | Cap of the energy cost | none | 11.1, 11.4, 12.3, 13.2, 18 |
| B (= 1) | r_ok | Reward bonus of a delivered task | none | 11.4, 18 |
| P (= 2) | C_pen | Extra cost of a drop inside C_drop | none | 11.4, 13.2, 18 |
| p̄_k,q | P̄_k,q | Mean dynamic power of a time-table row | W | 12.2, 12.3, 12.4, 18 |
| w1, w2 | w, 1 − w | They were other names for the same number | none | 11.4, 13.3 |

#### Learning

| Old | New | Meaning | Unit | Sections |
| --- | --- | --- | --- | --- |
| s (in 13.3) | σ | Slope of the front at the knee (s stays the state) | none | 13.3 |
| R_i | Δf_i | Range of objective i on the front: f_i^max − f_i^min | none | 13.3 |
| N (in 13.2) | N_task | Tasks in one NSGA-II window | none | 13.2 |
| N1–N4 (genes) | Nb1–Nb4 | Neighbour positions as gene values | none | 13.2 |
| n (n-step) | n_step | Length of the n-step return | none | 13.5, 18 |
| k (n-step sum) | j | Step inside the n-step return | none | 13.5 |
| k (FedAvg) | g | FedAvg round | none | 5.2, 5.3, 13.7 |
| l(a_E, a) = 0.8 | m_E | Margin of the expert loss | none | 13.5, 18 |
| α (TOF-Broker) | α_TOF | Boulder threshold | none | 7.1, 16.4, 18 |
| α, β0 (replay) | α_PER, β_PER | Prioritized-replay exponents | none | 13.5, 18 |
| h (in 16.2) | h_hot | Share of cars kept on z1's roads | none | 16.2, 18 |
| S1–S4 (DAG steps) | step 1–step 4 | Internal steps of a task | none | 14.2 |

### Symbols that stay

D, L, L_max, slack, d_A, d_k · t_now, t_free, t_rx, t_acc, t_wd · T_transfer, T̂_in,k, T̂_ret,k, T_exit, T_waste,
T_win, T_warmup, T_fl, T_ack, T_probe, T_keep, T_leave, G, m_wd, τ, Δt · S, B, B_k (bandwidth), B_WAN, R_car,
W_pipe, W_bh, W_WAN, W_radio, δ̄_k, TI, κ_up · ρ_fog, ρ_node, λ_car · c_q, n_q^hard, n_q^soft, q, q*, F_q ·
E_k, E^fwd, E^exec, E^ret, Ê_k (energy), E_ref, E_waste, e_bh, e_wan, e_down, PUE, P_tx · w, w3, w̃, w_p, w*,
C_drop, r, f₁, f₂, f̃ · s, a, s′, γ, Q(s, a), 𝒜(s), y, ε, K, H · J_DQ, J_n, J_E, a_E, λ1–λ3, ε_d, ε_a ·
v, 𝒰(a, b), u.

### Symbol → code

(Filled by the notation sweep: for every symbol of section 2 that has one, its Python name, its config key and
its message field.)

| Symbol | Python name | Config key | Message field |
| --- | --- | --- | --- |
| T̄^run_k,q | `RunRow.mean` (node/timetable.py); `OptionView.run` (zone/decision.py) | `time_table_placeholder.<type>.<class>.<slot>` (`mean`, or the number alone) | — |
| T̄^run_strong | `Derived.t_run_strong` (config.py) | computed from the time table and `tof.strongest_class` | — |
| T̄^run_cloud | `TimeTable.fastest(task, "cloud")`, its row's `mean` (config.py) | `time_table_placeholder.<type>.cloud` | — |
| T^run,lo, T^run,hi | `RunRow.low`, `RunRow.high`; `TimeRow.p5`, `TimeRow.p95` | `p5`, `p95` of a row; `time_table_placeholder.factors.p5` / `.p95` | — |
| T^run | `Report.busy` (node/executor.py) | — | — |
| T^bud_car | `Car.budget()` (car/car.py); `Upload.t_bud_car` (testbed/messages.py) | — | `t_bud_car` (testbed upload); `T_bud_car_ms` in 5.3 |
| T^bud_k | `TaskRequest.budget` (node/executor.py); `budget` in zone/zone.py | — | `budget` (TaskRequest); `budget_ms` in 5.3 |
| T̂_radio | `upload` in `anchor_deadline()` (zone/intake.py) | — | — |
| T^to_k | `t_to` of the attempt (zone/zone.py) | `timing.kappa_up` (κ_up) | — |
| T_abort | `RunRow.t_abort` | `t_abort` of a row; `time_table_placeholder.factors.t_abort` | — |
| T_age | `Scheduler.t_age` (config.py) | `scheduler.T_age` | — |
| L_late | `_cap()` (zone/zone.py): `l_max` for soft, 0 for hard | — | — |
| t_send | `t_send` (agent/reward.py, experiments/city.py) | — | — |
| δ, δ_min, δ_max | `hops.delay()` (net/hops.py) | `radio.delay`, `radio.downlink_delay`, `links.backhaul_delay`, `links.wan_delay` (each `[min, max]`) | — |
| δ_bh, δ_WAN | `d_bh`, `d_wan` (net/series.py, zone/zone.py) | `links.backhaul_delay`, `links.wan_delay` | — |
| δ̄_radio, δ̄_reply, δ̄_pub | `mean_radio_delay` (zone/intake.py); `_reply_delay()` (zone/zone.py); `StateStore.pub_delay` | the means of the delay intervals above; `links.broker_extra_delay` | — |
| Ŵ_WAN | `StateStore.wan.estimate`, `Snapshot.wan_wait` (zone/statestore.py) | — | `wan_wait` (Snapshot); `wan_wait_ms` in 5.2 |
| Ŵ^pipe_k | `wait` in `PathEstimates.t_in_cloud()` / the `backhaul_wait` argument (zone/estimates.py) | — | — |
| Ŵ^queue_k,q, Ŵ^queue_q | `wait_estimate()` (node/state.py), `Executor.wait_estimate()`, `Accepted.wait`, `OptionView.wait` | — | `wait` (Accepted); `wait_ms` in 5.3 |
| η | `StateStore.wan_eta`; `StatePlane.wan_ewma_eta` (config.py) | `stateplane.wan_ewma_eta` | — |
| S_res | `TaskType.result_bits` (config.py) | `task_types.<type>.result_kb` | — |
| ℓ_z | `Layout.zone_radius_m`; `Site.radius`, `ZoneStatus.radius` | `layout.zone_radius_m` | `radius` (zone status) |
| ℓ_nb | `Layout.l_nb_m`; `NeighbourTable.l_nb` (zone/layout.py) | `layout.l_nb_m` | — |
| x_car, x_z | `Gps.x`, `Gps.y`; `Site.x`, `Site.y` | `layout.zone_positions_m` (x_z) | `x`, `y` (GPS, zone status) |
| 𝒬_k | `usable` (zone/masks.py) | — | — |
| n^busy_q | `SlotState.n_busy` (node/state.py) | — | `n_busy` (SlotState and 5.2) |
| n^max_q | `SlotState.n_max` | `hardware.<class>.queue.<slot>` | `n_max` (SlotState and 5.2) |
| n_res | `Scheduler.n_res` | `scheduler.n_res` | — |
| n_head | `Scheduler.n_head`; `headroom` (zone/masks.py) | `scheduler.n_head` | — |
| Ω_q^hard, Ω_q^soft | `SlotState.omega_hard`, `SlotState.omega_soft` | — | `omega_hard`, `omega_soft` (SlotState); `omega_hard_ms`, `omega_soft_ms` in 5.2 |
| T^rem_q,j | `SlotState.t_rem` | — | `t_rem` (SlotState); `t_rem_ms` in 5.2 |
| n_up | `len(RadioUplink._active)` (net/radio.py) | — | — |
| C^time | `c_time()` (agent/reward.py) | — | — |
| C^en | `c_en()` (agent/reward.py, energy.py); column `mean_c_en` | — | — |
| C^en_max | `Derived.c_en_max` (config.py) | `energy.c_en_max` (`computed`) | — |
| r_ok | `Reward.r_ok`, `RewardParams.r_ok` (config.py, agent/reward.py) | `reward.r_ok` | — |
| C_pen | `Reward.c_pen` | `reward.C_pen` | — |
| P̄_k,q | `RunRow.power` (node/timetable.py); `TimeRow.p_dyn_w` (config.py) | `energy_j` of a measured row ÷ its mean; `hardware.<class>.p_dyn_w.<slot>` otherwise | — |
| w, 1 − w | `Reward.w` | `reward.w` | — |
| σ | `Knee.slope` (expert/knee.py) | — | — |
| Δf_i | `Knee.ranges` | — | — |
| N_task | the window's task count (expert/window.py) | — | — |
| Nb1–Nb4 | action indices 1–4 (agent/inputs.py, expert/nsga.py) | — | — |
| n_step | `Dqfd.n_step` (config.py) | `dqfd.n_step` | — |
| g | `RoundMsg.k`, `FedState.k` (fl/fedavg.py) | — | `k` (fl/round) |
| m_E | `Dqfd.margin` | `dqfd.margin` | — |
| α_TOF | `Tof.alpha` (config.py) | `tof.alpha`, `tof.alpha_sweep` | — |
| α_PER, β_PER | `Replay.alpha` (agent/replay.py); `Per.alpha`, `Per.beta0` (config.py) | `dqfd.per.alpha`, `dqfd.per.beta0` (`dqfd.per.beta_final` = 1) | — |
| h_hot | `Cars.h_hot` (config.py); `self.h` (experiments/city.py) | `cars.calibrated.h_hot` | — |
| step 1–step 4 | — (inside the measured run time) | — | — |

# Appendix B — Results at a glance

Fast mode, 8 methods × 6 scenarios × 20 evaluation seeds per map (960 runs per map). Each cell is the mean over the 20 seeds; the best value of a column is bold. Computed from `data/<map>/results/summary.csv` at commit 76e89b7. Confidence intervals, paired tests, the testbed runs and the limitations are in `docs/RESULTS.md`.

### VFC grid: tasks delivered on time (%)

| Method | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCNME | **99.9** | **99.8** | **98.4** | **94.2** | **93.3** | 92.1 |
| D3QN + greedy expert | **99.9** | **99.8** | 97.2 | 93.5 | 92.7 | 92.3 |
| D3QN from scratch | **99.9** | 99.6 | 86.4 | 93.6 | 93.1 | 91.9 |
| VFC PPO | 99.8 | 99.1 | 86.3 | 93.8 | 93.0 | 89.3 |
| Tang & Wong | 99.8 | 98.6 | 80.1 | 93.2 | 92.7 | 79.2 |
| QECO | 99.8 | 98.6 | 78.4 | 93.2 | 92.8 | 80.9 |
| Greedy | **99.9** | **99.8** | 92.3 | 92.6 | 92.2 | **96.9** |
| Random | 99.5 | 98.1 | 81.3 | 90.9 | 90.3 | 81.0 |

### LuST (Luxembourg): tasks delivered on time (%)

| Method | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCNME | 99.8 | **99.8** | 49.4 | 94.5 | 92.2 | 96.4 |
| D3QN + greedy expert | **99.9** | **99.8** | **53.7** | 94.0 | 92.0 | 96.7 |
| D3QN from scratch | 99.8 | 99.6 | 49.0 | 94.7 | 92.4 | 92.5 |
| VFC PPO | 99.6 | 98.9 | 44.5 | **94.9** | **92.5** | 94.0 |
| Tang & Wong | 99.7 | 98.7 | 48.8 | 94.2 | 91.8 | 81.2 |
| QECO | 99.7 | 99.1 | 46.3 | 94.1 | 92.0 | 84.8 |
| Greedy | **99.9** | 99.6 | 52.6 | 93.6 | 90.9 | **97.0** |
| Random | 99.4 | 98.3 | 43.0 | 92.0 | 89.4 | 84.5 |

### VFC grid: dynamic energy per task (J)

| Method | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCNME | 2.17 | **2.21** | **2.36** | 2.12 | 2.11 | 3.22 |
| D3QN + greedy expert | 2.17 | **2.21** | 2.57 | 2.11 | 2.10 | 3.33 |
| D3QN from scratch | 2.19 | 2.47 | 3.71 | 2.11 | 2.10 | 3.19 |
| VFC PPO | 2.62 | 3.15 | 3.92 | 2.12 | 2.12 | 4.18 |
| Tang & Wong | 3.04 | 3.25 | 3.97 | 2.13 | 2.11 | 4.37 |
| QECO | 3.04 | 3.23 | 3.96 | 2.11 | 2.10 | 4.19 |
| Greedy | **2.16** | 2.22 | 2.81 | **2.10** | **2.09** | **2.83** |
| Random | 3.19 | 3.50 | 3.98 | 2.18 | 2.16 | 4.50 |

### LuST (Luxembourg): dynamic energy per task (J)

| Method | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCNME | 2.24 | **2.22** | 2.90 | 2.13 | 2.08 | **2.66** |
| D3QN + greedy expert | 2.24 | 2.23 | 2.75 | 2.13 | 2.08 | 2.67 |
| D3QN from scratch | 2.33 | 2.53 | 2.95 | 2.14 | 2.09 | 3.08 |
| VFC PPO | 3.05 | 3.31 | 3.22 | 2.15 | 2.09 | 4.01 |
| Tang & Wong | 3.24 | 3.23 | 3.08 | 2.17 | 2.10 | 4.32 |
| QECO | 3.23 | 3.01 | 3.15 | 2.13 | 2.09 | 4.05 |
| Greedy | **2.22** | 2.27 | **2.71** | **2.12** | **2.06** | 2.70 |
| Random | 3.34 | 3.53 | 3.20 | 2.21 | 2.14 | 4.47 |

### Cold start on the VFC grid

Early-stage on time (the tasks of the first 1 000 decisions after deployment, %) · decisions until the on-time share of the last 1 000 tasks is ≥ 95 % to the end of the run (median over the seeds that get there; how many of them).

| method | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | --- | --- | --- | --- | --- | --- |
| PCNME | 99.9 ± 0.1 · 999 (20/20) | 99.9 ± 0.1 · 998 (20/20) | 99.2 ± 0.2 · 1,001 (17/20) | 96.5 ± 0.4 · 8,856 (5/20) | 96.2 ± 0.4 · 8,729 (3/20) | 95.0 ± 1.2 · 8,528 (8/20) |
| Greedy | 99.9 ± 0.1 · 999 (20/20) | 99.9 ± 0.0 · 999 (20/20) | 99.4 ± 0.2 · 8,955 (9/20) | 95.9 ± 0.4 · – (0/20) | 96.0 ± 0.4 · 9,243 (1/20) | 98.2 ± 0.3 · 3,907 (19/20) |
| D3QN + greedy demos | 99.9 ± 0.1 · 999 (20/20) | 99.9 ± 0.1 · 998 (20/20) | 99.3 ± 0.2 · 1,003 (14/20) | 95.9 ± 0.4 · 9,020 (5/20) | 96.0 ± 0.4 · 9,056 (2/20) | 96.9 ± 0.7 · 8,835 (2/20) |
| D3QN (scratch) | 99.9 ± 0.1 · 999 (20/20) | 99.8 ± 0.0 · 1,000 (20/20) | 96.1 ± 0.7 · 14,841 (3/20) | 96.1 ± 0.4 · 8,586 (4/20) | 96.0 ± 0.4 · 9,123 (4/20) | 95.7 ± 0.9 · 9,058 (4/20) |
| VFC PPO | 99.8 ± 0.1 · 999 (20/20) | 99.5 ± 0.2 · 999 (20/20) | 97.0 ± 0.9 · 14,095 (2/20) | 96.2 ± 0.4 · 8,460 (2/20) | 96.1 ± 0.4 · 9,254 (1/20) | 95.7 ± 1.1 · 8,494 (5/20) |
| Tang & Wong | 99.8 ± 0.1 · 1,000 (20/20) | 99.4 ± 0.2 · 1,004 (19/20) | 95.3 ± 1.0 · 14,384 (1/20) | 95.9 ± 0.4 · 8,820 (1/20) | 96.0 ± 0.4 · 9,175 (1/20) | 91.4 ± 1.2 · – (0/20) |
| QECO | 99.8 ± 0.1 · 999 (20/20) | 99.6 ± 0.2 · 1,001 (19/20) | 96.0 ± 1.0 · 14,624 (1/20) | 95.9 ± 0.4 · 8,819 (1/20) | 96.0 ± 0.4 · 9,269 (4/20) | 91.9 ± 1.3 · – (0/20) |
| Random | 99.6 ± 0.1 · 1,001 (20/20) | 99.1 ± 0.3 · 1,008 (18/20) | 97.1 ± 0.9 · 13,938 (1/20) | 94.3 ± 0.4 · – (0/20) | 94.2 ± 0.4 · – (0/20) | 93.6 ± 1.4 · – (0/20) |

Sources: `data/vfc/results/report.csv` (early_on_time), `data/vfc/results/extra-runs.csv` (decisions until stable).

# Appendix C — References

Numbers in square brackets in this document refer to this list. 16 of the 21 are in the thesis paper library; the 5 that are not are marked. [1] is the base paper of this work: its energy model (Eq. 1–9), delay model (Eq. 10–16) and bi-objective NSGA-II problem (Eq. 27) are the starting point of sections 4, 12 and 13.

| # | Reference | In the library |
| --- | --- | --- |
| 1 | V. Jafari, M. H. Rezvani, "Joint optimization of energy consumption and time delay in IoT-fog-cloud computing environments using NSGA-II metaheuristic algorithm," J. Ambient Intell. Humanized Comput., 14:1675–1698, 2023. [doi](https://doi.org/10.1007/s12652-021-03388-2) | Yes |
| 2 | M. P. Toopchinezhad, M. Ahmadi, "Deep Reinforcement Learning for Delay-Optimized Task Offloading in Vehicular Fog Computing," arXiv:2410.03472, 2024 (the VFC environment). [arXiv](https://arxiv.org/abs/2410.03472) | Yes |
| 3 | F. S. Behbehani, T. Elgorashi, J. M. H. Elmirghani, "Power Minimization in Vehicular Cloud Architecture," arXiv:2102.09011. [arXiv](https://arxiv.org/abs/2102.09011) | Yes |
| 4 | I. Mokni, S. Yassa, "A multi-objective approach for optimizing IoT applications offloading in fog–cloud environments with NSGA-II" (Tof-NSGAII), J. Supercomput., 80(19):27034–27072, 2024. [doi](https://doi.org/10.1007/s11227-024-06431-z) | Yes |
| 5 | M. Tang, V. W. S. Wong, "Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing Systems," IEEE Trans. Mobile Comput., 21(6):1985–1997, 2022. [arXiv](https://arxiv.org/abs/2005.02459) | Yes |
| 6 | I. Rahmaty, H. Shah-Mansouri, A. Movaghar, "QECO: A QoE-Oriented Computation Offloading Algorithm Based on Deep Reinforcement Learning for Mobile Edge Computing," IEEE Trans. Netw. Sci. Eng., 12(4), 2025. [doi](https://doi.org/10.1109/TNSE.2025.3556809) | Yes |
| 7 | K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," IEEE Trans. Evol. Comput., 6(2):182–197, 2002. [doi](https://doi.org/10.1109/4235.996017) | Yes |
| 8 | I. Das, "On characterizing the 'knee' of the Pareto curve based on normal-boundary intersection," Struct. Optim., 18(2–3):107–115, 1999 | **No** (cited through [9], which credits it) |
| 9 | X. Zhang, Y. Tian, Y. Jin, "A knee point-driven evolutionary algorithm for many-objective optimization," IEEE Trans. Evol. Comput., 19(6):761–776, 2015. [doi](https://doi.org/10.1109/TEVC.2014.2378512) | **No** (Das [8] and this paper are cited for the knee definition; neither is in the library) |
| 10 | T. Hester et al., "Deep Q-learning from Demonstrations," AAAI, 32(1), 2018. [link](https://ojs.aaai.org/index.php/AAAI/article/view/11757) | Yes |
| 11 | S. J. Bradtke, M. O. Duff, "Reinforcement Learning Methods for Continuous-Time Markov Decision Problems," NIPS 7, 1995 | Yes |
| 12 | Y. Yin et al., "Mobility-Aware Assisted Deep Reinforcement Learning for Collaborative Task Migration and Resource Allocation in Vehicular Edge Computing," IEEE Trans. Veh. Technol., 2026. [doi](https://doi.org/10.1109/tvt.2026.3660321) | Yes |
| 13 | C. E. Shannon, "A Mathematical Theory of Communication," Bell Syst. Tech. J., 1948 | **No** (textbook result) |
| 14 | Uptime Institute, Global Data Center Survey 2023 | **No** (web report) |
| 15 | F. Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning," CVPR, 2020 | Yes |
| 16 | Ultralytics, YOLOv8 model table with published inference times (docs.ultralytics.com; cite the release used) | **No** (web documentation) |
| 17 | H. van Hasselt, A. Guez, D. Silver, "Deep Reinforcement Learning with Double Q-learning," AAAI, 2016. [arXiv](https://arxiv.org/abs/1509.06461) | Yes |
| 18 | Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning," ICML, 2016. [arXiv](https://arxiv.org/abs/1511.06581) | Yes |
| 19 | L. Kleinrock, Queueing Systems, Volume 1: Theory, Wiley, 1975 | Yes (a scan) |
| 20 | J. Branke, K. Deb, H. Dierolf, M. Osswald, "Finding Knees in Multi-objective Optimization," PPSN VIII, LNCS 3242, 2004 | Yes |
| 21 | J. Blank, K. Deb, "pymoo: Multi-Objective Optimization in Python," IEEE Access, 8:89497–89509, 2020. [doi](https://doi.org/10.1109/ACCESS.2020.2990567) | Yes |

Other works named in the text (the closest prior work in section 1, the baselines in 16.1) are listed with links in the References document of the project.

# Appendix D — History of changes

Changes made to the design after the printed version of 27 Oct 2026, newest first by topic. CHANGES and QUESTIONS refer to `docs/CHANGES.md` and `docs/QUESTIONS.md` in the repository.

| Date | Section | Change | Note |
| --- | --- | --- | --- |
| 28 Oct | 16.2 | ρ_fog uses every task of the whole mix, boulders included (not pebbles only), so S4–S6 keep S2's arrival rate | CHANGES C2 (QUESTIONS Q1) |
| 16 Nov | 13.7 | Phase 2 (FedAvg + fine-tuning) is evaluated in fast mode only; on the testbed it is future work (since 15.5's trainer process holds the learning network, FedAvg would run there) | QUESTIONS Q64 |
| 16 Nov | all | The notation reaches the code: Python names, config keys, message fields and CSV columns renamed as in docs/NOTATION.md section 6 (for example mean_e_hat → mean_c_en, e_bar → c_en_max, reward B, P → r_ok, C_pen); the existing result files were rewritten the same way; config_hash changes with the keys | docs/NOTATION.md |
| 15 Nov | all | Notation: one base letter per kind of quantity, one unit (docs/NOTATION.md); names only, no rule, formula, number or behaviour changes | docs/NOTATION.md |
| 11 Nov | 15.5 | On the testbed a zone that learns online gets a second CPU: its trainer runs in its own process (replay, updates), the zone sends it the samples and installs the trainer's newest weights between two decisions. Fast mode, which has no compute time, learns in place as before | QUESTIONS Q62 |
| 7 Nov | 9.6 | A car always knows whether its serving zone serves it: the attach is repeated until confirmed, and a zone names the cars it does not serve in its reply to their GPS, which then attach again (testbed transport detail) | QUESTIONS Q61 |
| 6 Nov | 16.2, 18 | S6 (mixed fog) has loose deadlines, not tight: with tight ones every method was near a capacity ceiling (the hard tasks need 80 % of the 4 GPU slots with 14–20 ms of slack), so S6 measured which tasks a method sacrifices rather than placement on unequal nodes. The TOF-Broker is now exercised in S4 and S5 | QUESTIONS Q60 |
| 28 Oct | 13.2 | Random draws keyed by task, not by event order: a run time is a uniform keyed by (seed, task id, attempt) through the inverse CDF of the row it runs on; a delay is keyed by (seed, task id, attempt, hop); arrivals and mobility stay sequential per car | CHANGES C3 (QUESTIONS Q6) |
| 28 Oct | 8.5, 5.2 | The wait estimate is water-filling over each slot's remaining time T^rem_q,j (published per slot instead of their sum R): Σ max(0, Ŵ^queue_q − T^rem_q,j) = Ω_q; still 0 while a slot is free; for c_q = 1 the same as before | CHANGES C4 |
| 28 Oct | 14.4, 18 | Run times are lognormal, fitted to each row's mean and p95; the placeholder p5 factor is 0.75 (was 0.9); measured rows report the fitted p5 next to the measured one | CHANGES C5 (QUESTIONS Q7) |
| 29 Oct | 4.1, 4.2, 6A.3, 15.3, 17 | The radio uplink shares its airtime equally among the cars uploading at once (each advances at R_car ÷ n_up) instead of one car at a time; W_radio is the extra push time from sharing, still known exactly at t_rx | CHANGES C6 (QUESTIONS Q18) |
| 29 Oct | 18, 14.3 | The car's encode time is a config value, `timing.encode` (M, placeholder 0); the upload starts after it, and T^bud_car does not subtract it again | QUESTIONS Q14 |
| 29 Oct | 15.2 | Every drawn time is rounded to whole nanoseconds and every radio rate to whole bit/s, so one seed gives the same run on every OS | web review of PR #20 |
| 29 Oct | 16.2, 18 | λ_car is divided by the share of task creations with coverage, measured once per map by a positions-only calibration run (tag C); frames made in a gap are reported as uncovered | QUESTIONS Q29 |
| 29 Oct | 16.2, 18 | The hotspot share h_hot (cars on z1's roads) is set by the map calibration run so that z1 = 1.6 × the mean load | QUESTIONS Q30 |
| 29 Oct | 9.3 | The upload timeout T^to starts when the zone's own send has finished; before that only the deadline cap applies, and reaching it drops the attempt with no re-send | CHANGES C7 (QUESTIONS Q31) |
| 29 Oct | 16.2 | S3 shows an overloaded system: under Greedy the fog GPUs saturate (97–100 %); under cloud-only placement the WAN queue grows without limit (M2 test); it no longer says the WAN congests under every method | M3 acceptance runs after C7 |
| 29 Oct | 9.6, 6E.15 | The anchor keeps its subscriptions for a car until G after its last task of the car closes (not at the close), so the energy reports G waits for still reach it | CHANGES C8 (QUESTIONS Q35) |
| 29 Oct | 13.2 | The initial NSGA-II population holds the greedy plan, an energy-greedy plan and mutated copies of both (half the population), the rest random; so no front point is dominated by greedy | CHANGES C9 (QUESTIONS Q38) |
| 30 Oct | 16.3 | Evaluation seeds are never used for planning that any method learns from; the knee reference plans on evaluation seeds (a reference no method learns from), only for S2 and S4 on the first 5 evaluation seeds | CHANGES C10 (QUESTIONS Q45) |
| 17 Nov | 8.5, 16.3 | Leftover symbols: the pooled estimate in words (was R_q), N_early (was N); NOTATION.md section 2 extended | NOTATION.md |
| 17 Nov | all | Version 3: reorganised front part (one page, inputs and outputs, components, planned versus run), a purpose box on every section, worked examples and explained equations in 4, 7, 8.5, 10.2, 11, 12, 13 and 16, references and results as appendices. No rule, formula or number changed | this document |
