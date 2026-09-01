# AlgoGuard Refactor Plan: Data Packets and Real-Time Tracking

Status: Phases 0–2 implemented; research validation and retraining remain
Scope: Whole system (runtime app + research workspace)
Date: 2026-08-31

Implementation update (2026-09-01): the traffic-source abstraction, FlowTracker,
PCAP replay, live capture, UI integration, and capture-session persistence are now
present in the runtime. The phase descriptions below preserve the original plan;
Phases 3–4 remain the active research roadmap.

## 1. Summary and Recommendation

AlgoGuard today is a flow-replay system: the Live Monitor reads labelled UNSW-NB15 flow records from bundled CSVs and pushes them through the deployed Stacking Ensemble one at a time. Nothing in the pipeline ever sees a real packet, and "real time" is a timer that paces the replay.

This refactor makes AlgoGuard operate on **real data packets** and perform **real-time tracking** of actual traffic. The recommended approach:

> **Capture live packets, aggregate them into flows in real time, and compute the same 15 UNSW-NB15-style features the deployed model already expects.** The packet layer becomes real; the trained model, preprocessing pipeline, quality gate, and deployment machinery are all preserved.

Why this shape and not the alternatives:

- **Raw packet-level classification** (feeding individual packets or payload bytes to a model) would discard the entire existing research base. The six candidate models, the ranking methodology, the UNSW-NB15 datasets, and every result table in `research/` are flow-based. Packet-level ML also needs packet-labelled datasets and typically deep-learning architectures — a research restart, not a refactor.
- **Staying with flow CSVs** and only polishing the replay is not a refactor; it keeps the "does not capture live packets" limitation that this effort exists to remove.
- **Packets aggregated to flows** keeps model compatibility (the deployed artifact's `feature_columns` schema is unchanged), makes the detection surface real, and turns the flow-builder itself into a publishable research contribution: "can UNSW-NB15-trained ensembles classify flows metered live on commodity Windows hardware?"

Real-time tracking becomes **dual-mode through one pipeline**:

- **Live mode** — packets sniffed from a network interface, tracked into flows, classified as flows expire. This is the new headline capability.
- **Replay mode** — recorded PCAPs (and the existing CSVs) pushed through the *same* flow-tracking and feature-extraction code. This keeps research reproducible: every experiment can be re-run bit-for-bit, and ground-truth labels remain available for accuracy reporting.

The two modes share every stage after the source, so results from replay experiments transfer directly to live operation.

## 2. Current Architecture (what the refactor touches)

Grounded in the code as it stands today:

| Component | Today | After refactor |
| --- | --- | --- |
| `services/live_monitor_service.py` | Replays CSV rows on a timer in one worker thread; synthesizes fake endpoint IPs (`_synthetic_endpoints`) | Consumes classified flows from the new streaming pipeline; real endpoints; source becomes pluggable |
| Live Monitor UI (`/monitor`) | Dataset/speed/order pickers; match-vs-label column | Source picker (interface or recording); speed/order apply to replay only; match column only when ground truth exists |
| `services/simulation_service.py` | Manual single-flow prediction from form fields | Unchanged (still useful for what-if analysis) |
| `services/preprocessing_service.py`, `train.py` | CSV in, leakage-safe pipelines, ranking, quality gate | Unchanged, plus ability to train on flow CSVs *exported by the new flow meter* |
| `services/database_service.py` (SQLite) | `network_traffic` rows carry synthetic IPs and CSV features | Real 5-tuple (src/dst IP and port, protocol), capture session records, same persistence caps |
| `research/` | Archived notebooks, UNSW-NB15 raw CSV | Active workspace validating the packet pipeline (Section 6) |

Key properties to preserve: the app stays inference-only (training remains a `train.py` terminal task); one monitoring session per process; the browser polling model (`/monitor/status` with `since` sequence numbers); storage modes and the 300-record persistence cap; the deployment quality gate.

## 3. Target Architecture

```text
                    +---------------------------+
  Live mode         |                           |
  NIC --(Npcap)--> [PacketSource]               |
                    |                           |
  Replay mode       |      FlowTracker          |     ClassificationEngine
  .pcap file -----> [PcapSource]  ------------> | --> (deployed Stacking     --> feed / DB / alerts
                    |  5-tuple table,           |      artifact, unchanged)
  Legacy mode       |  timeouts, TCP state      |
  .csv file ------> [CsvSource*]                |
                    +---------------------------+
                     * CSV rows bypass FlowTracker (already flows)
```

### 3.1 Packet layer

- **Capture library: Scapy** (`AsyncSniffer`) over **Npcap** on Windows. Scapy is pure Python, BSD-friendly, already ubiquitous in IDS research, and its `AsyncSniffer` with a `prn` callback fits AlgoGuard's existing thread-plus-events design. PyShark is the fallback if dissection depth is ever needed, but it drags a tshark subprocess along and is slower per packet. NFStream is the off-the-shelf alternative for flow metering, but its feature set does not match UNSW-NB15's 15 features one-to-one — a custom tracker keeps the deployed model's schema exact, and is itself research material.
- Capture requires **Npcap installed** and (typically) Administrator rights. This retires the README's "no capture driver or administrator rights" property — the plan is to *degrade gracefully*: if Npcap or privileges are missing, the Live Monitor still offers replay modes and says why live mode is unavailable. Npcap's free license covers personal/academic use on a limited number of systems; fine for this project's stated academic scope.
- A **BPF filter** (e.g. `ip and (tcp or udp)`) keeps the callback cheap; ICMP/ARP can be added later.

### 3.2 FlowTracker (new `services/flow_tracker_service.py`)

A dictionary keyed by the canonical 5-tuple `(src_ip, src_port, dst_ip, dst_port, proto)`, direction assigned by who sent the first packet. Per-flow accumulators: packet/byte counts per direction, first/last timestamps per direction, TTL of first packet per direction, inter-arrival tracking. Flows are emitted (classified) when:

- TCP teardown is seen (FIN handshake or RST), or
- an **idle timeout** expires (proposed default 15 s, configurable), or
- an **active timeout** caps long-lived flows (proposed default 120 s), so a big download is reported in slices rather than never.

A single tracker thread owns the table; the sniffer callback only enqueues packets onto a bounded `queue.Queue` (drop-and-count on overflow so a traffic burst degrades statistics, not the process). This mirrors the current lock discipline in `live_monitor_service.py` — decide under the lock, write to SQLite outside it.

### 3.3 Feature extraction — mapping packets to the 15 deployed features

The deployed artifact's `feature_columns` stay authoritative. Computation per emitted flow:

| Feature | From packets | Fidelity notes |
| --- | --- | --- |
| `dur` | last_ts − first_ts | exact |
| `proto` | IP protocol field → `tcp`/`udp`/... | exact |
| `service` | port → service map (http, dns, ftp, ssh, smtp, ...; else `-`) | UNSW used payload inspection via Argus/Bro; port map is an approximation — validate in research |
| `state` | derived from TCP flag history (FIN, CON, INT, RST...; UDP → CON/INT) | approximation of Argus state machine — validate in research |
| `spkts` / `dpkts` | per-direction packet counts | exact |
| `sbytes` / `dbytes` | per-direction byte counts | UNSW counts link-layer bytes; decide L2 vs L3 accounting in research |
| `rate` | (spkts + dpkts − 1) / dur | matches UNSW definition; guard dur=0 |
| `sttl` / `dttl` | TTL of first packet in each direction | exact |
| `sload` / `dload` | bits per second per direction | exact given byte accounting choice |
| `sinpkt` / `dinpkt` | mean inter-arrival ms per direction | exact |

The honest risk: UNSW-NB15's features were generated by Argus and Bro/Zeek from 2015 lab traffic. A home-grown meter on 2026 traffic **will** shift some distributions (`service` and `state` especially). That is exactly what the research phase measures before anyone trusts live verdicts (Section 6), and why retraining on flows exported by *our own meter* is the end-state (Phase 4).

### 3.4 Real-time engine and app integration

- `live_monitor_service.py` is refactored around a **TrafficSource interface**: `LiveCaptureSource(interface, bpf)`, `PcapReplaySource(path, pacing)`, `CsvReplaySource(path, speed, order)` (the current behaviour, preserved). The worker loop consumes emitted flows instead of iterating CSV rows; classification, event feed, totals, persistence caps, and polling protocol survive intact.
- **Speed/order controls** apply only to replay sources; live mode runs at wire pace. `row_total` becomes unknown for live mode (UI shows a running count instead of progress-vs-total).
- `_synthetic_endpoints` is deleted; flows carry their real 5-tuple. For replay-from-CSV, endpoints display as `replay` instead of fabricated private IPs.
- The **match/mismatch statistic** (prediction vs recorded label) exists only for labelled replays; live flows have no ground truth and the UI must say so rather than showing a vacuous 100%.
- Latency tracking gains a second number: **detection lag** (flow-end → verdict) alongside the existing **inference latency** (predict call). Both go into the event feed; detection lag is the real-time claim a paper or defense panel will probe.
- New DB touches (additive migrations, consistent with `migrate.py` policy): `network_traffic` already has `source_ip`, `destination_ip`, `source_port`, `destination_port`, and `protocol` columns — they simply start receiving real values instead of synthetic IPs and zero ports. Genuinely new: a `capture_session` table (interface, filter, packet/drop counts, timings), and flows persisted from live mode tagged `live_capture` alongside the existing `live_monitor` tag.
- Privacy note for the manual/README: capturing on a real interface observes genuine traffic metadata on the host network. Only flow metadata (never payloads) is stored, sessions are capped as today, and capture should be run on networks the operator is authorized to monitor.

## 4. Technology Choices (with rationale)

| Concern | Choice | Rationale |
| --- | --- | --- |
| Packet capture | Scapy `AsyncSniffer` + Npcap | Pure-Python callback model fits existing threading; standard in the literature; PCAP read/write built in |
| Flow metering | Custom FlowTracker | Exact UNSW-NB15 schema; NFStream/CICFlowMeter emit different feature sets and would force model retraining on day one |
| PCAP replay | Scapy `PcapReader` through the same tracker | One code path for live and replay; reproducible experiments |
| UI transport | Keep existing polling (`/monitor/status`) | Already re-attach-safe on refresh; SSE/WebSockets deferred — they change Flask deployment assumptions for little research value |
| Model runtime | Unchanged joblib Stacking artifact | The refactor's core promise: real packets, same model |
| New dependency footprint | `scapy` only (requirements.txt) | Npcap is a system install, documented in README |

## 5. Phased Migration

Each phase leaves the app releasable and `pytest`/`ruff` green.

**Phase 0 — Groundwork (small).** Add `scapy`; document Npcap install and privilege requirements; introduce the `TrafficSource` abstraction inside `live_monitor_service.py` with `CsvReplaySource` as the only implementation. Pure refactor, zero behaviour change, tests updated to construct sources.

**Phase 1 — FlowTracker + PCAP replay.** Implement `flow_tracker_service.py` with unit tests built from small hand-made PCAPs (synthetic TCP handshake/teardown, UDP exchanges, timeout cases). Add `PcapReplaySource`. The Live Monitor can now stream a recorded capture through the deployed model end to end. This is the technical heart and is fully testable offline.

**Phase 2 — Live capture.** Add `LiveCaptureSource` with interface enumeration, BPF filter, drop counters, graceful degradation when Npcap/privileges are absent. UI gains the source picker and live-mode statistics; DB migrations for real endpoints and capture sessions land here.

**Phase 3 — Research validation (Section 6).** Feature-fidelity and latency studies; go/no-go evidence for trusting live verdicts.

**Phase 4 — Close the loop.** `train.py` accepts flow CSVs exported by our own meter (the CSV contract already fits — features before target, binary label); retrain and redeploy via the existing quality gate on captured-and-labelled data. Update README/user manual; retire "AlgoGuard does not capture live packets."

Suggested order of effort: 0 and 1 together (~the bulk of new code), 2 next, 3 in parallel with 2 once PCAPs flow, 4 last.

## 6. Research Workspace Refactor

`research/` graduates from archive to the active experimental arm. Structure:

```text
research/
|-- captures/            # recorded PCAPs (gitignored; scripts to regenerate)
|-- dataset/             # as today, plus meter-exported flow CSVs
|-- notebooks/
|   |-- archive/         # today's notebooks, moved, untouched (cited evidence)
|   |-- feature_fidelity.ipynb
|   |-- realtime_latency.ipynb
|   `-- retraining_eval.ipynb
`-- results/
```

Three experiments, in order:

1. **Feature fidelity.** UNSW publishes the original PCAPs alongside the flow CSVs. Run our FlowTracker over UNSW PCAP slices and compare emitted features against the published Argus-derived rows for the same connections — per-feature agreement and, where they diverge (`state`, `service`, byte accounting), quantified impact on the deployed model's verdicts. This is the single most important number the refactor produces: it says whether an UNSW-trained model can be trusted on our meter's output.
2. **Real-time latency.** Detection lag and throughput of the pipeline on replayed and live traffic: packets/second sustained before queue drops, flow-emission latency by timeout class, inference latency under load. Extends the existing efficiency tables (`final_table2_efficiency.csv`) from model-only to system-level.
3. **Retraining on self-metered data.** Export labelled flows (replayed UNSW PCAPs through our meter keep their labels; optionally add benign traffic captured locally), retrain the six candidates via the unchanged `train.py`, and compare against the current results tables. If fidelity in (1) is poor, this is the corrective; if good, it is the confirmation.

Labelling of any locally captured traffic stays conservative: local captures are used as *benign/background* examples only, and attack examples continue to come from published labelled datasets — no attack generation is needed for this plan.

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Feature distribution shift (Argus vs our meter) | Live verdicts untrustworthy | Experiment 1 before trusting live mode; Phase 4 retraining path |
| Npcap/admin unavailable on target machine | Live mode unusable | Graceful degradation to replay modes with a clear message |
| Packet bursts overwhelm Python callback | Dropped packets, skewed flows | Bounded queue with drop counters surfaced in UI; BPF filtering; document sustained-rate ceiling from Experiment 2 |
| Long-lived flows never emit | Missed detections | Active timeout slicing (120 s default) |
| Real IPs in SQLite raise privacy stakes | Scope/ethics questions | Metadata only, existing persistence caps, authorization note in docs |
| Scope creep into IPS/blocking | Never-ending project | Explicitly out of scope (below) |

## 8. Out of Scope

Traffic blocking or any IPS behaviour; payload inspection or DPI; multi-host/distributed capture; replacing the polling UI with push transport; non-Windows capture support (the code won't preclude it, but only Windows + Npcap is tested); production SOC hardening — the academic, controlled-local-use scope stands.

## 9. Open Questions for Review

1. Byte accounting: match UNSW's link-layer byte counts (needs L2 capture) or use IP-layer bytes and rely on retraining? Proposal: IP-layer, measure the difference in Experiment 1.
2. Timeout defaults (15 s idle / 120 s active) — happy to tune from Experiment 2.
3. Should manual Prediction stay exactly as is? (Plan says yes.)
4. Do we want a minimal `capture.py` CLI (peer of `train.py`) for headless PCAP recording, or is in-app capture enough?

---

*Originally prepared as the review-first deliverable for the packets/real-time
refactor. Phases 0–2 are implemented; the deployed-model contract and quality
gate remain unchanged.*
