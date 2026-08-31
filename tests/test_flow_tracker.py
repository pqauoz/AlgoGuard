"""Unit tests for packet-to-flow aggregation and UNSW-style feature extraction."""

import pytest

from services.flow_tracker_service import (
    ACTIVE_TIMEOUT_SECONDS,
    IDLE_TIMEOUT_SECONDS,
    FlowTracker,
    PacketInfo,
    compute_features,
)

CLIENT = "192.168.1.10"
SERVER = "93.184.216.34"


def tcp_packet(ts, src, dst, sport, dport, length=60, ttl=64, flags="A"):
    return PacketInfo(
        ts=ts,
        src_ip=src,
        dst_ip=dst,
        src_port=sport,
        dst_port=dport,
        proto="tcp",
        length=length,
        ttl=ttl,
        tcp_flags=flags,
    )


def udp_packet(ts, src, dst, sport, dport, length=80, ttl=64):
    return PacketInfo(
        ts=ts,
        src_ip=src,
        dst_ip=dst,
        src_port=sport,
        dst_port=dport,
        proto="udp",
        length=length,
        ttl=ttl,
    )


def http_session(start=1000.0):
    """A complete TCP session to port 80: handshake, data, full teardown."""
    c2s = {"src": CLIENT, "dst": SERVER, "sport": 52100, "dport": 80, "ttl": 62}
    s2c = {"src": SERVER, "dst": CLIENT, "sport": 80, "dport": 52100, "ttl": 254}
    return [
        tcp_packet(start + 0.000, flags="S", length=60, **c2s),
        tcp_packet(start + 0.020, flags="SA", length=60, **s2c),
        tcp_packet(start + 0.040, flags="A", length=52, **c2s),
        tcp_packet(start + 0.060, flags="PA", length=350, **c2s),
        tcp_packet(start + 0.120, flags="PA", length=1200, **s2c),
        tcp_packet(start + 0.140, flags="A", length=52, **c2s),
        tcp_packet(start + 0.200, flags="FA", length=52, **c2s),
        tcp_packet(start + 0.220, flags="FA", length=52, **s2c),
        tcp_packet(start + 0.240, flags="A", length=52, **c2s),
    ]


def run_all(tracker, packets):
    emitted = []
    for packet in packets:
        emitted.extend(tracker.add_packet(packet))
    return emitted


def test_complete_tcp_session_emits_one_finished_flow():
    tracker = FlowTracker()
    flows = run_all(tracker, http_session())
    flows.extend(tracker.flush())

    assert len(flows) == 1
    flow = flows[0]
    features = flow["features"]

    assert flow["source_ip"] == CLIENT
    assert flow["destination_ip"] == SERVER
    assert flow["source_port"] == 52100
    assert flow["destination_port"] == 80
    assert flow["end_reason"] == "fin"

    assert features["proto"] == "tcp"
    assert features["service"] == "http"
    assert features["state"] == "FIN"
    # The trailing ACK of the teardown is swallowed, not a second flow.
    assert features["spkts"] == 5
    assert features["dpkts"] == 3
    assert features["sbytes"] == 60 + 52 + 350 + 52 + 52
    assert features["dbytes"] == 60 + 1200 + 52
    assert features["sttl"] == 62
    assert features["dttl"] == 254
    assert features["dur"] == pytest.approx(0.220, abs=1e-6)
    assert features["rate"] == pytest.approx((8 - 1) / 0.220, rel=1e-4)
    assert features["sload"] == pytest.approx(features["sbytes"] * 8 / 0.220, rel=1e-4)
    assert features["sinpkt"] > 0
    assert features["dinpkt"] > 0


def test_reset_ends_a_flow_with_rst_state():
    tracker = FlowTracker()
    flows = run_all(
        tracker,
        [
            tcp_packet(1.0, CLIENT, SERVER, 52101, 443, flags="S"),
            tcp_packet(1.1, SERVER, CLIENT, 443, 52101, flags="R"),
        ],
    )
    assert len(flows) == 1
    assert flows[0]["features"]["state"] == "RST"
    assert flows[0]["features"]["service"] == "ssl"


def test_udp_exchange_is_con_and_one_sided_udp_is_int():
    tracker = FlowTracker()
    flows = run_all(
        tracker,
        [
            udp_packet(1.0, CLIENT, SERVER, 40000, 53),
            udp_packet(1.1, SERVER, CLIENT, 53, 40000),
            udp_packet(2.0, CLIENT, "10.9.9.9", 40001, 9999),
        ],
    )
    flows.extend(tracker.flush())

    by_port = {flow["destination_port"]: flow for flow in flows}
    assert by_port[53]["features"]["state"] == "CON"
    assert by_port[53]["features"]["service"] == "dns"
    assert by_port[9999]["features"]["state"] == "INT"
    assert by_port[9999]["features"]["service"] == "-"


def test_idle_flows_expire_after_the_idle_timeout():
    tracker = FlowTracker()
    tracker.add_packet(udp_packet(1.0, CLIENT, SERVER, 40000, 53))
    late = tracker.add_packet(
        udp_packet(1.0 + IDLE_TIMEOUT_SECONDS + 1.0, CLIENT, "10.1.1.1", 40002, 53)
    )
    assert len(late) == 1
    assert late[0]["end_reason"] == "idle_timeout"
    assert late[0]["destination_ip"] == SERVER


def test_long_lived_flows_are_sliced_by_the_active_timeout():
    tracker = FlowTracker()
    emitted = []
    for step in range(3):
        ts = 1.0 + step * (ACTIVE_TIMEOUT_SECONDS / 2 + 1)
        emitted.extend(
            tracker.add_packet(tcp_packet(ts, CLIENT, SERVER, 52102, 80, flags="A"))
        )
    assert any(flow["end_reason"] == "active_timeout" for flow in emitted)


def test_direction_is_assigned_by_first_packet():
    tracker = FlowTracker()
    tracker.add_packet(udp_packet(1.0, SERVER, CLIENT, 53, 40010))
    tracker.add_packet(udp_packet(1.1, CLIENT, SERVER, 40010, 53))
    flows = tracker.flush()
    assert len(flows) == 1
    assert flows[0]["source_ip"] == SERVER
    assert flows[0]["features"]["spkts"] == 1
    assert flows[0]["features"]["dpkts"] == 1


def test_syn_without_answer_is_req():
    tracker = FlowTracker()
    tracker.add_packet(tcp_packet(1.0, CLIENT, SERVER, 52103, 80, flags="S"))
    flows = tracker.flush()
    assert flows[0]["features"]["state"] == "REQ"


def test_features_are_the_fifteen_deployed_columns():
    tracker = FlowTracker()
    tracker.add_packet(udp_packet(1.0, CLIENT, SERVER, 40000, 53))
    flow = tracker.flush()[0]
    assert set(flow["features"]) == {
        "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes",
        "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sinpkt", "dinpkt",
    }


def test_zero_duration_flow_has_safe_rates():
    tracker = FlowTracker()
    tracker.add_packet(udp_packet(5.0, CLIENT, SERVER, 40000, 161))
    features = compute_features(next(iter(tracker._flows.values())))
    assert features["dur"] == 0.0
    assert features["rate"] == 0.0
    assert features["sload"] == 0.0
    assert features["sinpkt"] == 0.0
