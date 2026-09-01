"""Tests for the Live Monitor's traffic sources, including PCAP replay."""

import threading
import time

import pytest
import scapy.all as scapy_all

from services import traffic_source_service as traffic_sources
from services import live_monitor_service as monitor
from services.traffic_source_service import (
    CsvReplaySource,
    PcapReplaySource,
    TrafficSourceCancelled,
    TrafficSourceError,
    live_capture_available,
)


def write_sample_pcap(path, sessions=3):
    """Write a small capture: N complete HTTP sessions plus one DNS exchange."""
    Ether = scapy_all.Ether
    IP = scapy_all.IP
    TCP = scapy_all.TCP
    UDP = scapy_all.UDP

    packets = []
    base = 1_700_000_000.0
    for index in range(sessions):
        client = f"192.168.1.{10 + index}"
        server = "203.0.113.7"
        sport = 51000 + index
        start = base + index * 2.0
        stream = [
            (0.00, IP(src=client, dst=server, ttl=62) / TCP(sport=sport, dport=80, flags="S")),
            (0.02, IP(src=server, dst=client, ttl=250) / TCP(sport=80, dport=sport, flags="SA")),
            (0.04, IP(src=client, dst=server, ttl=62) / TCP(sport=sport, dport=80, flags="A")),
            (
                0.06,
                IP(src=client, dst=server, ttl=62)
                / TCP(sport=sport, dport=80, flags="PA")
                / (b"x" * 120),
            ),
            (
                0.10,
                IP(src=server, dst=client, ttl=250)
                / TCP(sport=80, dport=sport, flags="PA")
                / (b"y" * 400),
            ),
            (0.14, IP(src=client, dst=server, ttl=62) / TCP(sport=sport, dport=80, flags="FA")),
            (0.16, IP(src=server, dst=client, ttl=250) / TCP(sport=80, dport=sport, flags="FA")),
            (0.18, IP(src=client, dst=server, ttl=62) / TCP(sport=sport, dport=80, flags="A")),
        ]
        for offset, payload in stream:
            frame = Ether() / payload
            frame.time = start + offset
            packets.append(frame)

    query = (
        Ether() / IP(src="192.168.1.50", dst="198.51.100.9", ttl=64) / UDP(sport=40000, dport=53)
    )
    query.time = base + 30.0
    answer = (
        Ether() / IP(src="198.51.100.9", dst="192.168.1.50", ttl=120) / UDP(sport=53, dport=40000)
    )
    answer.time = base + 30.02
    packets.extend([query, answer])

    scapy_all.wrpcap(str(path), packets)
    return path


def test_pcap_source_aggregates_packets_into_flows(tmp_path):
    pcap_path = write_sample_pcap(tmp_path / "sample.pcap", sessions=3)

    source = PcapReplaySource(str(pcap_path))
    source.prepare()

    assert source.row_total is None  # sequential replay does not pre-buffer the full capture
    events = []
    while True:
        try:
            events.append(source.next_event())
        except StopIteration:
            break

    assert source.row_total == 4  # three HTTP sessions plus one DNS exchange

    http_events = [event for event in events if event["record"]["service"] == "http"]
    dns_events = [event for event in events if event["record"]["service"] == "dns"]
    assert len(http_events) == 3
    assert len(dns_events) == 1

    event = http_events[0]
    assert event["actual"] is None
    assert event["source_ip"].startswith("192.168.1.")
    assert event["destination_ip"] == "203.0.113.7"
    assert event["destination_port"] == 80
    assert event["record"]["state"] == "FIN"
    assert event["record"]["spkts"] >= 4
    assert event["record"]["sttl"] == 62
    assert event["record"]["dttl"] == 250


def test_pcap_source_rejects_captures_without_ip_flows(tmp_path):
    empty = tmp_path / "empty.pcap"
    scapy_all.wrpcap(str(empty), [scapy_all.Ether() / scapy_all.ARP()])
    source = PcapReplaySource(str(empty))
    with pytest.raises(TrafficSourceError, match="no classifiable IP flows"):
        source.prepare()


def test_pcap_source_honors_cancellation_before_preparation(tmp_path):
    pcap_path = write_sample_pcap(tmp_path / "cancelled.pcap", sessions=1)
    cancel_event = threading.Event()
    cancel_event.set()

    source = PcapReplaySource(str(pcap_path), cancel_event=cancel_event)
    with pytest.raises(TrafficSourceCancelled, match="cancelled"):
        source.prepare()


def test_sequential_pcap_replay_does_not_buffer_the_full_capture(tmp_path):
    pcap_path = write_sample_pcap(tmp_path / "large.pcap", sessions=50)

    source = PcapReplaySource(str(pcap_path), order="sequential")
    source.prepare()

    assert source.row_total is None
    assert source.packets_read < 50
    assert source._flows == []
    source.close()


def test_csv_source_replays_rows_with_labels(tmp_path, trained_bundle):
    csv_path = tmp_path / "sample.csv"
    trained_bundle["frame"].to_csv(csv_path, index=False)

    source = CsvReplaySource(str(csv_path))
    source.prepare()
    assert source.row_total == len(trained_bundle["frame"])

    event = source.next_event()
    assert event["actual"] in {"Normal", "Attack"}
    assert event["source_ip"].startswith("10.")
    assert event["end_reason"] == "replay"


def test_live_capture_availability_reports_a_reason_when_unavailable():
    available, reason = live_capture_available()
    assert isinstance(available, bool)
    if not available:
        assert reason


def test_windows_without_pcap_provider_disables_live_capture(monkeypatch):
    from scapy.config import conf

    monkeypatch.setattr(traffic_sources.os, "name", "nt")
    monkeypatch.setattr(conf, "use_pcap", False)

    available, reason = live_capture_available()

    assert available is False
    assert "Npcap" in reason


def wait_for(condition, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture()
def deployed_stack(monkeypatch, trained_bundle):
    from services import database_service as db
    from services.deployment_service import deploy_model

    with db.get_connection() as connection:
        row = connection.execute(
            "SELECT model_id FROM detection_model WHERE run_id = ? AND model_name = ?",
            (trained_bundle["run_id"], "Stacking Ensemble"),
        ).fetchone()
    deploy_model(row["model_id"], 1)
    monkeypatch.setitem(monitor.SPEED_CHOICES, "fast", 0.001)
    return trained_bundle


def test_monitor_replays_a_pcap_end_to_end(monkeypatch, tmp_path, deployed_stack):
    write_sample_pcap(tmp_path / "office.pcap", sessions=3)
    monkeypatch.setattr(monitor, "CAPTURE_FOLDER", str(tmp_path))

    monitor.start_session(
        1, source_type="pcap", capture_file="office.pcap", speed="fast", persist="none"
    )
    assert wait_for(lambda: monitor.get_status()["session"]["state"] == "completed")

    status = monitor.get_status()
    session = status["session"]
    assert session["source_type"] == "pcap"
    assert session["row_total"] == 4
    assert session["totals"]["flows"] == 4
    assert session["totals"]["labelled"] == 0
    assert session["totals"]["mismatches"] == 0

    events = status["events"]
    assert all(event["actual"] is None for event in events)
    assert all(event["match"] is None for event in events)
    assert any(event["destination_port"] == 80 for event in events)
    assert all(event["prediction"] in {"Normal", "Attack"} for event in events)


def test_monitor_rejects_bad_capture_selections(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor, "CAPTURE_FOLDER", str(tmp_path))
    with pytest.raises(monitor.LiveMonitorError, match="captures folder"):
        monitor.start_session(1, source_type="pcap", capture_file="../evil.pcap")
    with pytest.raises(monitor.LiveMonitorError, match="recordings"):
        monitor.start_session(1, source_type="pcap", capture_file="notes.txt")
    with pytest.raises(monitor.LiveMonitorError, match="missing"):
        monitor.start_session(1, source_type="pcap", capture_file="ghost.pcap")
    with pytest.raises(monitor.LiveMonitorError, match="source type"):
        monitor.start_session(1, source_type="telepathy")


def test_capture_filter_excludes_the_application_port(monkeypatch):
    from services.traffic_source_service import algoguard_port, build_capture_filter

    monkeypatch.setenv("ALGOGUARD_PORT", "5000")
    assert algoguard_port() == 5000
    assert build_capture_filter({5000}) == "ip and (tcp or udp) and not port 5000"
    assert build_capture_filter({}) == "ip and (tcp or udp)"

    monkeypatch.setenv("ALGOGUARD_PORT", "not-a-port")
    assert algoguard_port() == 5000  # falls back instead of crashing capture


def test_live_source_defaults_to_excluding_algoguards_own_traffic(monkeypatch):
    from services.traffic_source_service import LiveCaptureSource

    monkeypatch.setenv("ALGOGUARD_PORT", "5077")
    source = LiveCaptureSource("lo")
    assert source.exclude_ports == {5077}
    assert "not port 5077" in source.bpf_filter


def test_own_traffic_is_dropped_even_when_the_bpf_filter_did_not_apply():
    """The Python-side guard is what protects unfiltered fallback captures."""
    from scapy.all import IP, TCP, Ether

    from services.traffic_source_service import LiveCaptureSource

    source = LiveCaptureSource("lo", exclude_ports={5000})

    own = Ether() / IP(src="127.0.0.1", dst="127.0.0.1") / TCP(sport=44444, dport=5000, flags="S")
    other = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=44444, dport=80, flags="S")
    own.time = other.time = 1_700_000_000.0

    source._on_packet(own)
    assert source.next_event(timeout=0.1) is None
    assert source.packets_excluded == 1
    assert source.packets_captured == 0

    source._on_packet(other)
    source.next_event(timeout=0.1)
    assert source.packets_captured == 1
    assert source.stats()["excluded"] == 1


def test_monitor_options_advertise_all_source_types():
    options = monitor.get_options()
    values = {item["value"] for item in options["sources"]}
    assert values == {"csv", "pcap", "live"}
    assert "live_capture" in options
    assert isinstance(options["captures"], list)
