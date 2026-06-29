from collections import defaultdict


def build_flows(capture):

    flows = defaultdict(lambda: {

        "start_time": None,
        "end_time": None,

        "spkts": 0,
        "dpkts": 0,

        "sbytes": 0,
        "dbytes": 0,

        "sttl": [],
        "dttl": []

    })

    for packet in capture:

        try:

            src = packet.ip.src
            dst = packet.ip.dst

            proto = packet.transport_layer

            src_port = getattr(packet[proto], "srcport", "0")
            dst_port = getattr(packet[proto], "dstport", "0")

            flow = (
                src,
                dst,
                src_port,
                dst_port,
                proto
            )

            reverse = (
                dst,
                src,
                dst_port,
                src_port,
                proto
            )

            timestamp = packet.sniff_timestamp
            length = int(packet.length)
            ttl = int(packet.ip.ttl)

            if flow in flows:

                f = flows[flow]

                f["spkts"] += 1
                f["sbytes"] += length
                f["sttl"].append(ttl)

            elif reverse in flows:

                f = flows[reverse]

                f["dpkts"] += 1
                f["dbytes"] += length
                f["dttl"].append(ttl)

            else:

                f = flows[flow]

                f["start_time"] = float(timestamp)

                f["spkts"] = 1
                f["sbytes"] = length
                f["sttl"] = [ttl]

            f["end_time"] = float(timestamp)

        except Exception:
            continue

    return flows