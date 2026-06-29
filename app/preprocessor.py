import pandas as pd
import numpy as np
import joblib


def preprocess_flows(flows):

    # Load encoders
    proto_encoder = joblib.load("models/proto_encoder.pkl")
    state_encoder = joblib.load("models/state_encoder.pkl")

    records = []

    for flow_key, flow in flows.items():

        src_ip, dst_ip, src_port, dst_port, proto = flow_key

        # Duration
        dur = max(flow["end_time"] - flow["start_time"], 0.000001)

        # Packet rate
        rate = (flow["spkts"] + flow["dpkts"]) / dur

        # Average TTL
        sttl = np.mean(flow["sttl"]) if flow["sttl"] else 0
        dttl = np.mean(flow["dttl"]) if flow["dttl"] else 0

        # Approximate UNSW state
        if proto.upper() == "TCP":
            state = "CON"
        else:
            state = "INT"

        records.append({
            "dur": dur,
            "proto": proto.lower(),
            "state": state,
            "spkts": flow["spkts"],
            "dpkts": flow["dpkts"],
            "sbytes": flow["sbytes"],
            "dbytes": flow["dbytes"],
            "rate": rate,
            "sttl": sttl,
            "dttl": dttl
        })

    # Create DataFrame
    df = pd.DataFrame(records)

    # Handle unknown protocols safely
    df["proto"] = df["proto"].apply(
        lambda x: x if x in proto_encoder.classes_ else "tcp"
    )

    df["proto"] = proto_encoder.transform(df["proto"])

    # Handle unknown states safely
    df["state"] = df["state"].apply(
        lambda x: x if x in state_encoder.classes_ else "INT"
    )

    df["state"] = state_encoder.transform(df["state"])

    return df