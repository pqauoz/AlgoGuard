from feature_extractor import read_capture
from flow_builder import build_flows

capture = read_capture(
    "captures/small_local_traffic.pcapng"
)

flows = build_flows(capture)

print("=" * 60)
print("Number of flows:", len(flows))
print("=" * 60)

for key, value in list(flows.items())[:5]:

    print()

    print("Flow")

    print(key)

    print(value)