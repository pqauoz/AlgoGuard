from app.feature_extractor import read_capture
from app.flow_builder import build_flows
from app.preprocessor import preprocess_flows

capture = read_capture("captures/small_local_traffic.pcapng")

flows = build_flows(capture)

features = preprocess_flows(flows)

print(features.head())

print("\nShape:", features.shape)