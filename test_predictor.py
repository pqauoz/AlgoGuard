from app.feature_extractor import read_capture
from app.flow_builder import build_flows
from app.preprocessor import preprocess_flows
from app.predictor import predict

print("=" * 60)
print("AlgoGuard Prediction Test")
print("=" * 60)

capture = read_capture("captures/small_local_traffic.pcapng")

flows = build_flows(capture)

features = preprocess_flows(flows)

predictions, probabilities = predict(features)

print(f"\nTotal Flows: {len(predictions)}")

print("\nPredictions:")

for i, pred in enumerate(predictions, start=1):

    label = "Attack" if pred == 1 else "Normal"

    if probabilities is not None:
        confidence = probabilities[i-1].max() * 100
        print(f"Flow {i:02d}: {label:7} ({confidence:.2f}%)")
    else:
        print(f"Flow {i:02d}: {label}")