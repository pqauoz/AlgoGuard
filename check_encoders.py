import joblib

proto_encoder = joblib.load("models/proto_encoder.pkl")
state_encoder = joblib.load("models/state_encoder.pkl")

print("PROTO CLASSES:")
print(proto_encoder.classes_)

print("\nSTATE CLASSES:")
print(state_encoder.classes_)