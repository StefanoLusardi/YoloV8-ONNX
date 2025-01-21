import onnxruntime as ort
import numpy as np

# Load the ONNX model
onnx_model_path = "yolov8n.onnx"
session = ort.InferenceSession(onnx_model_path)

# Prepare a sample input (ensure it matches the model's input size, e.g., 640x640)
input_shape = (1, 3, 640, 640)  # Batch size, channels, height, width
sample_input = np.random.random(input_shape).astype(np.float32)

# Run inference
outputs = session.run(None, {"images": sample_input})

# Print the output for verification
print("Inference outputs:", outputs)
