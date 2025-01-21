from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
