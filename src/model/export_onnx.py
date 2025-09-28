from ultralytics import YOLO

metadata_model = YOLO("./assets/models/metadata/weights/best.pt")
metadata_model.export(format="onnx", dynamic=True, simplify=True, opset=12) 

# content_model = YOLO("./assets/models/content/weights/best.pt")
# content_model.export(format="onnx", dynamic=True, simplify=True, opset=12) 