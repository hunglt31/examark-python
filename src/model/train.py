from ultralytics import YOLO

def train_metadata_model():
    model = YOLO("yolov8m.pt") 
    results = model.train(
        data="./dataset/metadata/data.yaml", 
        epochs=500,
        patience=20,
        batch=32,
        imgsz=640,
        save=True,
        device=[0, 1],
        workers=8,
        project="./assets/models",
        name="metadata_yolov8m",
        classes=[0, 1],
        close_mosaic=False,
        lr0=1e-4,
        )
    return results

def train_content_model():
    model = YOLO("yolov8m.pt") 
    results = model.train(
        data="./dataset/content/data.yaml", 
        epochs=500,
        patience=20,
        batch=16,
        imgsz=640,
        save=True,
        device=0,
        workers=8,
        project="./assets/models",
        name="content_yolov8m",
        classes=[0, 1],
        close_mosaic=False,
        lr0=1e-4,
        )
    return results

if __name__ == "__main__":
    yolov8m_metadata = train_metadata_model()
    yolov8m_content = train_content_model()