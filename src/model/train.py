import argparse
from ultralytics import YOLO


def train_model(model_architecture: str, data_dir: str, name: str, batch: int, device):
    model = YOLO(f"{model_architecture}.pt")
    results = model.train(
        data=f"{data_dir}/data.yaml",
        epochs=500,
        patience=20,
        batch=batch,
        imgsz=640,
        save=True,
        device=device,
        workers=8,
        project="./assets/models",
        name=name,
        classes=[0, 1],
        close_mosaic=False,
        lr0=1e-4,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 models")

    parser.add_argument(
        "-a", "--model_architecture", type=str, default="yolov8m",
        help="YOLO model architecture (e.g., yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)"
    )
    parser.add_argument(
        "-t", "--model_type", type=str, choices=["metadata", "content"], required=True,
        help="Select training mode: metadata or content"
    )
    parser.add_argument(
        "-d", "--data_dir", type=str, required=True,
        help="Path to dataset directory containing data.yaml"
    )
    args = parser.parse_args()

    if args.model_type == "metadata":
        yolov8_metadata = train_model(
            args.model_architecture,
            args.data_dir,
            name=f"metadata_{args.model_architecture}",
            batch=32,
            device=[0, 1],
        )
    elif args.model_type == "content":
        yolov8_content = train_model(
            args.model_architecture,
            args.data_dir,
            name=f"content_{args.model_architecture}",
            batch=16,
            device=0,
        )
